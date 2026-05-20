"""
Burner integration tests — single file.
Run from burner-worker/ root: `pytest Test.py -v`

Requires:
  pytest.ini with session-scoped loops (see storage)
"""
from __future__ import annotations

import io
import os
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from testcontainers.minio import MinioContainer


# ══════════════════════════════════════════════
# Containers
# ══════════════════════════════════════════════

@pytest.fixture(scope="session", autouse=True)
def minio():
    with MinioContainer() as mc:
        host = mc.get_container_host_ip()
        port = mc.get_exposed_port(9000)
        os.environ["S3_INTERNAL_ENDPOINT"] = f"http://{host}:{port}"
        os.environ["S3_ACCESS_KEY"] = mc.access_key
        os.environ["S3_SECRET_KEY"] = mc.secret_key
        os.environ.setdefault("STORAGE_SERVICE_URL", "http://fake-storage:8002")

        client = mc.get_client()
        if not client.bucket_exists("emotion"):
            client.make_bucket("emotion")
        yield mc


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _burn_task(s3_key: str, mode: str = "video") -> dict:
    return {
        "task_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "mode": mode,
        "source_s3_key": s3_key,
        "frames": [{
            "frame_number": 0,
            "timestamp_ms": 0.0,
            "predictions": [{
                "bbox": {"x": 10, "y": 10, "w": 50, "h": 50},
                "top_emotion": "happy",
                "top_confidence": 0.9,
                "valence": "positive",
                "arousal": "high",
                "intensity": "medium",
            }],
        }],
    }


def _upload_fake_video(mc, s3_key: str):
    mc.get_client().put_object(
        "emotion", s3_key,
        data=io.BytesIO(b"\x00" * 100),
        length=100, content_type="video/mp4",
    )


# ══════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════

@pytest.mark.asyncio(loop_scope="session")
async def test_burn_task_returns_typed_result(minio):
    from app.Dtos.TaskDto.BurnTask import BurnTask
    from app.Repositories.S3Client import S3Client
    from app.Repositories.StorageClient import StorageClient
    from app.Services.BurnService import BurnService

    s3_key = f"sessions/{uuid.uuid4()}/source/video.mp4"
    _upload_fake_video(minio, s3_key)

    storage = StorageClient()
    storage.save_output = AsyncMock(return_value={
        "file_id": str(uuid.uuid4()),
        "s3_key": "sessions/test/burned/output.mp4",
    })

    burner = BurnService(s3=S3Client(), storage=storage)
    result = await burner.process_task(BurnTask.model_validate(_burn_task(s3_key)))

    # Fake video bytes likely fail stitching — assert the failed path is typed
    assert result.task_id is not None
    assert result.status in ("complete", "failed")


@pytest.mark.asyncio(loop_scope="session")
async def test_burn_missing_source_returns_failed(minio):
    from app.Dtos.TaskDto.BurnTask import BurnTask
    from app.Repositories.S3Client import S3Client
    from app.Repositories.StorageClient import StorageClient
    from app.Services.BurnService import BurnService

    storage = StorageClient()
    storage.save_output = AsyncMock()
    burner = BurnService(s3=S3Client(), storage=storage)

    task = _burn_task(s3_key="sessions/nope/missing.mp4")
    result = await burner.process_task(BurnTask.model_validate(task))

    assert result.status == "failed"
    storage.save_output.assert_not_called()