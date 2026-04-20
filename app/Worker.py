"""
Burner Worker — stateless Kafka consumer.

Consumes burn tasks from the orchestrator. Source and output paths are
assigned by the orchestrator; the burner just executes.

Task schema (from orchestrator):
{
    "session_id": str,
    "task_type": "frame" | "video",
    "worker_id": str,
    "source_s3_key": str,          # where to read the source from MinIO
    "output_s3_key": str,          # where storage should write the annotated output
    "detections": [...]            # for frame tasks
    "frame_annotations": {...}     # for video tasks
}

Result schema (to orchestrator):
{
    "session_id": str,
    "task_type": "frame" | "video",
    "worker_id": str,
    "status": "success" | "error",
    "file_id": str | null,         # storage's record id for the saved output
    "output_s3_key": str | null,
    "processing_ms": float,
    "error": str | null
}
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import signal
import time
from typing import Any

import boto3
import httpx

from app.Config import WORKER_ID
from app.Kafka import create_consumer, create_producer, publish_result
from app.Stitcher import annotate_single_frame, stitch_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("burner-worker")


# ── S3 / MinIO ──────────────────────────────────────────

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "emotion-recognition")
STORAGE_SERVICE_URL = os.getenv("STORAGE_SERVICE_URL", "http://storage:8002")


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def _download_source(s3_key: str) -> bytes:
    """Download the source file for this burn task from MinIO."""
    s3 = _s3_client()
    actual_key = s3_key
    if s3_key.endswith("/"):
        objs = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_key, MaxKeys=1)
        if "Contents" not in objs or not objs["Contents"]:
            raise ValueError(f"No objects under prefix {s3_key}")
        actual_key = objs["Contents"][0]["Key"]
    obj = s3.get_object(Bucket=S3_BUCKET, Key=actual_key)
    return obj["Body"].read()


# ── Worker ──────────────────────────────────────────────

class BurnerWorker:
    def __init__(self):
        self.consumer = None
        self.producer = None
        self._running = False

    async def start(self):
        self.consumer = await create_consumer()
        self.producer = await create_producer()
        self._running = True
        logger.info("Burner worker %s started", WORKER_ID)

    async def stop(self):
        self._running = False
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        logger.info("Burner worker %s stopped", WORKER_ID)

    # ── Annotation (pure CPU work) ─────────────────────

    def _annotate_frame(self, source_bytes: bytes, detections: list) -> bytes:
        return annotate_single_frame(source_bytes, detections)

    def _annotate_video(self, source_bytes: bytes, raw_annotations: dict) -> bytes:
        # JSON turns int keys into strings; flip them back
        frame_annotations = {int(k): v for k, v in raw_annotations.items()}
        return stitch_video(source_bytes, frame_annotations)

    # ── Storage handoff ────────────────────────────────

    async def _save_to_storage(
        self,
        session_id: str,
        output_s3_key: str,
        data: bytes,
        mime_type: str,
        file_type: str,
    ) -> str:
        """POST bytes to storage /internal/save-output. Returns file_id."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{STORAGE_SERVICE_URL}/internal/save-output",
                json={
                    "session_id": session_id,
                    "category": "burned",
                    "file_type": file_type,
                    "s3_key": output_s3_key,
                    "mime_type": mime_type,
                    "data_b64": base64.b64encode(data).decode("utf-8"),
                },
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Storage save-output failed {resp.status_code}: {resp.text}")
            return resp.json()["id"]

    # ── Task handling ──────────────────────────────────

    async def _handle_task(self, task: dict[str, Any]) -> dict[str, Any]:
        session_id = task.get("session_id", "unknown")
        task_type = task.get("task_type", "unknown")
        source_key = task.get("source_s3_key")
        output_key = task.get("output_s3_key")

        start = time.perf_counter()

        try:
            if not source_key or not output_key:
                raise ValueError("Task missing source_s3_key or output_s3_key")

            # 1. Pull source from MinIO
            source_bytes = await asyncio.get_event_loop().run_in_executor(
                None, _download_source, source_key
            )

            # 2. Annotate
            if task_type == "video":
                mime = "video/mp4"
                file_type = "video"
                annotations = task.get("frame_annotations", {})
                output_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._annotate_video, source_bytes, annotations
                )
            else:
                mime = "image/jpeg"
                file_type = "image"
                detections = task.get("detections", [])
                output_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._annotate_frame, source_bytes, detections
                )

            # 3. Hand off to storage (it writes MinIO + registers)
            file_id = await self._save_to_storage(
                session_id=session_id,
                output_s3_key=output_key,
                data=output_bytes,
                mime_type=mime,
                file_type=file_type,
            )

            elapsed = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "Burn complete: session=%s type=%s out=%s file_id=%s (%dms)",
                session_id, task_type, output_key, file_id, elapsed,
            )
            return {
                "session_id": session_id,
                "task_type": task_type,
                "worker_id": WORKER_ID,
                "status": "success",
                "file_id": file_id,
                "output_s3_key": output_key,
                "processing_ms": elapsed,
                "error": None,
            }

        except Exception as e:
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            logger.error("Burn error session=%s type=%s: %s", session_id, task_type, e, exc_info=True)
            return {
                "session_id": session_id,
                "task_type": task_type,
                "worker_id": WORKER_ID,
                "status": "error",
                "file_id": None,
                "output_s3_key": output_key,
                "processing_ms": elapsed,
                "error": str(e),
            }

    async def run(self):
        await self.start()
        try:
            async for message in self.consumer:
                if not self._running:
                    break
                task = message.value
                logger.info(
                    "Received burn task: session=%s type=%s",
                    task.get("session_id"),
                    task.get("task_type"),
                )
                result = await self._handle_task(task)
                await publish_result(self.producer, result)
                await self.consumer.commit()
        except asyncio.CancelledError:
            logger.info("Worker loop cancelled")
        finally:
            await self.stop()


async def main():
    worker = BurnerWorker()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(worker)))
    await worker.run()


async def _shutdown(worker: BurnerWorker):
    logger.info("Shutdown signal received")
    worker._running = False


if __name__ == "__main__":
    asyncio.run(main())
