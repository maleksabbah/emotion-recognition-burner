"""
BurnService — fetch source → annotate → save → result.

One method, top-down.
"""
from __future__ import annotations

import logging
import time

from app.Config import WORKER_ID
from app.Dtos.TaskDto.BurnResult import BurnResult
from app.Dtos.TaskDto.BurnTask import BurnTask
from app.Repositories.S3Client import S3Client
from app.Repositories.StorageClient import StorageClient
from app.Services.Annotator import annotate_image, stitch_video

logger = logging.getLogger("burner.service")


class BurnService:
    def __init__(self, s3: S3Client, storage: StorageClient):
        self.s3 = s3
        self.storage = storage

    async def process_task(self, task: BurnTask) -> BurnResult:
        start = time.perf_counter()

        # 1. Fetch source from MinIO
        try:
            source_bytes = await self.s3.fetch_object(task.source_s3_key)
        except Exception as e:
            return self._failed(task, f"source fetch failed: {e}", start)

        # 2. Convert typed predictions → annotator's dict shape
        annotations: dict[int, list[dict]] = {}
        for frame in task.frames:
            annotations[frame.frame_number] = [
                {
                    "bbox": [p.bbox.x, p.bbox.y, p.bbox.x + p.bbox.w, p.bbox.y + p.bbox.h],
                    "predictions": {
                        "emotion":   {"label": p.top_emotion, "confidence": p.top_confidence},
                        "valence":   {"label": p.valence},
                        "arousal":   {"label": p.arousal},
                        "intensity": {"label": p.intensity},
                    },
                }
                for p in frame.predictions
            ]

        # 3. Annotate — video or photo
        try:
            if task.mode == "video":
                burned_bytes = stitch_video(source_bytes, annotations)
                mime_type = "video/mp4"
            else:
                # Photo: only one frame, use its predictions if any
                first_frame_preds = next(iter(annotations.values()), [])
                burned_bytes = annotate_image(source_bytes, first_frame_preds)
                mime_type = "image/jpeg"
        except Exception as e:
            return self._failed(task, f"annotate failed: {e}", start)

        # 4. POST to storage (which writes to MinIO + records FileRecord)
        try:
            saved = await self.storage.save_output(
                session_id=task.session_id,
                data=burned_bytes,
                mime_type=mime_type,
                file_type="burned",
            )
        except Exception as e:
            return self._failed(task, f"storage save failed: {e}", start)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return BurnResult(
            task_id=task.task_id,
            session_id=task.session_id,
            status="complete",
            burned_s3_key=saved.get("s3_key"),
            file_id=saved.get("file_id"),
            worker_id=WORKER_ID,
            processing_time_ms=elapsed_ms,
        )

    def _failed(self, task: BurnTask, error: str, start: float) -> BurnResult:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.error("burn failed: %s", error)
        return BurnResult(
            task_id=task.task_id,
            session_id=task.session_id,
            status="failed",
            error=error,
            worker_id=WORKER_ID,
            processing_time_ms=elapsed_ms,
        )
