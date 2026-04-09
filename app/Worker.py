"""
Burner Worker — stateless Kafka consumer.

Consumes burn tasks from the orchestrator, annotates frames or video,
publishes results back.

Task schema (from orchestrator):
{
    "session_id": str,
    "task_type": "frame" | "video",
    "worker_id": str,

    # For frame tasks:
    "frame_b64": str,              # base64 JPEG
    "detections": [{bbox, predictions}],

    # For video tasks:
    "video_b64": str,              # base64 MP4
    "frame_annotations": {frame_idx: [detections]}
}

Result schema:
{
    "session_id": str,
    "task_type": "frame" | "video",
    "worker_id": str,
    "status": "success" | "error",
    "output_b64": str,             # base64 annotated JPEG or MP4
    "processing_ms": float,
    "error": str | null
}
"""
from __future__ import annotations

import asyncio
import base64
import logging
import signal
import time
from typing import Any

from app.Config import WORKER_ID
from app.Kafka import create_consumer, create_producer, publish_result
from app.Stitcher import annotate_single_frame, stitch_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("burner-worker")

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

    def _process_frame_task(self,task:dict[str,Any]) -> bytes:
        """Annotate a single frame. Returns JPEG bytes."""
        frame_bytes = base64.b64decode(task["frame_b64"])
        detections = task.get("detections", [])
        return annotate_single_frame(frame_bytes, detections)
    def _process_video_task(self, task: dict[str, Any]) -> bytes:
        """Annotate a full video. Returns MP4 bytes."""
        video_bytes = base64.b64decode(task["video_b64"])

        # Convert string keys back to int (JSON serialization turns them to strings)

        raw_annotations = task.get("frame_annotations", {})
        frame_annotations = {int(k):v for k,v in raw_annotations.items()}

        return stitch_video(video_bytes, frame_annotations)
    async def _handle_task(self, task:dict[str, Any]) -> dict[str, Any]:
        """Process a single burn task."""
        session_id = task.get("session_id","unknown")
        task_type = task.get("task_type","unknown")

        start = time.perf_counter()

        try:
            if task_type == "video":
                output_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_video_task, task
                )
            else:
                output_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_frame_task, task
                )
            elapsed = round((time.perf_counter() - start) * 1000,1)
            output_b64 = base64.b64encode(output_bytes).decode("utf-8")

            return {
                "session_id": session_id,
                "task_type": task_type,
                "worker_id": WORKER_ID,
                "status": "success",
                "output_b64": output_b64,
                "processing_ms": elapsed,
                "error": None,
            }
        except Exception as e:
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            logger.error("Error on session=%s type=%s: %s", session_id, task_type, e)
            return {
                "session_id": session_id,
                "task_type": task_type,
                "worker_id": WORKER_ID,
                "status": "error",
                "output_b64": None,
                "processing_ms": elapsed,
                "error": str(e),
            }

    async def run(self):
        """Main consumer loop."""
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

async def _shutdown(worker:BurnerWorker):
    logger.info("Shutdown signal received")
    worker._running = False

if __name__ == "__main__":
    asyncio.run(main())
























