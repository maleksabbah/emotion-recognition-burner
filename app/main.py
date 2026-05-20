"""
Burner worker entry point.

One Kafka loop: consume burn_tasks → BurnService.process_task → publish
burn_results. No live path — burning is video/photo (batch) only.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from app.Config import WORKER_ID
from app.Dtos.TaskDto.BurnTask import BurnTask
from app.Repositories.KafkaConsumer import KafkaConsumer
from app.Repositories.KafkaProducer import KafkaProducer
from app.Repositories.S3Client import S3Client
from app.Repositories.StorageClient import StorageClient
from app.Services.BurnService import BurnService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("burner")


async def kafka_loop(
    consumer: KafkaConsumer,
    producer: KafkaProducer,
    burner: BurnService,
) -> None:
    async for raw in consumer.consume():
        try:
            task = BurnTask.model_validate(raw)
        except Exception as e:
            logger.error("Bad BurnTask: %s", e)
            continue

        try:
            result = await burner.process_task(task)
        except Exception as e:
            logger.exception("process_task crashed: %s", e)
            continue

        await producer.publish_burn_result(result.model_dump())


async def run() -> None:
    logger.info("Burner starting (id=%s)", WORKER_ID)

    s3 = S3Client()
    storage = StorageClient()
    consumer = KafkaConsumer()
    producer = KafkaProducer()

    await storage.start()
    await consumer.start()
    await producer.start()
    logger.info("Burner ready")

    burner = BurnService(s3=s3, storage=storage)

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    task = asyncio.create_task(kafka_loop(consumer, producer, burner))

    await stop.wait()
    logger.info("Shutting down...")

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    await consumer.stop()
    await producer.stop()
    await storage.stop()
    logger.info("Burner stopped")


if __name__ == "__main__":
    asyncio.run(run())


