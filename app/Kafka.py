"""
Kafka helpers — consume burn_tasks, publish to burn_results.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.Config import (
    KAFKA_BOOTSTRAP,
    KAFKA_GROUP_ID,
    TOPIC_BURN_TASKS,
    TOPIC_BURN_RESULTS,
)

logger = logging.getLogger("burner.kafka")


async def create_consumer() -> AIOKafkaConsumer:
    consumer = AIOKafkaConsumer(
        TOPIC_BURN_TASKS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_interval_ms=600_000,
    )
    await consumer.start()
    logger.info("Kafka consumer started on topic=%s", TOPIC_BURN_TASKS)
    return consumer


async def create_producer() -> AIOKafkaProducer:
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()
    logger.info("Kafka producer started")
    return producer


async def publish_result(producer: AIOKafkaProducer, result: dict[str, Any]) -> None:
    await producer.send_and_wait(
        TOPIC_BURN_RESULTS,
        value=result,
        headers=[("worker_id", result.get("worker_id", "unknown").encode())],
    )
    logger.debug("Published burn result for session=%s", result.get("session_id"))