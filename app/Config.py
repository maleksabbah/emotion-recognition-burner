"""
Burner worker configuration — env vars loaded once.
"""
from __future__ import annotations

import os
import uuid


# ─── Identity ──────────────────────────────────────────────────────────

WORKER_ID = os.getenv("WORKER_ID", f"burner-{uuid.uuid4().hex[:8]}")


# ─── Kafka ─────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
BURN_TASKS_TOPIC = os.getenv("BURN_TASKS_TOPIC", "burn_tasks")
BURN_RESULTS_TOPIC = os.getenv("BURN_RESULTS_TOPIC", "burn_results")
BURN_GROUP_ID = os.getenv("BURN_GROUP_ID", "burner-workers")


# ─── S3 / MinIO (source fetch) ─────────────────────────────────────────

S3_INTERNAL_ENDPOINT = os.getenv("S3_INTERNAL_ENDPOINT", "http://minio:9000")
S3_BUCKET = os.getenv("S3_BUCKET", "emotion")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")


# ─── Storage service (POST burned output) ─────────────────────────────

STORAGE_SERVICE_URL = os.getenv("STORAGE_SERVICE_URL", "http://storage:8002")