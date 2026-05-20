"""
S3Client — boto3 wrapper to fetch the source video/photo for burning.

Burner reads from MinIO using the s3_key the orchestrator passes in the
BurnTask. Writes go through StorageClient (not directly), so storage stays
the single registry.
"""
from __future__ import annotations

import asyncio
import logging

import boto3
from botocore.client import Config as BotoConfig

from app.Config import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_INTERNAL_ENDPOINT,
    S3_REGION,
    S3_SECRET_KEY,
)

logger = logging.getLogger("burner.s3")


class S3Client:
    def __init__(self) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=S3_INTERNAL_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
            config=BotoConfig(signature_version="s3v4"),
        )

    async def fetch_object(self, s3_key: str) -> bytes:
        def _go() -> bytes:
            resp = self._client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            return resp["Body"].read()
        return await asyncio.to_thread(_go)