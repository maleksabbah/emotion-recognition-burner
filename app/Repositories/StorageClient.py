"""
StorageClient — burner → storage HTTP boundary.

The burner doesn't write to MinIO directly. It POSTs the burned bytes
(base64) to storage's /internal/save-output, which writes to MinIO AND
records the FileRecord. Single source of truth.
"""
from __future__ import annotations

import base64
import logging

import httpx

from app.Config import STORAGE_SERVICE_URL

logger = logging.getLogger("burner.storage")


class StorageClient:
    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()

    async def save_output(
        self,
        session_id: str,
        data: bytes,
        mime_type: str,
        file_type: str = "burned",
    ) -> dict:
        if not self._http:
            raise RuntimeError("Call start() first")

        body = {
            "session_id": session_id,
            "data": base64.b64encode(data).decode("ascii"),
            "mime_type": mime_type,
            "file_type": file_type,
        }
        resp = await self._http.post(
            f"{STORAGE_SERVICE_URL}/internal/save-output", json=body
        )
        if resp.status_code != 200:
            logger.error(
                "save_output failed %s: %s", resp.status_code, resp.text
            )
            resp.raise_for_status()
        return resp.json()