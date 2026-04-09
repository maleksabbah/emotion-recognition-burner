"""
Stitcher — processes video files by annotating each frame.

Takes:
  - Input video bytes
  - Frame-level detection + prediction data
Produces:
  - Annotated output video bytes (MP4)
"""
from __future__ import annotations

import logging
import tempfile
import os
from typing import Any

import cv2
import numpy as np

from app.Overlay import annotate_frame

logger = logging.getLogger("burner.stitcher")

def stitch_video(
        video_bytes: bytes,
        frame_annotations: dict[int,list[dict[str,Any]]],
        output_codec: str = "mp4v",
) -> bytes:
    """
        Read input video, annotate each frame, write output video.

        Args:
            video_bytes: raw input video file bytes
            frame_annotations: {frame_index: [detection_dicts]}
            output_codec: fourcc codec string

        Returns:
            Annotated video as bytes
        """
    # Write input to temp file (OpenCV needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        input_path = tmp_in.name

    output_path = input_path.replace(".mp4", "_annotated.mp4")

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Unable to open input video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*output_codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise ValueError("Failed to create output video writer")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break


            # Annotate if we have data for this frame
            if frame_idx in frame_annotations:
                annotate_frame(frame, frame_annotations[frame_idx])

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        logger.info(
            "Stitched %d frames (%d annotated) at %dfps %dx%d",
            frame_idx,
            len(frame_annotations),
            int(fps),
            width,
            height,
        )

        # Read output
        with open(output_path,"rb") as f:
            return f.read()
    finally:
        # Cleanup temp files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

def annotate_single_frame(
        frame_bytes: bytes,
        detections: list[dict[str,Any]]
) -> bytes:
    """
    Annotate a single frame (JPEG/PNG bytes).
    Returns annotated JPEG bytes.
    """
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode")

    annotate_frame(frame, detections)

    _, buf = cv2.imencode(".jpg", frame,[cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()









