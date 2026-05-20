"""
Annotator — draw emotion overlays on video frames or single images.

  annotate_frame(frame, detections)         per-frame drawing (in place)
  stitch_video(video_bytes, annotations)    annotate every frame of a video
  annotate_image(image_bytes, detections)   annotate one photo

Style:
  - one bounding box per face, colored by valence
  - one evocative WORD below the face (ELATED / ENRAGED / SERENE / etc.)
  - confidence as a small decimal underneath
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("burner.annotator")

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Emotion crosstab: (emotion, valence, arousal) → display word ─────

_CROSSTAB: dict[tuple[str, str, str], str] = {
    ("happy",    "positive", "high"): "ELATED",
    ("happy",    "positive", "low"):  "CONTENT",
    ("happy",    "neutral",  "high"): "AMUSED",
    ("happy",    "neutral",  "low"):  "PLEASED",
    ("neutral",  "neutral",  "low"):  "STILL",
    ("neutral",  "positive", "low"):  "SERENE",
    ("neutral",  "negative", "low"):  "BLANK",
    ("neutral",  "neutral",  "high"): "ALERT",
    ("sad",      "negative", "low"):  "MELANCHOLY",
    ("sad",      "negative", "high"): "ANGUISHED",
    ("sad",      "neutral",  "low"):  "WISTFUL",
    ("angry",    "negative", "high"): "ENRAGED",
    ("angry",    "negative", "low"):  "BITTER",
    ("angry",    "neutral",  "high"): "TENSE",
    ("fear",     "negative", "high"): "TERRIFIED",
    ("fear",     "negative", "low"):  "UNEASY",
    ("fear",     "neutral",  "high"): "STARTLED",
    ("disgust",  "negative", "high"): "REPULSED",
    ("disgust",  "negative", "low"):  "DISDAINFUL",
    ("surprise", "positive", "high"): "AWESTRUCK",
    ("surprise", "neutral",  "high"): "STARTLED",
    ("surprise", "negative", "high"): "SHOCKED",
}

_VALENCE_COLOR = {
    "positive": (90, 160, 240),    # warm amber  (BGR)
    "negative": (180, 120, 80),    # cool slate
}
_NEUTRAL_COLOR = (220, 220, 210)


# ──────────────────────────────────────────────────────────────────────
# Per-frame drawing
# ──────────────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, detections: list[dict[str, Any]]) -> None:
    """Draw bbox + word + confidence for every detection. Mutates frame."""
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = map(int, det.get("bbox", [0, 0, 100, 100]))
        preds = det.get("predictions", {})

        emotion = preds.get("emotion", {}).get("label", "unknown").lower()
        confidence = float(preds.get("emotion", {}).get("confidence", 0.0))
        valence = preds.get("valence", {}).get("label", "neutral").lower()
        arousal = preds.get("arousal", {}).get("label", "low").lower()

        # 1. Box — color by valence, thickness by arousal
        color = _VALENCE_COLOR.get(valence, _NEUTRAL_COLOR)
        thickness = 3 if arousal == "high" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 2. Word — pick from crosstab, scale to face width
        word = _CROSSTAB.get((emotion, valence, arousal)) or _fallback_word(emotion, arousal)
        face_w = x2 - x1
        word_scale = max(0.9, min(2.4, face_w / 260.0))
        word_thick = max(2, int(word_scale * 2))
        (tw, th), _ = cv2.getTextSize(word, FONT, word_scale, word_thick)
        word_x = max(12, min(w - tw - 12, x1 + (face_w - tw) // 2))
        word_y = min(h - 12, y2 + th + 20)
        _draw_text(frame, word, (word_x, word_y), (240, 240, 232), word_scale, word_thick)

        # 3. Confidence — small decimal under the word
        conf_text = f".{int(round(confidence * 100)):02d}"
        conf_scale = max(0.5, word_scale * 0.42)
        conf_thick = max(1, int(conf_scale * 2))
        (cw, ch), _ = cv2.getTextSize(conf_text, FONT, conf_scale, conf_thick)
        conf_x = max(12, min(w - cw - 12, x1 + (face_w - cw) // 2))
        conf_y = min(h - 6, word_y + ch + 14)
        _draw_text(frame, conf_text, (conf_x, conf_y), (170, 170, 160), conf_scale, conf_thick)


def _draw_text(
    frame: np.ndarray, text: str, anchor: tuple[int, int],
    color: tuple[int, int, int], scale: float, thickness: int,
) -> None:
    """Text with a soft shadow for legibility on any background."""
    x, y = anchor
    cv2.putText(frame, text, (x + 2, y + 2), FONT, scale, (20, 20, 20), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _fallback_word(emotion: str, arousal: str) -> str:
    """If exact crosstab miss: pick any word with same emotion + arousal."""
    for (e, _v, a), word in _CROSSTAB.items():
        if e == emotion and a == arousal:
            return word
    return emotion.upper()


# ──────────────────────────────────────────────────────────────────────
# Video stitching
# ──────────────────────────────────────────────────────────────────────

def stitch_video(
    video_bytes: bytes,
    frame_annotations: dict[int, list[dict[str, Any]]],
    output_codec: str = "mp4v",
) -> bytes:
    """Decode video → annotate every frame that has detections → re-encode."""
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

        fourcc = cv2.VideoWriter_fourcc(*output_codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError("Failed to create output video writer")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_annotations:
                annotate_frame(frame, frame_annotations[frame_idx])
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        logger.info(
            "Stitched %d frames (%d annotated) at %dfps %dx%d",
            frame_idx, len(frame_annotations), int(fps), width, height,
        )

        with open(output_path, "rb") as f:
            return f.read()
    finally:
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.unlink(p)


# ──────────────────────────────────────────────────────────────────────
# Single-image annotation (photo path)
# ──────────────────────────────────────────────────────────────────────

def annotate_image(image_bytes: bytes, detections: list[dict[str, Any]]) -> bytes:
    """Decode → annotate → re-encode as JPEG."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")

    annotate_frame(frame, detections)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()