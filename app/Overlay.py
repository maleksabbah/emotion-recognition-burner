"""
Overlay — minimalist emotion annotation.

Design:
  - Single clean bounding box around the face.
  - One evocative word derived from the emotion + valence + arousal crosstab
    (e.g. "ELATED", "SERENE", "ENRAGED", "MELANCHOLY").
  - Confidence shown as a decimal (e.g. .92).
  - No legend, no confidence bars, no secondary labels.

Typography carries the weight, not information density.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Emotion word crosstab ──────────────────────────────
# Given (emotion_label, valence_label, arousal_label), pick a single evocative word.
# Falls back to the raw emotion label if no crosstab match.

_CROSSTAB = {
    # happy
    ("happy",    "positive", "high"): "ELATED",
    ("happy",    "positive", "low"):  "CONTENT",
    ("happy",    "neutral",  "high"): "AMUSED",
    ("happy",    "neutral",  "low"):  "PLEASED",

    # neutral / calm
    ("neutral",  "neutral",  "low"):  "STILL",
    ("neutral",  "positive", "low"):  "SERENE",
    ("neutral",  "negative", "low"):  "BLANK",
    ("neutral",  "neutral",  "high"): "ALERT",

    # sad
    ("sad",      "negative", "low"):  "MELANCHOLY",
    ("sad",      "negative", "high"): "ANGUISHED",
    ("sad",      "neutral",  "low"):  "WISTFUL",

    # angry
    ("angry",    "negative", "high"): "ENRAGED",
    ("angry",    "negative", "low"):  "BITTER",
    ("angry",    "neutral",  "high"): "TENSE",

    # fear
    ("fear",     "negative", "high"): "TERRIFIED",
    ("fear",     "negative", "low"):  "UNEASY",
    ("fear",     "neutral",  "high"): "STARTLED",

    # disgust
    ("disgust",  "negative", "high"): "REPULSED",
    ("disgust",  "negative", "low"):  "DISDAINFUL",

    # surprise
    ("surprise", "positive", "high"): "AWESTRUCK",
    ("surprise", "neutral",  "high"): "STARTLED",
    ("surprise", "negative", "high"): "SHOCKED",
}


def _word_for(emotion: str, valence: str, arousal: str) -> str:
    key = (emotion.lower(), valence.lower(), arousal.lower())
    if key in _CROSSTAB:
        return _CROSSTAB[key]
    # Loose fallback: match on emotion + arousal
    for (e, v, a), word in _CROSSTAB.items():
        if e == key[0] and a == key[2]:
            return word
    return emotion.upper()


# ── Drawing ────────────────────────────────────────────

def _bbox_color(valence: str) -> tuple[int, int, int]:
    """BGR. Warm for positive, cool for negative, off-white for neutral."""
    v = valence.lower()
    if v == "positive":
        return (90, 160, 240)     # warm amber
    if v == "negative":
        return (180, 120, 80)     # cool slate
    return (220, 220, 210)        # bone


def _draw_bbox(frame: np.ndarray, bbox: list[int], color: tuple[int, int, int], thickness: int) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def _draw_word(
    frame: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    """Draw bold uppercase text with a soft shadow for legibility on any background."""
    x, y = anchor
    # shadow
    cv2.putText(frame, text, (x + 2, y + 2), FONT, scale, (20, 20, 20), thickness + 1, cv2.LINE_AA)
    # main
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, detection: dict[str, Any]) -> None:
    """Minimalist per-face annotation."""
    h, w = frame.shape[:2]
    bbox = detection.get("bbox", [0, 0, 100, 100])
    preds = detection.get("predictions", {})

    emotion = preds.get("emotion", {}).get("label", "unknown")
    confidence = float(preds.get("emotion", {}).get("confidence", 0.0))
    valence = preds.get("valence", {}).get("label", "neutral")
    arousal = preds.get("arousal", {}).get("label", "low")

    color = _bbox_color(valence)

    # Thickness scales subtly with arousal — high feels more urgent
    thickness = 3 if arousal.lower() == "high" else 2
    _draw_bbox(frame, bbox, color, thickness)

    # Big word below the box, centered-ish under the face
    word = _word_for(emotion, valence, arousal)
    x1, y1, x2, y2 = map(int, bbox)
    face_w = x2 - x1

    # Scale letter size to the face width (1px per ~15px of face, clamped)
    word_scale = max(0.9, min(2.4, face_w / 260.0))
    word_thick = max(2, int(word_scale * 2))

    (tw, th), _ = cv2.getTextSize(word, FONT, word_scale, word_thick)
    word_x = max(12, min(w - tw - 12, x1 + (face_w - tw) // 2))
    word_y = min(h - 12, y2 + th + 20)
    _draw_word(frame, word, (word_x, word_y), (240, 240, 232), word_scale, word_thick)

    # Confidence as a small decimal under the word (e.g. ".92")
    conf_text = f".{int(round(confidence * 100)):02d}"
    conf_scale = max(0.5, word_scale * 0.42)
    conf_thick = max(1, int(conf_scale * 2))
    (cw, ch), _ = cv2.getTextSize(conf_text, FONT, conf_scale, conf_thick)
    conf_x = max(12, min(w - cw - 12, x1 + (face_w - cw) // 2))
    conf_y = min(h - 6, word_y + ch + 14)
    _draw_word(frame, conf_text, (conf_x, conf_y), (170, 170, 160), conf_scale, conf_thick)


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    show_legend: bool = False,  # kept for compat, ignored
) -> np.ndarray:
    """Annotate all detections on a frame. Returns the frame (also mutated in place)."""
    for det in detections:
        draw_detection(frame, det)
    return frame
