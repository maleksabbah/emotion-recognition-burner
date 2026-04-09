"""
Overlay — draws emotion annotations on frames.

Visual system:
  - Emotion  → base color (red=angry, yellow=happy, etc.)
  - Intensity → saturation. Low=faded toward grey, High=vivid
  - Valence  → color temperature. Negative=cooler/blue shift, Positive=warmer/orange shift
  - Arousal  → box thickness. Low=thin(1), High=thick(4)

Also draws:
  - Emotion label + confidence
  - Confidence bar
  - Corner legend explaining the visual system
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from app.Config import (
    FONT_SCALE, TEXT_THICKNESS,
    BAR_WIDTH, BAR_HEIGHT,
    EMOTION_COLORS, DEFAULT_COLOR,
)

logger = logging.getLogger("burner.overlay")

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Intensity labels → scale factor ───────────────────
INTENSITY_SCALE = {
    "low": 0.35,
    "medium": 0.65,
    "high": 1.0,
}

# ── Valence labels → color shift ──────────────────────
# Negative shifts blue channel up, positive shifts red channel up (BGR)
VALENCE_SHIFT = {
    "negative": (30, -10, -20),  # cooler: more blue, less red
    "neutral": (0, 0, 0),
    "positive": (-20, -10, 30),  # warmer: more red, less blue
}

# ── Arousal labels → box thickness ────────────────────
AROUSAL_THICKNESS = {
    "low": 1,
    "high": 4,
}

def _clamp(val:int) -> int:
    """Clamp a color channel value to 0-255."""
    return max(0, min(255, val))


def get_base_color(emotion: str) -> tuple[int, int, int]:
    """Get the raw BGR base color for an emotion."""
    return EMOTION_COLORS.get(emotion, DEFAULT_COLOR)

def apply_intensity(
    color: tuple[int, int, int],
    intensity_label: str,
) -> tuple[int, int, int]:
    """
    Scale color saturation by intensity.
    Low intensity fades toward grey (160,160,160).
    High intensity keeps full color.
    """
    scale = INTENSITY_SCALE.get(intensity_label, 0.65)
    grey = 160
    b = int(grey + (color[0] - grey) * scale)
    g = int(grey + (color[1] - grey) * scale)
    r = int(grey + (color[2] - grey) * scale)
    return (_clamp(b), _clamp(g), _clamp(r))

def apply_valence(
    color: tuple[int, int, int],
    valence_label: str,
) -> tuple[int, int, int]:
    """
    Shift color temperature by valence.
    Negative → cooler (blue shift), Positive → warmer (red shift).
    """
    shift = VALENCE_SHIFT.get(valence_label, (0, 0, 0))
    return (
        _clamp(color[0] + shift[0]),
        _clamp(color[1] + shift[1]),
        _clamp(color[2] + shift[2]),
    )


def get_box_thickness(arousal_label: str) -> int:
    """Get box thickness from arousal level."""
    return AROUSAL_THICKNESS.get(arousal_label, 2)


def compute_display_color(predictions: dict[str, Any]) -> tuple[int, int, int]:
    """
    Compute the final display color from all prediction heads.
    Emotion → base color → intensity scales it → valence shifts it.
    """
    emotion_label = predictions.get("emotion", {}).get("label", "unknown")
    intensity_label = predictions.get("intensity", {}).get("label", "medium")
    valence_label = predictions.get("valence", {}).get("label", "neutral")

    color = get_base_color(emotion_label)
    color = apply_intensity(color, intensity_label)
    color = apply_valence(color, valence_label)
    return color

# ── Drawing primitives ─────────────────────────────────


def draw_bbox(
        frame: np.ndarray,
        bbox: list[int],
        color: tuple[int, int, int],
        thickness: int = 2,
) -> None:
    """Draw a bounding box on the frame."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def draw_label(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> int:
    """
    Draw a text label with background.
    Returns the y-offset for the next line.
    """
    x, y = position
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICKNESS)

    cv2.rectangle(
        frame,
        (x,y - th - 4),
        (x + tw + 4,y + baseline),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x + 2, y - 2), FONT, FONT_SCALE, color, TEXT_THICKNESS)

    return th + baseline + 4

def draw_confidence_bar(
    frame: np.ndarray,
    position: tuple[int, int],
    confidence: float,
    color: tuple[int, int, int],
) -> int:
    """
    Draw a horizontal confidence bar.
    Returns the height used.
    """
    x, y = position
    bar_w = BAR_WIDTH
    bar_h = BAR_HEIGHT

    # Background
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), cv2.FILLED)

    # Filled portion
    fill_w = int(bar_w * confidence)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), color, cv2.FILLED)

    # Border
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (100, 100, 100), 1)

    return bar_h + 4

# ── Legend ──────────────────────────────────────────────
def draw_legend(frame: np.ndarray) -> None:
    """Draw a corner legend explaining the visual system."""
    h, w = frame.shape[:2]
    legend_x = 8
    legend_y = 8
    line_h = 16
    small_font = 0.4
    small_thick = 1

    # Semi-transparent background
    overlay = frame.copy()
    legend_w = 180
    legend_h = 140
    cv2.rectangle(overlay, (legend_x, legend_y),
                  (legend_x + legend_w, legend_y + legend_h),
                  (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Emotion colors
    y = legend_y + 14
    for emotion, color in EMOTION_COLORS.items():
        cv2.circle(frame, (legend_x + 8, y - 3), 4, color, cv2.FILLED)
        cv2.putText(frame, emotion, (legend_x + 18, y),
                    FONT, small_font, (220, 220, 220), small_thick)
        y += line_h

    # Visual guide
    y += 4
    cv2.putText(frame, "vivid=strong  faded=weak", (legend_x + 4, y),
                FONT, 0.32, (160, 160, 160), small_thick)
    y += line_h - 2
    cv2.putText(frame, "thick=high arousal", (legend_x + 4, y),
                FONT, 0.32, (160, 160, 160), small_thick)
    y += line_h - 2
    cv2.putText(frame, "warm=positive  cool=negative", (legend_x + 4, y),
                FONT, 0.32, (160, 160, 160), small_thick)


# ── Main drawing functions ─────────────────────────────

def draw_detection(
        frame: np.ndarray,
        detection: dict[str, Any],
) -> None:
    """
    Draw a single detection with all annotations.

    Visual encoding:
      - Emotion  → base color
      - Intensity → color saturation
      - Valence  → color temperature (warm/cool)
      - Arousal  → box thickness
    """
    bbox = detection.get("bbox", [0, 0, 100, 100])
    predictions = detection.get("predictions", {})

    emotion_data = predictions.get("emotion", {})
    emotion_label = emotion_data.get("label", "unknown")
    emotion_conf = emotion_data.get("confidence", 0.0)
    arousal_label = predictions.get("arousal", {}).get("label", "low")

    # Compute final color from emotion + intensity + valence
    color = compute_display_color(predictions)
    thickness = get_box_thickness(arousal_label)

    # Bounding box with arousal-based thickness
    draw_bbox(frame, bbox, color, thickness)

    x1, y1 = bbox[0], bbox[1]
    cursor_y = y1 - 4

    # Main label: "happy 92%"
    main_text = f"{emotion_label} {emotion_conf:.0%}"
    h = draw_label(frame, main_text, (x1, cursor_y), color)
    cursor_y -= h

    # Confidence bar
    draw_confidence_bar(frame, (x1, cursor_y - BAR_HEIGHT), emotion_conf, color)
    cursor_y -= BAR_HEIGHT + 6

    # Secondary info below the box
    x_bottom = bbox[0]
    y_bottom = bbox[3] + 16

    for head_name in ("intensity", "valence", "arousal"):
        head_data = predictions.get(head_name, {})
        if head_data:
            label = head_data.get("label", "?")
            conf = head_data.get("confidence", 0.0)
            text = f"{head_name}: {label} ({conf:.0%})"
            draw_label(frame, text, (x_bottom, y_bottom), (180, 180, 180))
            y_bottom += 20


def annotate_frame(
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        show_legend: bool = True,
) -> np.ndarray:
    """
    Annotate a frame with all detections.
    Returns the annotated frame (modifies in-place and returns).
    """
    if show_legend and detections:
        draw_legend(frame)

    for det in detections:
        draw_detection(frame, det)
    return frame










