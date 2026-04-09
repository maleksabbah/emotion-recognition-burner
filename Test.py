"""
Burner Service Test Suite
Run: pytest Test.py -v -o asyncio_mode=auto -o python_files=Test.py -o python_classes=Test
"""
from __future__ import annotations

import base64
import tempfile
import os
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest


# ══════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════

@pytest.fixture
def sample_frame_bgr():
    """A 480x640 BGR test image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (400, 400), (100, 100, 100), -1)
    return img


@pytest.fixture
def sample_frame_b64(sample_frame_bgr):
    """Base64 JPEG of the sample frame."""
    _, buf = cv2.imencode(".jpg", sample_frame_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@pytest.fixture
def sample_detection():
    """A single detection with predictions."""
    return {
        "bbox": [200, 150, 380, 350],
        "predictions": {
            "emotion": {
                "label": "happy",
                "confidence": 0.92,
                "probabilities": {
                    "angry": 0.01, "disgust": 0.01, "fear": 0.01,
                    "happy": 0.92, "neutral": 0.03, "sad": 0.01, "surprise": 0.01,
                },
            },
            "intensity": {"label": "high", "confidence": 0.75},
            "valence": {"label": "positive", "confidence": 0.88},
            "arousal": {"label": "high", "confidence": 0.65},
        },
    }


@pytest.fixture
def sample_video_bytes(sample_frame_bgr):
    """A short 10-frame MP4 video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, 30.0, (640, 480))
        for i in range(10):
            frame = sample_frame_bgr.copy()
            cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


@pytest.fixture
def mock_producer():
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


# ══════════════════════════════════════════════
# Overlay tests
# ══════════════════════════════════════════════

class TestOverlay:
    def test_get_base_color_known(self):
        from app.Overlay import get_base_color
        color = get_base_color("happy")
        assert color == (0, 200, 255)

    def test_get_base_color_unknown(self):
        from app.Overlay import get_base_color
        color = get_base_color("confused")
        assert color == (180, 180, 180)

    def test_apply_intensity_high(self):
        from app.Overlay import apply_intensity
        color = (0, 0, 220)  # angry red
        result = apply_intensity(color, "high")
        # High intensity should keep full color
        assert result == (0, 0, 220)

    def test_apply_intensity_low(self):
        from app.Overlay import apply_intensity
        color = (0, 0, 220)  # angry red
        result = apply_intensity(color, "low")
        # Low intensity should fade toward grey (160)
        assert result[0] > color[0]  # blue channel moves toward 160
        assert result[2] < color[2]  # red channel moves toward 160

    def test_apply_valence_negative(self):
        from app.Overlay import apply_valence
        color = (100, 100, 100)
        result = apply_valence(color, "negative")
        # Negative = cooler, blue channel goes up
        assert result[0] > color[0]

    def test_apply_valence_positive(self):
        from app.Overlay import apply_valence
        color = (100, 100, 100)
        result = apply_valence(color, "positive")
        # Positive = warmer, red channel goes up
        assert result[2] > color[2]

    def test_apply_valence_neutral(self):
        from app.Overlay import apply_valence
        color = (100, 100, 100)
        result = apply_valence(color, "neutral")
        assert result == color

    def test_get_box_thickness_high(self):
        from app.Overlay import get_box_thickness
        assert get_box_thickness("high") == 4

    def test_get_box_thickness_low(self):
        from app.Overlay import get_box_thickness
        assert get_box_thickness("low") == 1

    def test_compute_display_color(self):
        from app.Overlay import compute_display_color
        predictions = {
            "emotion": {"label": "happy", "confidence": 0.9},
            "intensity": {"label": "high"},
            "valence": {"label": "positive"},
        }
        color = compute_display_color(predictions)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_compute_display_color_missing_heads(self):
        from app.Overlay import compute_display_color
        # Should use defaults and not crash
        color = compute_display_color({})
        assert isinstance(color, tuple)

    def test_clamp(self):
        from app.Overlay import _clamp
        assert _clamp(-10) == 0
        assert _clamp(300) == 255
        assert _clamp(128) == 128

    def test_draw_bbox(self, sample_frame_bgr):
        from app.Overlay import draw_bbox
        frame = sample_frame_bgr.copy()
        draw_bbox(frame, [200, 150, 380, 350], (0, 255, 0), 2)
        assert not np.array_equal(frame, sample_frame_bgr)

    def test_draw_label(self, sample_frame_bgr):
        from app.Overlay import draw_label
        frame = sample_frame_bgr.copy()
        height = draw_label(frame, "test label", (100, 100), (255, 255, 255))
        assert height > 0
        assert not np.array_equal(frame, sample_frame_bgr)

    def test_draw_confidence_bar(self, sample_frame_bgr):
        from app.Overlay import draw_confidence_bar
        frame = sample_frame_bgr.copy()
        height = draw_confidence_bar(frame, (100, 100), 0.75, (0, 255, 0))
        assert height > 0
        assert not np.array_equal(frame, sample_frame_bgr)

    def test_draw_confidence_bar_zero(self, sample_frame_bgr):
        from app.Overlay import draw_confidence_bar
        frame = sample_frame_bgr.copy()
        height = draw_confidence_bar(frame, (100, 100), 0.0, (0, 255, 0))
        assert height > 0

    def test_draw_detection(self, sample_frame_bgr, sample_detection):
        from app.Overlay import draw_detection
        frame = sample_frame_bgr.copy()
        draw_detection(frame, sample_detection)
        assert not np.array_equal(frame, sample_frame_bgr)

    def test_draw_legend(self, sample_frame_bgr):
        from app.Overlay import draw_legend
        frame = sample_frame_bgr.copy()
        draw_legend(frame)
        assert not np.array_equal(frame, sample_frame_bgr)

    def test_annotate_frame_multiple_detections(self, sample_frame_bgr, sample_detection):
        from app.Overlay import annotate_frame
        frame = sample_frame_bgr.copy()
        det2 = {
            "bbox": [50, 50, 150, 150],
            "predictions": {
                "emotion": {"label": "sad", "confidence": 0.7, "probabilities": {}},
                "intensity": {"label": "low", "confidence": 0.5},
                "valence": {"label": "negative", "confidence": 0.6},
                "arousal": {"label": "low", "confidence": 0.4},
            },
        }
        result = annotate_frame(frame, [sample_detection, det2])
        assert result is frame
        assert not np.array_equal(result, sample_frame_bgr)

    def test_annotate_frame_empty_detections(self, sample_frame_bgr):
        from app.Overlay import annotate_frame
        frame = sample_frame_bgr.copy()
        original = sample_frame_bgr.copy()
        annotate_frame(frame, [])
        assert np.array_equal(frame, original)

    def test_draw_detection_missing_predictions(self, sample_frame_bgr):
        """Should not crash with missing prediction data."""
        from app.Overlay import draw_detection
        frame = sample_frame_bgr.copy()
        det = {"bbox": [100, 100, 200, 200], "predictions": {}}
        draw_detection(frame, det)

    def test_high_arousal_thicker_than_low(self, sample_frame_bgr):
        """High arousal should produce a thicker bounding box."""
        from app.Overlay import get_box_thickness
        assert get_box_thickness("high") > get_box_thickness("low")

    def test_intensity_changes_color(self):
        """Different intensity levels should produce different colors."""
        from app.Overlay import apply_intensity
        base = (0, 0, 220)
        low = apply_intensity(base, "low")
        high = apply_intensity(base, "high")
        assert low != high


# ══════════════════════════════════════════════
# Stitcher tests
# ══════════════════════════════════════════════

class TestStitcher:
    def test_annotate_single_frame(self, sample_frame_b64, sample_detection):
        from app.Stitcher import annotate_single_frame
        frame_bytes = base64.b64decode(sample_frame_b64)
        result = annotate_single_frame(frame_bytes, [sample_detection])
        assert len(result) > 0
        # Decode to verify it's valid JPEG
        arr = np.frombuffer(result, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape == (480, 640, 3)

    def test_annotate_single_frame_no_detections(self, sample_frame_b64):
        from app.Stitcher import annotate_single_frame
        frame_bytes = base64.b64decode(sample_frame_b64)
        result = annotate_single_frame(frame_bytes, [])
        assert len(result) > 0

    def test_annotate_single_frame_invalid_bytes(self):
        from app.Stitcher import annotate_single_frame
        with pytest.raises(ValueError, match="Failed to decode"):
            annotate_single_frame(b"not an image", [])

    def test_stitch_video(self, sample_video_bytes, sample_detection):
        from app.Stitcher import stitch_video
        annotations = {
            0: [sample_detection],
            3: [sample_detection],
            7: [sample_detection],
        }
        result = stitch_video(sample_video_bytes, annotations)
        assert len(result) > 0
        # Verify it's a valid video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(result)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == 10
        cap.release()
        os.unlink(tmp_path)

    def test_stitch_video_no_annotations(self, sample_video_bytes):
        from app.Stitcher import stitch_video
        result = stitch_video(sample_video_bytes, {})
        assert len(result) > 0


# ══════════════════════════════════════════════
# Worker tests
# ══════════════════════════════════════════════

class TestBurnerWorker:
    @pytest.mark.asyncio
    async def test_handle_frame_task(self, sample_frame_b64, sample_detection):
        from app.Worker import BurnerWorker
        worker = BurnerWorker()

        task = {
            "session_id": "sess-1",
            "task_type": "frame",
            "frame_b64": sample_frame_b64,
            "detections": [sample_detection],
        }
        result = await worker._handle_task(task)
        assert result["status"] == "success"
        assert result["session_id"] == "sess-1"
        assert result["task_type"] == "frame"
        assert result["output_b64"] is not None
        assert result["processing_ms"] > 0
        assert result["error"] is None

        # Verify output is valid JPEG
        output_bytes = base64.b64decode(result["output_b64"])
        arr = np.frombuffer(output_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None

    @pytest.mark.asyncio
    async def test_handle_video_task(self, sample_video_bytes, sample_detection):
        from app.Worker import BurnerWorker
        worker = BurnerWorker()

        video_b64 = base64.b64encode(sample_video_bytes).decode("utf-8")
        task = {
            "session_id": "sess-1",
            "task_type": "video",
            "video_b64": video_b64,
            "frame_annotations": {
                "0": [sample_detection],
                "5": [sample_detection],
            },
        }
        result = await worker._handle_task(task)
        assert result["status"] == "success"
        assert result["task_type"] == "video"
        assert result["output_b64"] is not None

    @pytest.mark.asyncio
    async def test_handle_task_error(self):
        from app.Worker import BurnerWorker
        worker = BurnerWorker()

        task = {
            "session_id": "sess-1",
            "task_type": "frame",
            "frame_b64": base64.b64encode(b"bad data").decode(),
            "detections": [],
        }
        result = await worker._handle_task(task)
        assert result["status"] == "error"
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_handle_task_string_keys_conversion(self, sample_video_bytes, sample_detection):
        """JSON serialization turns dict keys to strings — worker should handle this."""
        from app.Worker import BurnerWorker
        worker = BurnerWorker()

        video_b64 = base64.b64encode(sample_video_bytes).decode("utf-8")
        task = {
            "session_id": "sess-1",
            "task_type": "video",
            "video_b64": video_b64,
            "frame_annotations": {"3": [sample_detection]},  # string key
        }
        result = await worker._handle_task(task)
        assert result["status"] == "success"


# ══════════════════════════════════════════════
# Kafka integration tests
# ══════════════════════════════════════════════

class TestKafkaIntegration:
    @pytest.mark.asyncio
    async def test_publish_result(self, mock_producer):
        from app.Kafka import publish_result
        result = {
            "session_id": "s1",
            "worker_id": "burner-1",
            "status": "success",
        }
        await publish_result(mock_producer, result)
        mock_producer.send_and_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_result_includes_worker_header(self, mock_producer):
        from app.Kafka import publish_result
        result = {"worker_id": "b1", "session_id": "s1"}
        await publish_result(mock_producer, result)
        headers = mock_producer.send_and_wait.call_args[1]["headers"]
        assert headers == [("worker_id", b"b1")]