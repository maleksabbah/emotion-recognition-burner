"""
Burner Service configuration — loaded from environment variables.
"""
import os

# ── Kafka ──────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "burner-worker-group")

TOPIC_BURN_TASKS = os.getenv("TOPIC_BURN_TASKS", "burn_tasks")
TOPIC_BURN_RESULTS = os.getenv("TOPIC_BURN_RESULTS", "burn_results")

# ── Drawing ────────────────────────────────────────────
FONT_SCALE = float(os.getenv("FONT_SCALE", "0.6"))
BOX_THICKNESS = int(os.getenv("BOX_THICKNESS", "2"))
TEXT_THICKNESS = int(os.getenv("TEXT_THICKNESS", "1"))
BAR_WIDTH = int(os.getenv("BAR_WIDTH", "100"))
BAR_HEIGHT = int(os.getenv("BAR_HEIGHT", "12"))

# ── Emotion colors (BGR) ──────────────────────────────
EMOTION_COLORS = {
    "angry":    (0, 0, 220),
    "disgust":  (0, 140, 0),
    "fear":     (200, 100, 0),
    "happy":    (0, 200, 255),
    "neutral":  (200, 200, 200),
    "sad":      (220, 150, 0),
    "surprise": (255, 0, 200),
}
DEFAULT_COLOR = (180, 180, 180)

# ── Worker ─────────────────────────────────────────────
WORKER_ID = os.getenv("WORKER_ID", "burner-worker-1")