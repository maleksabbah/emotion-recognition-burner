# EmotionRecognitionBurnerService
# Kafka consumer: stitch video/photo with emotion overlays
# Talks to: Kafka, MinIO (read source), Storage (HTTP for save_output)
# Scale with: docker compose up --scale burner=N

FROM python:3.11-slim

WORKDIR /app

# OpenCV system deps + ffmpeg (cv2.VideoWriter on slim Linux needs it for mp4)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

CMD ["python", "main.py"]
