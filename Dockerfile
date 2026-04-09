# EmotionRecognitionBurnerService
# Stateless Kafka consumer: annotates frames/video with emotion overlays
# Talks to: Kafka only (burn_tasks → burn_results)
# No HTTP port — scale with: docker compose up --scale burner=2

FROM python:3.11-slim

WORKDIR /app

# OpenCV system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

CMD ["python", "main.py"]
