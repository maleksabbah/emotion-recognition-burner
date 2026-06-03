# EmotionRecognitionBurner

Annotation worker for the Mntis platform. Consumes `BurnTask`
messages once a session's detections and predictions are all in,
downloads the original image, draws bounding boxes and emotion
labels onto it, and uploads the annotated result back to MinIO.
Publishes a `BurnResult` so the orchestrator can mark the session
complete.

Built as a long-running Kafka consumer (no HTTP surface). Uses
OpenCV for drawing.

## Architecture

The service follows a clean, layered architecture where each layer has
one responsibility and depends only on the layer beneath it:

* **Main loop** — `main.py` runs a Kafka consumer on `burn_tasks`
  and a producer for `burn_results`. Replaces what would otherwise
  be HTTP routes.
* **Services** — the business logic. `BurnService` orchestrates a
  task end-to-end: fetch the source image from MinIO by `s3_key`,
  hand it to the renderer with the detections + predictions, take
  the resulting JPEG bytes, upload the annotated image to MinIO
  under the session's `burned/` prefix, record the burned file
  via the storage service, and emit the `BurnResult` with the
  new `burned_s3_key`. `Renderer` does the drawing — cv2 decodes
  the bytes into a numpy array, iterates over `(detection,
  prediction)` pairs, draws a rectangle with `cv2.rectangle`, and
  writes the emotion label + confidence with `cv2.putText` using
  the Hershey font. Knows nothing about Kafka.
* **Repositories** — data access. `S3Client` for the MinIO
  round-trip. `StorageClient` is the typed HTTP client used to
  record the burned file in the storage service. The service
  layer never touches raw boto3 or HTTP directly.
* **Dtos** — `BurnTask` (input — session_id, source_s3_key, list
  of detections with bboxes, list of predictions with top_emotion
  + top_confidence) and `BurnResult` (output — status,
  burned_s3_key, optional error).
* **Config** — wiring: Kafka consumer (group `burner-workers`,
  reads `burn_tasks`, writes `burn_results`), Kafka producer,
  S3 client, storage service URL, label font / colors / line
  weights.

This separation keeps the queue layer swappable, the business
logic testable in isolation, and the renderer free to evolve (e.g.
swap cv2 for PIL with TrueType fonts) without touching the rest.

The burner is the only worker that doesn't run on crops — it
works on the original full-resolution image with bbox coordinates
the media worker computed earlier. That keeps the artefact the
user downloads identical to what they uploaded, just with the
predictions baked on top.

Part of a multi-service system — see the [platform overview](../EmotionRecognitionDocker)
for the full architecture, pipeline flow, and the other services.
