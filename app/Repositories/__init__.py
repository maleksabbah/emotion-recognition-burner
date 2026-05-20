"""
Burner worker repositories — transport boundaries.

  KafkaConsumer    consume burn_tasks
  KafkaProducer    publish burn_results
  S3Client         fetch source video from MinIO
  StorageClient    POST burned output bytes to storage service
"""
from app.Repositories.KafkaConsumer import KafkaConsumer
from app.Repositories.KafkaProducer import KafkaProducer
from app.Repositories.S3Client import S3Client
from app.Repositories.StorageClient import StorageClient

__all__ = ["KafkaConsumer", "KafkaProducer", "S3Client", "StorageClient"]