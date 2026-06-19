from celery import Celery

from app.config import settings

celery_app = Celery(
    "football_vision",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Ensure task modules are imported so Celery auto-discovers @app.task decorators
from batch_processor import worker  # noqa: F401, E402
