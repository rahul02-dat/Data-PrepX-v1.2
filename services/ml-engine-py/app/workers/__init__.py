# Re-export celery_app so `celery -A app.workers worker` resolves correctly.
from app.workers.celery_app import celery_app  # noqa: F401
