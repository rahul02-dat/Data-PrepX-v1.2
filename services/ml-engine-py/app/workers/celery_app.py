from __future__ import annotations

import os

from celery import Celery

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_RESULT_BACKEND = _REDIS_URL.rstrip("/").rsplit("/", 1)[0] + "/1"

celery_app = Celery(
    "dataprepx",
    broker=_REDIS_URL,
    backend=_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        "app.workers.tasks.run_validation_gates": {"queue": "gates"},
        "app.workers.tasks.run_imputation": {"queue": "preprocessing"},
        "app.workers.tasks.run_outlier_detection": {"queue": "preprocessing"},
        "app.workers.tasks.run_estimation": {"queue": "estimation"},
        "app.workers.tasks.run_rl_episode": {"queue": "estimation"},
        "app.workers.tasks.run_maml_adaptation": {"queue": "estimation"},
        "app.workers.tasks.run_summarizer": {"queue": "summarization"},
    },
    broker_transport_options={"visibility_timeout": 21600},
    result_expires=86400,
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["app.workers"])
