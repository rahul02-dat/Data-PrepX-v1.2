from __future__ import annotations

import os

from celery import Celery

# ---------------------------------------------------------------------------
# Broker / backend
# ---------------------------------------------------------------------------
# Both broker and result-backend use the same Redis instance (REDIS_URL).
# Using db 0 for the broker and db 1 for the results backend keeps keys from
# colliding and lets operators FLUSHDB independently if needed.
_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_RESULT_BACKEND = _REDIS_URL.rstrip("/").rsplit("/", 1)[0] + "/1"

celery_app = Celery(
    "dataprepx",
    broker=_REDIS_URL,
    backend=_RESULT_BACKEND,
)

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------
celery_app.conf.update(
    # Task serialisation — JSON keeps tasks debuggable without pickletool
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Acknowledge tasks only after they complete (not on delivery). Combined with
    # idempotency checks in tasks.py this means a worker SIGKILL causes a re-delivery
    # and the idempotency guard prevents a duplicate lineage write.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Named queues — each logical pipeline stage gets its own queue so operators
    # can scale worker pools per stage independently.
    task_routes={
        "app.workers.tasks.run_validation_gates": {"queue": "gates"},
        "app.workers.tasks.run_imputation": {"queue": "preprocessing"},
        "app.workers.tasks.run_outlier_detection": {"queue": "preprocessing"},
        "app.workers.tasks.run_estimation": {"queue": "estimation"},
        "app.workers.tasks.run_rl_episode": {"queue": "estimation"},
        "app.workers.tasks.run_maml_adaptation": {"queue": "estimation"},
        "app.workers.tasks.run_summarizer": {"queue": "summarization"},
    },
    # Visibility timeout long enough for the heaviest tasks (Optuna runs can be
    # tens of minutes at full n_trials). 6 hours is a safe upper bound; if a task
    # takes longer than this it is a runaway and should be killed.
    broker_transport_options={"visibility_timeout": 21600},
    # Limit result retention to 24 hours — results are persisted in Postgres via
    # lineage.py; the Redis result backend is only used for AsyncResult.get() in
    # the polling loop, not as the system of record.
    result_expires=86400,
    # Worker settings
    worker_prefetch_multiplier=1,  # one task at a time per worker thread
)

# Auto-discover tasks so `celery -A app.workers worker` picks up tasks.py
celery_app.autodiscover_tasks(["app.workers"])
