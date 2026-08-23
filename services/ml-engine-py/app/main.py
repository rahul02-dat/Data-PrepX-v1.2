from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.jobs import router as jobs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: verify the Celery broker (Redis) is reachable before accepting
    requests. A worker that can't reach its broker will silently enqueue tasks
    into a void, so we fail loudly at startup instead.
    """
    from app.workers.celery_app import celery_app

    try:
        # Ping the broker with a 5-second timeout. inspect().ping() requires at
        # least one running worker; conn.ensure_connection() only needs Redis.
        with celery_app.connection_for_read() as conn:
            conn.ensure_connection(max_retries=3, interval_start=0.5, interval_step=0.5)
    except Exception as exc:
        raise RuntimeError(
            f"ml-engine-py: Celery broker unreachable at startup. "
            f"Is Redis running? Error: {exc}"
        ) from exc

    yield
    # No explicit shutdown needed; Celery connections are closed when workers
    # themselves stop.


app = FastAPI(
    title="dataprepx-ml-engine",
    version="0.0.1",
    description="DataPrepX v2 ML engine: pipeline core + async Celery task graph.",
    lifespan=lifespan,
)

app.include_router(jobs_router)


# Health check endpoint
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "ml-engine-py"}
