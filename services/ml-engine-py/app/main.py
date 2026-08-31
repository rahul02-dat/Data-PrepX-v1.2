from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.jobs import router as jobs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify Celery broker connectivity on service startup."""
    from app.workers.celery_app import celery_app

    try:
        with celery_app.connection_for_read() as conn:
            conn.ensure_connection(max_retries=3, interval_start=0.5, interval_step=0.5)
    except Exception as exc:
        raise RuntimeError(f"ml-engine-py: Celery broker unreachable at startup: {exc}") from exc

    yield


app = FastAPI(
    title="dataprepx-ml-engine",
    version="0.0.1",
    description="DataPrepX v2 ML engine",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "ml-engine-py"}
