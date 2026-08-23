"""
Phase 8 — FastAPI jobs router for ml-engine-py.

Endpoints:
  POST /v1/jobs          Submit a pipeline run; enqueue the Celery chain.
  GET  /v1/jobs/{job_id}/status  Poll current status (Go gateway polls this).

The Go gateway calls these endpoints instead of running the pipeline in-process.
It stores the returned celery_task_id on the runs row and polls /status at ~1s
intervals, turning each status transition into a WebSocket message to the frontend.

Dataset storage (Phase 8 scope):
  Callers must pre-register their dataset (insert a row into `datasets`) and pass
  the resulting dataset_id. Uploading a CSV/Parquet file as multipart is deferred
  to Phase 9 (frontend). For now, the gateway test harness or integration tests
  can call POST /v1/datasets/register with a JSON-encoded DataFrame to pre-register.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import psycopg
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.pipeline.config import load_pipeline_config
from app.pipeline.db import get_connection
from app.pipeline.hashing import hash_dataframe
from app.pipeline.lineage import LineageRecorder
from app.workers.pipeline_chain import enqueue_pipeline

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

_GIT_SHA = os.environ.get("GIT_SHA", "dev")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class DatasetSpec(BaseModel):
    """Inline dataset payload — a list of records + ordered column names.

    This is used in Phase 8 where there's no file upload endpoint yet.
    The gateway passes through what the client submitted, or for tests
    callers can POST directly to ml-engine-py.
    """

    rows: list[dict[str, Any]]
    columns: list[str]


class JobRequest(BaseModel):
    """Parameters for a new pipeline run."""

    dataset: DatasetSpec
    target_column: str
    task_type: str = Field(
        ..., pattern="^(classification|regression)$"
    )
    imputation_method: str = Field("mice", pattern="^(mice|knn)$")
    outlier_method: str = Field(
        "isolation_forest", pattern="^(isolation_forest|lof|none)$"
    )
    seed: int = 42
    n_trials: int = Field(30, ge=1, le=500)
    cv_folds: int = Field(5, ge=2, le=20)
    stacking_cv_folds: int = Field(5, ge=2, le=20)
    # Optional: pass pre-computed reference dataset rows for drift detection.
    reference_dataset: DatasetSpec | None = None


class JobResponse(BaseModel):
    job_id: str
    celery_task_id: str
    status: str = "queued"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    celery_task_id: str | None = None
    last_step: str | None = None


# ---------------------------------------------------------------------------
# POST /v1/jobs
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=JobResponse)
def submit_job(req: JobRequest) -> JobResponse:
    """
    Register a new pipeline run in Postgres, enqueue the Celery task chain,
    and return the job_id + celery_task_id so the gateway can poll status.
    """
    df = pd.DataFrame(req.dataset.rows, columns=req.dataset.columns)
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="dataset must contain at least one row",
        )

    config = load_pipeline_config()

    conn: psycopg.Connection = get_connection()
    try:
        recorder = LineageRecorder(conn)

        schema_json = {col: str(dtype) for col, dtype in df.dtypes.items()}
        dataset_id, _content_hash = recorder.register_dataset(df, schema_json)

        run_id, _run_key, _created = recorder.get_or_create_run(
            dataset_id=dataset_id,
            dataset_content_hash=hash_dataframe(df),
            config=config,
            git_sha=_GIT_SHA,
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to register run in lineage: {exc}",
        ) from exc

    # Enqueue the Celery pipeline chain
    try:
        async_result = enqueue_pipeline(
            run_id=run_id,
            dataset_rows=req.dataset.rows,
            dataset_columns=req.dataset.columns,
            target_column=req.target_column,
            task_type=req.task_type,
            imputation_method=req.imputation_method,
            outlier_method=req.outlier_method,
            seed=req.seed,
            n_trials=req.n_trials,
            cv_folds=req.cv_folds,
            stacking_cv_folds=req.stacking_cv_folds,
            reference_rows=req.reference_dataset.rows if req.reference_dataset else None,
            reference_columns=(
                req.reference_dataset.columns if req.reference_dataset else None
            ),
        )
        celery_task_id = async_result.id
    except Exception as exc:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"failed to enqueue pipeline tasks (is Redis reachable?): {exc}",
        ) from exc

    # Persist celery_task_id on the runs row so the gateway polling loop can
    # retrieve it from Postgres without needing to call this endpoint again.
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET celery_task_id = %s WHERE id = %s",
                (celery_task_id, run_id),
            )
        conn.commit()
    except Exception:
        # Non-fatal: the task is already enqueued; the task_id is returned in
        # the response even if the DB write fails. The gateway still works.
        pass
    finally:
        conn.close()

    return JobResponse(job_id=run_id, celery_task_id=celery_task_id)


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}/status
# ---------------------------------------------------------------------------


@router.get("/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Return the current status of a run from Postgres.

    The Go gateway polls this at ~1s intervals and turns each new status value
    into a WebSocket message. The status values match the constants in
    gateway-go/internal/jobs/model.go: queued, running, gate-check, optimizing,
    done, failed.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, celery_task_id FROM runs WHERE id = %s",
                (job_id,),
            )
            row = cur.fetchone()
    except Exception as exc:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"database error: {exc}",
        ) from exc

    if row is None:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    db_status, celery_task_id = row

    # Also fetch the most recent pipeline_step so the frontend can show
    # fine-grained progress (e.g. "imputation done, running estimation").
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT step_type FROM pipeline_steps
                WHERE run_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (job_id,),
            )
            step_row = cur.fetchone()
        last_step = step_row[0] if step_row else None
    except Exception:
        last_step = None
    finally:
        conn.close()

    return JobStatusResponse(
        job_id=job_id,
        status=db_status,
        celery_task_id=celery_task_id,
        last_step=last_step,
    )
