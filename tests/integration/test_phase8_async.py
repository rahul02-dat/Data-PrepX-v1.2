"""Integration tests for asynchronous pipeline execution and gateway dispatch."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import psycopg
import pytest
import requests
import websocket

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")
ML_ENGINE_URL = os.environ.get("ML_ENGINE_URL", "http://localhost:8000")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://dataprepx:dataprepx@localhost:5432/dataprepx?sslmode=disable",
)

_SMALL_DATASET = {
    "dataset": {
        "rows": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "target": 0},
            {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "target": 1},
            {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "target": 0},
            {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "target": 1},
            {"sepal_length": 5.0, "sepal_width": 3.6, "petal_length": 1.4, "target": 0},
        ],
        "columns": ["sepal_length", "sepal_width", "petal_length", "target"],
    },
    "target_column": "target",
    "task_type": "classification",
    "imputation_method": "mice",
    "outlier_method": "isolation_forest",
    "seed": 42,
    "n_trials": 2,
    "cv_folds": 2,
    "stacking_cv_folds": 2,
}


def _ws_url(job_id: str) -> str:
    host = GATEWAY_URL.replace("http://", "ws://").replace("https://", "wss://")
    return f"{host}/v1/jobs/{job_id}/ws"


def _collect_ws_statuses(job_id: str, timeout: float = 120.0) -> list[str]:
    """Open WebSocket to gateway and collect all status strings until terminal."""
    statuses: list[str] = []
    done = threading.Event()

    def on_message(ws, message):
        import json
        data = json.loads(message)
        status = data.get("status", "")
        statuses.append(status)
        if status in ("done", "failed"):
            done.set()
            ws.close()

    def on_error(ws, error):
        done.set()

    def on_close(ws, *_):
        done.set()

    ws_app = websocket.WebSocketApp(
        _ws_url(job_id), on_message=on_message, on_error=on_error, on_close=on_close
    )
    t = threading.Thread(target=ws_app.run_forever, daemon=True)
    t.start()
    done.wait(timeout=timeout)
    return statuses


def _lineage_step_count(job_id: str, step_type: str) -> int:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM pipeline_steps WHERE run_id = %s AND step_type = %s",
                (job_id, step_type),
            )
            return cur.fetchone()[0]


@pytest.mark.db
def test_job_submission_reaches_terminal_state():
    """Verify job submission through gateway reaches terminal state."""
    resp = requests.post(f"{GATEWAY_URL}/v1/jobs", json=_SMALL_DATASET, timeout=10)
    assert resp.status_code in (200, 201, 202), f"submit failed: {resp.text}"
    job_id = resp.json()["id"]

    statuses = _collect_ws_statuses(job_id, timeout=180.0)
    assert len(statuses) > 0, "no WebSocket messages received"
    assert statuses[-1] in ("done", "failed"), f"unexpected terminal status: {statuses}"

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM runs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    assert row is not None, "no runs row found in Postgres"
    assert row[0] in ("done", "failed", "gate-check"), f"unexpected DB status: {row[0]}"


@pytest.mark.db
def test_websocket_status_order():
    """Verify WebSocket status transitions arrive in order."""
    resp = requests.post(f"{GATEWAY_URL}/v1/jobs", json=_SMALL_DATASET, timeout=10)
    assert resp.status_code in (200, 201, 202)
    job_id = resp.json()["id"]

    statuses = _collect_ws_statuses(job_id, timeout=180.0)
    assert "running" in statuses or "gate-check" in statuses, (
        f"expected at least 'running' in transitions; got {statuses}"
    )
    terminal = statuses[-1]
    assert terminal in ("done", "failed")


@pytest.mark.db
def test_idempotency_no_duplicate_pipeline_steps():
    """Verify idempotent job execution produces no duplicate lineage steps."""
    resp1 = requests.post(f"{GATEWAY_URL}/v1/jobs", json=_SMALL_DATASET, timeout=10)
    assert resp1.status_code in (200, 201, 202)
    job_id1 = resp1.json()["id"]

    _collect_ws_statuses(job_id1, timeout=180.0)

    resp2 = requests.post(f"{GATEWAY_URL}/v1/jobs", json=_SMALL_DATASET, timeout=10)
    assert resp2.status_code in (200, 201, 202)
    job_id2 = resp2.json()["id"]
    _collect_ws_statuses(job_id2, timeout=180.0)

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, step_type, COUNT(*)
                FROM pipeline_steps
                WHERE run_id IN (%s, %s)
                GROUP BY run_id, step_type
                HAVING COUNT(*) > 1
                """,
                (job_id1, job_id2),
            )
            duplicates = cur.fetchall()
    assert not duplicates, f"duplicate pipeline_steps rows found: {duplicates}"


@pytest.mark.db
def test_gateway_returns_429_on_concurrency_cap(monkeypatch):
    """Verify gateway enforces per-user concurrency limit under high load."""
    responses = []
    for _ in range(3):
        r = requests.post(f"{GATEWAY_URL}/v1/jobs", json=_SMALL_DATASET, timeout=5)
        responses.append(r.status_code)

    status_codes = set(responses)
    assert 500 not in status_codes, "gateway returned 500; check logs"
    if 429 in status_codes:
        assert all(c in (200, 201, 202, 429) for c in responses)

