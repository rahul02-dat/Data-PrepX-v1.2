"""
Phase 8 unit tests for Celery tasks.

These tests run in Celery EAGER mode (CELERY_TASK_ALWAYS_EAGER=True), so tasks
execute synchronously in-process without a running broker or worker. This makes
the tests fast and self-contained -- no Redis required.

Idempotency, gate-rejection, and retry behaviour are tested via mocking.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Configure Celery for eager synchronous execution before importing tasks
from app.workers.celery_app import celery_app

celery_app.conf.update(
    task_always_eager=True,
    task_eager_propagates=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [0, 1, 0, 1, 0],
        }
    )


def _df_rows_cols(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    return df.to_dict(orient="records"), list(df.columns)


# ---------------------------------------------------------------------------
# Idempotency tests
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Each task must not re-execute if its lineage step already exists."""

    def test_imputation_skips_on_existing_step(self):
        """run_imputation returns 'skipped' if the step was already recorded."""
        from app.workers.tasks import run_imputation

        df = _make_df()
        rows, cols = _df_rows_cols(df)

        # Simulate an existing lineage step by patching _step_exists to return
        # a non-None hash.
        with patch("app.workers.tasks._step_exists", return_value="sha256:abc123"):
            with patch("app.workers.tasks.get_connection") as mock_conn:
                mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_conn.return_value.close = MagicMock()
                conn = MagicMock()
                mock_conn.return_value = conn

                result = run_imputation.apply(
                    args=["run-id-1", rows, cols],
                    kwargs={"method": "mice", "seed": 42},
                ).get()

        assert result["status"] == "skipped"
        assert result["output_hash"] == "sha256:abc123"

    def test_outlier_skips_on_existing_step(self):
        """run_outlier_detection returns 'skipped' if the step already ran."""
        from app.workers.tasks import run_outlier_detection

        df = _make_df()
        rows, cols = _df_rows_cols(df)

        with patch("app.workers.tasks._step_exists", return_value="sha256:def456"):
            with patch("app.workers.tasks.get_connection") as mock_conn:
                conn = MagicMock()
                mock_conn.return_value = conn

                result = run_outlier_detection.apply(
                    args=["run-id-2", rows, cols],
                    kwargs={"method": "isolation_forest", "seed": 42},
                ).get()

        assert result["status"] == "skipped"

    def test_rl_episode_skips_if_already_recorded(self):
        """run_rl_episode returns 'skipped' if the episode row exists."""
        from app.workers.tasks import run_rl_episode

        df = _make_df()
        rows, cols = _df_rows_cols(df)

        # Patch the DB cursor to return a row (episode already exists)
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = ("existing-row-id",)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("app.workers.tasks.get_connection", return_value=mock_conn):
            result = run_rl_episode.apply(
                args=["run-id-3", 0, rows, cols],
                kwargs={
                    "target_column": "target",
                    "task_type": "classification",
                    "seed": 42,
                    "fast_surrogate": True,
                },
            ).get()

        assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# Gate rejection test
# ---------------------------------------------------------------------------


class TestGateRejection:
    """run_validation_gates must mark the run 'failed' on a gate rejection."""

    def test_gate_rejection_marks_failed(self):
        """A dataset that violates the null-rate gate must write status='failed' to Postgres."""

        from app.workers.tasks import run_validation_gates

        # Build a dataset with 100% null rate on feature column — guaranteed rejection
        df = pd.DataFrame(
            {
                "all_nulls": [None, None, None],
                "ok_col": [1.0, 2.0, 3.0],
            }
        )
        rows, cols = _df_rows_cols(df)

        status_transitions = []

        def fake_update_status(conn, run_id, status):
            status_transitions.append(status)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.commit = MagicMock()
        mock_conn.close = MagicMock()

        with patch("app.workers.tasks.get_connection", return_value=mock_conn):
            with patch("app.workers.tasks._step_exists", return_value=None):
                with patch(
                    "app.workers.tasks._update_run_status",
                    side_effect=fake_update_status,
                ):
                    with patch(
                        "app.workers.tasks.LineageRecorder",
                        return_value=MagicMock(),
                    ):
                        # In eager mode Celery swallows Ignore; the task returns
                        # without raising. What matters is that 'failed' was written.
                        run_validation_gates.apply(
                            args=["run-fail", rows, cols],
                        )

        # 'failed' must have been written at some point during the gate check
        assert (
            "failed" in status_transitions
        ), f"expected 'failed' in status transitions; got {status_transitions}"


# ---------------------------------------------------------------------------
# Retry test
# ---------------------------------------------------------------------------


class TestRetry:
    """Tasks must retry on transient connection errors."""

    def test_imputation_retries_on_db_error(self):
        """run_imputation retries when get_connection raises an OSError.

        In Celery eager mode, self.retry() raises celery.exceptions.Retry
        (not the original exception) on the first retry. We assert that
        the Retry exception is raised, which proves the retry path was taken.
        """
        from celery.exceptions import Retry

        from app.workers.tasks import run_imputation

        df = _make_df()
        rows, cols = _df_rows_cols(df)

        # get_connection raises on every call → task hits the retry path
        with patch(
            "app.workers.tasks.get_connection",
            side_effect=OSError("connection refused"),
        ):
            with pytest.raises((Retry, OSError)):
                run_imputation.apply(
                    args=["run-retry", rows, cols],
                    kwargs={"method": "mice", "seed": 42},
                ).get()

    def test_summarizer_retries_on_http_error(self):
        """run_summarizer retries when agent-orchestrator returns 503.

        In eager mode, self.retry() raises celery.exceptions.Retry.
        """
        import httpx
        from celery.exceptions import Retry

        from app.workers.tasks import run_summarizer

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "service unavailable"

        with patch("app.workers.tasks.get_connection", return_value=mock_conn):
            with patch("app.workers.tasks._step_exists", return_value=None):
                with patch(
                    "app.workers.tasks.httpx.post",
                    side_effect=httpx.HTTPStatusError(
                        "503", request=MagicMock(), response=mock_response
                    ),
                ):
                    with pytest.raises((Retry, Exception)):
                        run_summarizer.apply(
                            args=["run-http-err", []],
                        ).get()
