"""Unit tests for lineage.py against a mocked psycopg connection.

No live Postgres is available in this environment, so these tests verify the SQL
call shape and control flow (idempotency branching, structured-reason recording)
via mocks rather than a real database round-trip. postgres_store.go's own tests
took the analogous approach for Phase 1 (see its file header) for the same reason.
Before this ships, it should also be exercised against a real Postgres instance
(docker compose up + make migrate) -- flagging that as a follow-up verification
step this sandbox cannot perform.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.pipeline.config import PipelineConfig
from app.pipeline.hashing import compute_run_key, hash_config, hash_dataframe
from app.pipeline.lineage import LineageRecorder, verify_output_hash
from app.pipeline.validation_gates import GateChainResult, GateResult


def _mock_conn_with_cursor(fetchone_return=None, fetchall_return=None):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = fetchone_return
    cursor.fetchall.return_value = fetchall_return or []
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


def test_register_dataset_inserts_when_new():
    conn, cursor = _mock_conn_with_cursor(fetchone_return={"id": "dataset-uuid-1"})
    recorder = LineageRecorder(conn)
    df = pd.DataFrame({"a": [1, 2, 3]})

    dataset_id, content_hash = recorder.register_dataset(df, {"a": "int64"})

    assert dataset_id == "dataset-uuid-1"
    assert content_hash == hash_dataframe(df)
    insert_sql = cursor.execute.call_args_list[0].args[0]
    assert "INSERT INTO datasets" in insert_sql
    assert "ON CONFLICT (content_hash) DO NOTHING" in insert_sql


def test_register_dataset_falls_back_to_select_on_conflict():
    conn = MagicMock()
    cursor = MagicMock()
    # First execute (INSERT ... RETURNING) yields no row (conflict); second
    # execute (SELECT) yields the existing row.
    cursor.fetchone.side_effect = [None, {"id": "existing-dataset-id"}]
    conn.cursor.return_value.__enter__.return_value = cursor
    recorder = LineageRecorder(conn)
    df = pd.DataFrame({"a": [1]})

    dataset_id, _ = recorder.register_dataset(df, {"a": "int64"})

    assert dataset_id == "existing-dataset-id"
    assert cursor.execute.call_count == 2
    assert "SELECT id FROM datasets" in cursor.execute.call_args_list[1].args[0]


def test_get_or_create_run_creates_new_run_and_computes_run_key():
    conn, cursor = _mock_conn_with_cursor(fetchone_return={"id": "run-uuid-1"})
    recorder = LineageRecorder(conn)
    config = PipelineConfig()

    run_id, run_key, created = recorder.get_or_create_run(
        dataset_id="dataset-1", dataset_content_hash="hash-abc", config=config, git_sha="deadbeef"
    )

    assert run_id == "run-uuid-1"
    assert created is True
    expected_config_hash = hash_config(config.as_dict())
    assert run_key == compute_run_key("hash-abc", expected_config_hash, "deadbeef")
    insert_sql = cursor.execute.call_args_list[0].args[0]
    assert "ON CONFLICT (run_key) DO NOTHING" in insert_sql


def test_get_or_create_run_is_idempotent_on_identical_inputs():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [None, {"id": "existing-run-id"}]
    conn.cursor.return_value.__enter__.return_value = cursor
    recorder = LineageRecorder(conn)
    config = PipelineConfig()

    run_id, _, created = recorder.get_or_create_run(
        dataset_id="dataset-1", dataset_content_hash="hash-abc", config=config, git_sha="deadbeef"
    )

    assert run_id == "existing-run-id"
    assert created is False


def test_get_or_create_run_raises_on_impossible_conflict_state():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [None, None]  # insert conflicted, but lookup also empty
    conn.cursor.return_value.__enter__.return_value = cursor
    recorder = LineageRecorder(conn)

    with pytest.raises(RuntimeError):
        recorder.get_or_create_run(
            dataset_id="d1", dataset_content_hash="h1", config=PipelineConfig(), git_sha="g1"
        )


def test_record_gate_chain_writes_one_row_per_gate_plus_audit_and_status_update():
    conn, cursor = _mock_conn_with_cursor()
    recorder = LineageRecorder(conn)
    chain = GateChainResult(
        passed=False,
        results=[
            GateResult("max_null_rate_gate", passed=True, details={"overall_null_rate": 0.1}),
            GateResult("drift_gate", passed=False, reason="drift detected", details={"a": 1}),
        ],
    )

    recorder.record_gate_chain("run-1", chain)

    executed_sql = [call.args[0] for call in cursor.execute.call_args_list]
    gate_inserts = [s for s in executed_sql if "INSERT INTO gate_evaluations" in s]
    audit_inserts = [s for s in executed_sql if "INSERT INTO audit_log" in s]
    status_updates = [s for s in executed_sql if "UPDATE runs SET status" in s]

    assert len(gate_inserts) == 2
    assert len(audit_inserts) == 1
    assert len(status_updates) == 1
    # Rejected chain -> status must be 'failed', not silently left as queued/running.
    assert status_updates[0]  # sanity: statement exists
    last_update_args = [
        call.args
        for call in cursor.execute.call_args_list
        if "UPDATE runs SET status" in call.args[0]
    ][0]
    assert last_update_args[1][0] == "failed"


def test_record_gate_chain_sets_gate_check_status_when_passed():
    conn, cursor = _mock_conn_with_cursor()
    recorder = LineageRecorder(conn)
    chain = GateChainResult(passed=True, results=[GateResult("max_null_rate_gate", passed=True)])

    recorder.record_gate_chain("run-1", chain)

    update_call = [
        call for call in cursor.execute.call_args_list if "UPDATE runs SET status" in call.args[0]
    ][0]
    assert update_call.args[1][0] == "gate-check"


def test_record_pipeline_step_inserts_step_then_transformation():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": "step-uuid-1"}
    conn.cursor.return_value.__enter__.return_value = cursor
    recorder = LineageRecorder(conn)

    step_id = recorder.record_pipeline_step(
        "run-1",
        step_type="mice_imputation",
        input_hash="in-hash",
        output_hash="out-hash",
        params={"max_iter": 10},
        seed=42,
        transform_code_hash="code-hash-1",
        description="MICE imputation",
    )

    assert step_id == "step-uuid-1"
    executed_sql = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("INSERT INTO pipeline_steps" in s for s in executed_sql)
    assert any("INSERT INTO transformations" in s for s in executed_sql)


def test_replay_run_raises_lookup_error_when_run_missing():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn.cursor.return_value.__enter__.return_value = cursor
    recorder = LineageRecorder(conn)

    with pytest.raises(LookupError):
        recorder.replay_run("nonexistent-run")


def test_replay_run_returns_run_and_ordered_steps():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = {
        "id": "run-1",
        "dataset_id": "dataset-1",
        "git_sha": "abc123",
        "config_hash": "sha256:cfg",
        "run_key": "sha256:key",
    }
    cursor.fetchall.return_value = [
        {
            "id": "step-1",
            "step_type": "mice_imputation",
            "input_hash": "in-1",
            "output_hash": "out-1",
            "params_json": {"max_iter": 5},
            "seed": 1,
        }
    ]
    conn.cursor.return_value.__enter__.return_value = cursor
    recorder = LineageRecorder(conn)

    record = recorder.replay_run("run-1")

    assert record.run_id == "run-1"
    assert record.run_key == "sha256:key"
    assert len(record.steps) == 1
    assert record.steps[0].step_type == "mice_imputation"


# --- verify_output_hash ---------------------------------------------------------


def test_verify_output_hash_matches():
    assert verify_output_hash("sha256:abc", "sha256:abc") is True


def test_verify_output_hash_mismatch():
    assert verify_output_hash("sha256:abc", "sha256:def") is False


def test_verify_output_hash_none_recorded_always_fails():
    assert verify_output_hash(None, "sha256:anything") is False
