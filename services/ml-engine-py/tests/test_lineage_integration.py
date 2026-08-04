"""Integration test for lineage.py against a real Postgres instance.

Excluded from the default `pytest` run (see pyproject.toml's `addopts = "-m 'not
db'"`) because it needs DATABASE_URL pointing at a migrated Postgres 16+ database
with the pgcrypto extension enabled. This is what closes the gap flagged when
test_lineage.py was written mocked-only: run this in CI (see
.github/workflows/ci.yml's `ml-engine-py-db-integration` job) or locally with

    docker compose up -d postgres
    make migrate
    DATABASE_URL=postgres://dataprepx:dataprepx@localhost:5432/dataprepx?sslmode=disable \
        .venv/bin/pytest -v -m db
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from app.pipeline.config import PipelineConfig
from app.pipeline.lineage import LineageRecorder
from app.pipeline.validation_gates import GateChainResult, GateResult

pytestmark = pytest.mark.db


@pytest.fixture()
def conn():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set; skipping live-Postgres lineage tests")
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    yield connection
    connection.close()


def test_register_dataset_is_idempotent_against_real_db(conn):
    recorder = LineageRecorder(conn)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    schema = {"a": "int64", "b": "object"}

    dataset_id_1, hash_1 = recorder.register_dataset(df, schema)
    dataset_id_2, hash_2 = recorder.register_dataset(df.copy(), schema)

    assert dataset_id_1 == dataset_id_2
    assert hash_1 == hash_2


def test_get_or_create_run_is_idempotent_against_real_db(conn):
    recorder = LineageRecorder(conn)
    df = pd.DataFrame({"a": [1, 2, 3]})
    dataset_id, content_hash = recorder.register_dataset(df, {"a": "int64"})
    config = PipelineConfig()

    run_id_1, run_key_1, created_1 = recorder.get_or_create_run(
        dataset_id=dataset_id,
        dataset_content_hash=content_hash,
        config=config,
        git_sha="test-sha-1",
    )
    run_id_2, run_key_2, created_2 = recorder.get_or_create_run(
        dataset_id=dataset_id,
        dataset_content_hash=content_hash,
        config=config,
        git_sha="test-sha-1",
    )

    assert run_id_1 == run_id_2
    assert run_key_1 == run_key_2
    assert created_1 is True
    assert created_2 is False


def test_record_gate_chain_and_replay_round_trip(conn):
    recorder = LineageRecorder(conn)
    df = pd.DataFrame({"a": [1, 2, None, 4]})
    dataset_id, content_hash = recorder.register_dataset(df, {"a": "float64"})
    run_id, _, _ = recorder.get_or_create_run(
        dataset_id=dataset_id,
        dataset_content_hash=content_hash,
        config=PipelineConfig(),
        git_sha="test-sha-2",
    )

    chain = GateChainResult(
        passed=True,
        results=[
            GateResult("max_null_rate_gate", passed=True, details={"overall_null_rate": 0.25})
        ],
    )
    recorder.record_gate_chain(run_id, chain)

    step_id = recorder.record_pipeline_step(
        run_id,
        step_type="mice_imputation",
        input_hash=content_hash,
        output_hash="sha256:fake-output-hash",
        params={"max_iter": 10},
        seed=1,
        transform_code_hash="sha256:fake-code-hash",
        description="test step",
    )

    record = recorder.replay_run(run_id)

    assert record.run_id == run_id
    assert len(record.steps) == 1
    assert record.steps[0].id == step_id
    assert record.steps[0].output_hash == "sha256:fake-output-hash"

    with conn.cursor() as cur:
        cur.execute("SELECT status FROM runs WHERE id = %s", (run_id,))
        assert cur.fetchone()[0] == "gate-check"

        cur.execute("SELECT count(*) FROM gate_evaluations WHERE run_id = %s", (run_id,))
        assert cur.fetchone()[0] == 1


def test_replay_run_raises_for_unknown_id(conn):
    recorder = LineageRecorder(conn)
    with pytest.raises(LookupError):
        recorder.replay_run("00000000-0000-0000-0000-000000000000")