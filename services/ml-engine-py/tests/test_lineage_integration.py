from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from app.pipeline.config import ImputationConfig, OutlierDetectionConfig, PipelineConfig
from app.pipeline.hashing import hash_dataframe, hash_source
from app.pipeline.imputation import impute
from app.pipeline.lineage import LineageRecorder
from app.pipeline.outliers import detect_outliers
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


def test_imputation_and_outlier_steps_record_into_lineage(conn):
    # Tests integration of imputation and outlier steps into the lineage recorder.
    recorder = LineageRecorder(conn)
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    raw.loc[rng.random(50) < 0.2, "a"] = np.nan

    dataset_id, content_hash = recorder.register_dataset(raw, {"a": "float64", "b": "float64"})
    run_id, _, _ = recorder.get_or_create_run(
        dataset_id=dataset_id,
        dataset_content_hash=content_hash,
        config=PipelineConfig(),
        git_sha="test-sha-imputation",
    )

    imputation_result = impute(raw, ImputationConfig(method="mice"), seed=1)
    imputed_hash = hash_dataframe(imputation_result.dataframe)
    impute_step_id = recorder.record_pipeline_step(
        run_id,
        step_type="mice_imputation",
        input_hash=content_hash,
        output_hash=imputed_hash,
        params=imputation_result.params,
        seed=1,
        transform_code_hash=hash_source(impute),
        description="Phase 3 MICE imputation",
    )

    outlier_result = detect_outliers(
        imputation_result.dataframe, OutlierDetectionConfig(method="isolation_forest"), seed=1
    )
    outlier_hash = hash_dataframe(outlier_result.dataframe)
    outlier_step_id = recorder.record_pipeline_step(
        run_id,
        step_type="isolation_forest_outliers",
        input_hash=imputed_hash,
        output_hash=outlier_hash,
        params=outlier_result.params,
        seed=1,
        transform_code_hash=hash_source(detect_outliers),
        description="Phase 3 Isolation Forest outlier detection",
    )

    record = recorder.replay_run(run_id)
    step_ids = {s.id for s in record.steps}
    assert {impute_step_id, outlier_step_id} == step_ids

    # Ensures replay reproduces identical output hashes for reproducibility.
    replayed_impute = impute(raw, ImputationConfig(method="mice", mice_max_iter=25), seed=1)
    assert hash_dataframe(replayed_impute.dataframe) == imputed_hash
    replayed_outliers = detect_outliers(
        replayed_impute.dataframe, OutlierDetectionConfig(method="isolation_forest"), seed=1
    )
    assert hash_dataframe(replayed_outliers.dataframe) == outlier_hash


def test_replay_run_raises_for_unknown_id(conn):
    recorder = LineageRecorder(conn)
    with pytest.raises(LookupError):
        recorder.replay_run("00000000-0000-0000-0000-000000000000")
