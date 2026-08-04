"""Immutable lineage recording (CLAUDE.md §5.3).

Every transformation is content-addressed: hash(input) + hash(transform code) +
hash(hyperparameters) + git SHA. This module writes that record to Postgres and
provides the read-path needed to replay a run.

Scope note: Phase 2 has no transformations to execute yet (imputation/outliers land
in Phase 3, RL in Phase 5, etc.), so `replay_run` returns the full recorded DAG --
everything needed to re-execute it -- rather than actually re-executing anything.
`verify_output_hash` is the primitive later phases call once real transform
functions exist: it takes a freshly recomputed hash and checks it against what was
recorded, which is the actual "replay reproduces identical output hashes" check
required by the Phase 2 acceptance criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from app.pipeline.config import PipelineConfig
from app.pipeline.hashing import compute_run_key, hash_config, hash_dataframe
from app.pipeline.validation_gates import GateChainResult


@dataclass(frozen=True)
class PipelineStepRecord:
    id: str
    step_type: str
    input_hash: str
    output_hash: str | None
    params: dict[str, Any]
    seed: int | None


@dataclass(frozen=True)
class ReplayRecord:
    run_id: str
    dataset_id: str
    git_sha: str
    config_hash: str
    run_key: str
    steps: list[PipelineStepRecord]


class LineageRecorder:
    """Wraps a psycopg connection with the Phase 2 lineage operations.

    Callers own the connection's lifecycle (open/close/commit); this class does not
    manage transactions itself beyond the single statement it's executing, so it
    composes with a caller that wants to wrap a whole pipeline run in one transaction.
    """

    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    def register_dataset(
        self,
        df: pd.DataFrame,
        schema_json: dict[str, Any],
        *,
        reference_dataset_id: str | None = None,
    ) -> tuple[str, str]:
        """Register a dataset by content hash, idempotently. Returns (dataset_id, content_hash)."""
        content_hash = hash_dataframe(df)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO datasets (content_hash, schema_json, reference_dataset_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                (content_hash, psycopg.types.json.Json(schema_json), reference_dataset_id),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute("SELECT id FROM datasets WHERE content_hash = %s", (content_hash,))
                row = cur.fetchone()
            return str(row["id"]), content_hash

    def get_or_create_run(
        self,
        *,
        dataset_id: str | None,
        dataset_content_hash: str,
        config: PipelineConfig,
        git_sha: str,
    ) -> tuple[str, str, bool]:
        """Idempotent run creation keyed on run_key (see ADR 0003). Returns
        (run_id, run_key, created)."""
        config_dict = config.as_dict()
        config_hash_value = hash_config(config_dict)
        run_key = compute_run_key(dataset_content_hash, config_hash_value, git_sha)

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO runs (dataset_id, git_sha, config_hash, run_key, status)
                VALUES (%s, %s, %s, %s, 'queued')
                ON CONFLICT (run_key) DO NOTHING
                RETURNING id
                """,
                (dataset_id, git_sha, config_hash_value, run_key),
            )
            row = cur.fetchone()
            if row is not None:
                return str(row["id"]), run_key, True

            cur.execute("SELECT id FROM runs WHERE run_key = %s", (run_key,))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    f"run_key {run_key} conflicted on insert but no row found on lookup; "
                    "this indicates a concurrent transaction issue, not expected in "
                    "single-writer use."
                )
            return str(row["id"]), run_key, False

    def record_gate_chain(self, run_id: str, gate_chain: GateChainResult) -> None:
        """Persist every gate's structured verdict, plus one audit_log summary entry."""
        with self._conn.cursor() as cur:
            for result in gate_chain.results:
                cur.execute(
                    """
                    INSERT INTO gate_evaluations (run_id, gate_name, passed, reason, details)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        result.gate_name,
                        result.passed,
                        result.reason,
                        psycopg.types.json.Json(result.details),
                    ),
                )
            cur.execute(
                """
                INSERT INTO audit_log (run_id, actor, action)
                VALUES (%s, %s, %s)
                """,
                (
                    run_id,
                    "validation_gates",
                    "gate_chain_passed" if gate_chain.passed else "gate_chain_rejected",
                ),
            )
            cur.execute(
                "UPDATE runs SET status = %s, updated_at = now() WHERE id = %s",
                ("gate-check" if gate_chain.passed else "failed", run_id),
            )

    def record_pipeline_step(
        self,
        run_id: str,
        *,
        step_type: str,
        input_hash: str,
        output_hash: str | None,
        params: dict[str, Any],
        seed: int | None,
        transform_code_hash: str,
        description: str | None = None,
    ) -> str:
        """Record one DAG step plus its transformation record. Returns the step id."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO pipeline_steps
                    (run_id, step_type, input_hash, output_hash, params_json, seed)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    run_id,
                    step_type,
                    input_hash,
                    output_hash,
                    psycopg.types.json.Json(params),
                    seed,
                ),
            )
            step_id = str(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO transformations (step_id, transform_code_hash, description)
                VALUES (%s, %s, %s)
                """,
                (step_id, transform_code_hash, description),
            )
            return step_id

    def replay_run(self, run_id: str) -> ReplayRecord:
        """Fetch everything recorded about a run, in step order, sufficient to
        re-execute the same hashed DAG against the same input hash."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, dataset_id, git_sha, config_hash, run_key FROM runs WHERE id = %s",
                (run_id,),
            )
            run_row = cur.fetchone()
            if run_row is None:
                raise LookupError(f"no run found with id {run_id}")

            cur.execute(
                """
                SELECT id, step_type, input_hash, output_hash, params_json, seed
                FROM pipeline_steps
                WHERE run_id = %s
                ORDER BY created_at ASC
                """,
                (run_id,),
            )
            steps = [
                PipelineStepRecord(
                    id=str(r["id"]),
                    step_type=r["step_type"],
                    input_hash=r["input_hash"],
                    output_hash=r["output_hash"],
                    params=r["params_json"],
                    seed=r["seed"],
                )
                for r in cur.fetchall()
            ]

        return ReplayRecord(
            run_id=str(run_row["id"]),
            dataset_id=str(run_row["dataset_id"]) if run_row["dataset_id"] else None,
            git_sha=run_row["git_sha"],
            config_hash=run_row["config_hash"],
            run_key=run_row["run_key"],
            steps=steps,
        )


def verify_output_hash(recorded_output_hash: str | None, recomputed_hash: str) -> bool:
    """The actual reproducibility check: does re-executing a step produce the same
    output hash that was recorded the first time. A None recorded hash always fails
    verification -- there is nothing to compare against, which is a data problem to
    surface, not a pass-by-default."""
    if recorded_output_hash is None:
        return False
    return recorded_output_hash == recomputed_hash
