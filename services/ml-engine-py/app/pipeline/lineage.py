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
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    def register_dataset(
        self,
        df: pd.DataFrame,
        schema_json: dict[str, Any],
        *,
        reference_dataset_id: str | None = None,
    ) -> tuple[str, str]:
        """Register dataset content hash and schema in database."""
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
        """Get existing run or create new run row keyed on deterministic run_key."""
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
                    f"run_key {run_key} conflicted on insert but no row found on lookup"
                )
            return str(row["id"]), run_key, False

    def record_gate_chain(self, run_id: str, gate_chain: GateChainResult) -> None:
        """Persist gate evaluation results and write audit log entry."""
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
        """Record executed pipeline step and transformation code hash."""
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

    def record_hyperparameter_trial(
        self,
        run_id: str,
        *,
        model_family: str,
        trial_number: int,
        params: dict[str, Any],
        score: float | None,
    ) -> str:
        """Persist hyperparameter trial configuration and evaluation score."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO hyperparameters (run_id, model_family, trial_number, params_json, score)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (run_id, model_family, trial_number, psycopg.types.json.Json(params), score),
            )
            return str(cur.fetchone()["id"])

    def record_study_trials(self, run_id: str, trials: list[Any]) -> list[str]:
        """Record all trials from an Optuna study in bulk."""
        return [
            self.record_hyperparameter_trial(
                run_id,
                model_family=trial.model_family,
                trial_number=trial.trial_number,
                params=trial.params,
                score=trial.score,
            )
            for trial in trials
        ]

    def record_metric(
        self,
        run_id: str,
        *,
        name: str,
        value: float,
        ci_low: float | None = None,
        ci_high: float | None = None,
    ) -> str:
        """Record named evaluation metric with optional confidence interval bounds."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO metrics (run_id, name, value, ci_low, ci_high)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (run_id, name, value, ci_low, ci_high),
            )
            return str(cur.fetchone()["id"])

    def record_rl_episode(
        self,
        *,
        episode_number: int,
        state: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        run_id: str | None,
    ) -> str:
        """Record RL agent episode state, action, and reward."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO rl_episodes (episode_number, state_json, action_json, reward, run_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    episode_number,
                    psycopg.types.json.Json(state),
                    psycopg.types.json.Json(action),
                    reward,
                    run_id,
                ),
            )
            return str(cur.fetchone()["id"])

    def replay_run(self, run_id: str) -> ReplayRecord:
        """Fetch recorded run lineage and pipeline steps for audit/replay."""
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
    """Verify reproducibility by comparing recorded output hash to recomputed hash."""
    if recorded_output_hash is None:
        return False
    return recorded_output_hash == recomputed_hash
