from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import httpx
import numpy as np
import pandas as pd
import psycopg
from celery import Task
from celery.exceptions import Ignore

from app.pipeline.config import (
    ImputationConfig,
    OutlierDetectionConfig,
    load_pipeline_config,
)
from app.pipeline.db import get_connection
from app.pipeline.estimation.optuna_search import OptunaSearchConfig
from app.pipeline.estimation.stacking import run_stacking
from app.pipeline.hashing import hash_dataframe, hash_source
from app.pipeline.imputation import impute
from app.pipeline.lineage import LineageRecorder
from app.pipeline.meta_learning.adaptive_loop import AdaptiveState, run_adaptive_step
from app.pipeline.meta_learning.maml import MAMLLearner
from app.pipeline.outliers import detect_outliers
from app.pipeline.rl_optimizer.environment import PreprocessingEnv, build_action_space
from app.pipeline.rl_optimizer.q_learning import QLearningAgent
from app.pipeline.rl_optimizer.reward_functions import (
    fast_surrogate_reward_fn,
    full_stack_reward_fn,
)
from app.pipeline.rl_optimizer.state_discretization import discretize_state
from app.pipeline.validation_gates import (
    DriftGate,
    MaxNullRateGate,
    SchemaConformanceGate,
    run_gates,
)
from app.workers.celery_app import celery_app

log = logging.getLogger(__name__)

_AGENT_ORCHESTRATOR_URL = os.environ.get("AGENT_ORCHESTRATOR_URL", "http://localhost:8001")
_SUMMARIZER_TIMEOUT_S = int(os.environ.get("SUMMARIZER_TIMEOUT_S", "120"))


def _backoff(retries: int, base: int = 5) -> int:
    """Calculate exponential backoff delay in seconds."""
    return base * (2**retries)


def _update_run_status(conn: psycopg.Connection, run_id: str, status: str) -> None:
    """Update runs row status in Postgres."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE runs SET status = %s, updated_at = now() WHERE id = %s",
            (status, run_id),
        )
    conn.commit()


def _step_exists(
    conn: psycopg.Connection,
    run_id: str,
    step_type: str,
    input_hash: str,
) -> str | None:
    """Check if pipeline step has already been recorded for idempotency."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT output_hash FROM pipeline_steps
            WHERE run_id = %s AND step_type = %s AND input_hash = %s
            LIMIT 1
            """,
            (run_id, step_type, input_hash),
        )
        row = cur.fetchone()
    return row[0] if row else None


@celery_app.task(
    name="app.workers.tasks.run_validation_gates",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def run_validation_gates(
    self: Task,
    run_id: str,
    dataset_rows: list[dict],
    dataset_columns: list[str],
    *,
    reference_rows: list[dict] | None = None,
    reference_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Execute dataset validation gate chain and record lineage."""
    log.info("run_validation_gates: run_id=%s", run_id)
    try:
        conn = get_connection()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))

    try:
        _update_run_status(conn, run_id, "running")

        df = pd.DataFrame(dataset_rows, columns=dataset_columns)
        input_hash = hash_dataframe(df)

        existing = _step_exists(conn, run_id, "validation_gates", input_hash)
        if existing is not None:
            log.info("run_validation_gates: idempotent skip run_id=%s", run_id)
            _update_run_status(conn, run_id, "gate-check")
            return {"status": "skipped", "output_hash": existing, "passed": True}

        cfg = load_pipeline_config()
        gates = [
            MaxNullRateGate(cfg.max_null_rate_gate),
            SchemaConformanceGate(cfg.schema_conformance_gate),
        ]
        ref_df: pd.DataFrame | None = None
        if reference_rows and reference_columns:
            ref_df = pd.DataFrame(reference_rows, columns=reference_columns)
            gates.append(DriftGate(cfg.drift_gate))

        gate_result = run_gates(gates, df, reference_df=ref_df)

        recorder = LineageRecorder(conn)
        recorder.record_gate_chain(run_id, gate_result)

        if not gate_result.passed:
            _update_run_status(conn, run_id, "failed")
            failures = [{"gate": r.gate_name, "reason": r.reason} for r in gate_result.failures]
            log.warning("run_validation_gates: gate REJECTED run_id=%s failures=%s", run_id, failures)
            self.update_state(state="FAILURE", meta={"run_id": run_id, "gate_failures": failures})
            raise Ignore()

        output_hash = hash_dataframe(df)
        recorder.record_pipeline_step(
            run_id,
            step_type="validation_gates",
            input_hash=input_hash,
            output_hash=output_hash,
            params={"gates": [g.name for g in gates]},
            seed=None,
            transform_code_hash=hash_source(run_gates),
            description="validation gate chain",
        )

        _update_run_status(conn, run_id, "gate-check")
        conn.commit()
        return {
            "status": "done",
            "output_hash": output_hash,
            "passed": True,
            "run_id": run_id,
            "dataset_rows": dataset_rows,
            "dataset_columns": dataset_columns,
        }
    except Ignore:
        raise
    except Exception as exc:
        log.exception("run_validation_gates: error run_id=%s", run_id)
        try:
            _update_run_status(conn, run_id, "failed")
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))
    finally:
        conn.close()


@celery_app.task(
    name="app.workers.tasks.run_imputation",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def run_imputation(
    self: Task,
    run_id: str,
    dataset_rows: list[dict],
    dataset_columns: list[str],
    *,
    method: str = "mice",
    seed: int = 42,
) -> dict[str, Any]:
    """Execute MICE or KNN missing value imputation on dataset."""
    log.info("run_imputation: run_id=%s method=%s", run_id, method)
    try:
        conn = get_connection()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))

    try:
        _update_run_status(conn, run_id, "running")

        df = pd.DataFrame(dataset_rows, columns=dataset_columns)
        input_hash = hash_dataframe(df)

        existing = _step_exists(conn, run_id, "imputation", input_hash)
        if existing is not None:
            log.info("run_imputation: idempotent skip run_id=%s", run_id)
            cfg = ImputationConfig(method=method)
            result = impute(df, cfg, seed=seed)
            return {
                "status": "skipped",
                "output_hash": existing,
                "dataset_rows": result.dataframe.to_dict(orient="records"),
                "dataset_columns": list(result.dataframe.columns),
            }

        cfg = ImputationConfig(method=method)
        result = impute(df, cfg, seed=seed)

        output_hash = hash_dataframe(result.dataframe)
        recorder = LineageRecorder(conn)
        recorder.record_pipeline_step(
            run_id,
            step_type="imputation",
            input_hash=input_hash,
            output_hash=output_hash,
            params=result.params,
            seed=seed,
            transform_code_hash=hash_source(impute),
            description=f"imputation method={method}",
        )

        conn.commit()
        return {
            "status": "done",
            "output_hash": output_hash,
            "run_id": run_id,
            "dataset_rows": result.dataframe.to_dict(orient="records"),
            "dataset_columns": list(result.dataframe.columns),
        }
    except Exception as exc:
        log.exception("run_imputation: error run_id=%s", run_id)
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))
    finally:
        conn.close()


@celery_app.task(
    name="app.workers.tasks.run_outlier_detection",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def run_outlier_detection(
    self: Task,
    run_id: str,
    dataset_rows: list[dict],
    dataset_columns: list[str],
    *,
    method: str = "isolation_forest",
    seed: int = 42,
) -> dict[str, Any]:
    """Execute IsolationForest or LOF outlier detection on numeric features."""
    log.info("run_outlier_detection: run_id=%s method=%s", run_id, method)
    try:
        conn = get_connection()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))

    try:
        _update_run_status(conn, run_id, "running")

        df = pd.DataFrame(dataset_rows, columns=dataset_columns)
        input_hash = hash_dataframe(df)

        existing = _step_exists(conn, run_id, "outlier_detection", input_hash)
        if existing is not None:
            log.info("run_outlier_detection: idempotent skip run_id=%s", run_id)
            cfg = OutlierDetectionConfig(method=method)
            result = detect_outliers(df, cfg, seed=seed)
            return {
                "status": "skipped",
                "output_hash": existing,
                "dataset_rows": result.dataframe.to_dict(orient="records"),
                "dataset_columns": list(result.dataframe.columns),
            }

        cfg = OutlierDetectionConfig(method=method)
        result = detect_outliers(df, cfg, seed=seed)

        output_hash = hash_dataframe(result.dataframe)
        recorder = LineageRecorder(conn)
        recorder.record_pipeline_step(
            run_id,
            step_type="outlier_detection",
            input_hash=input_hash,
            output_hash=output_hash,
            params=result.params,
            seed=seed,
            transform_code_hash=hash_source(detect_outliers),
            description=f"outlier detection method={method}",
        )

        conn.commit()
        return {
            "status": "done",
            "output_hash": output_hash,
            "run_id": run_id,
            "dataset_rows": result.dataframe.to_dict(orient="records"),
            "dataset_columns": list(result.dataframe.columns),
            "diagnostics": result.diagnostics,
        }
    except Exception as exc:
        log.exception("run_outlier_detection: error run_id=%s", run_id)
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))
    finally:
        conn.close()


@celery_app.task(
    name="app.workers.tasks.run_estimation",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    soft_time_limit=14400,
    time_limit=14700,
)
def run_estimation(
    self: Task,
    run_id: str,
    dataset_rows: list[dict],
    dataset_columns: list[str],
    *,
    target_column: str,
    task_type: str,
    seed: int = 42,
    n_trials: int = 30,
    cv_folds: int = 5,
    stacking_cv_folds: int = 5,
) -> dict[str, Any]:
    """Execute Optuna HPO and stacked ensemble training."""
    log.info("run_estimation: run_id=%s task_type=%s n_trials=%d", run_id, task_type, n_trials)
    try:
        conn = get_connection()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))

    try:
        _update_run_status(conn, run_id, "optimizing")

        df = pd.DataFrame(dataset_rows, columns=dataset_columns)
        input_hash = hash_dataframe(df)

        existing = _step_exists(conn, run_id, "estimation", input_hash)
        if existing is not None:
            log.info("run_estimation: idempotent skip run_id=%s", run_id)
            return {"status": "skipped", "output_hash": existing, "run_id": run_id}

        X = df.drop(columns=[target_column])
        y = df[target_column].to_numpy()

        cfg = OptunaSearchConfig(n_trials=n_trials, cv_folds=cv_folds, seed=seed)
        result = run_stacking(
            X,
            y,
            task_type,
            families=["xgboost", "lightgbm", "random_forest", "linear"],
            config=cfg,
            seed=seed,
            stacking_cv_folds=stacking_cv_folds,
        )

        recorder = LineageRecorder(conn)
        recorder.record_study_trials(run_id, result.all_trials)
        recorder.record_metric(
            run_id,
            name="stacking_cv_score",
            value=result.stacking_cv_score,
        )
        recorder.record_metric(
            run_id,
            name=f"best_family_{result.single_best_family}_cv_score",
            value=result.single_best_cv_score,
        )

        output_hash = hash_dataframe(X)
        recorder.record_pipeline_step(
            run_id,
            step_type="estimation",
            input_hash=input_hash,
            output_hash=output_hash,
            params={
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "stacking_cv_folds": stacking_cv_folds,
                "seed": seed,
                "task_type": task_type,
                "target_column": target_column,
                "best_family": result.single_best_family,
                "stacking_cv_score": result.stacking_cv_score,
            },
            seed=seed,
            transform_code_hash=hash_source(run_stacking),
            description="Bayesian HPO + stacked ensemble",
        )

        _update_run_status(conn, run_id, "done")
        conn.commit()
        return {
            "status": "done",
            "output_hash": output_hash,
            "run_id": run_id,
            "stacking_cv_score": result.stacking_cv_score,
            "best_family": result.single_best_family,
        }
    except Exception as exc:
        log.exception("run_estimation: error run_id=%s", run_id)
        try:
            _update_run_status(conn, run_id, "failed")
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))
    finally:
        conn.close()


@celery_app.task(
    name="app.workers.tasks.run_rl_episode",
    bind=True,
    max_retries=2,
    default_retry_delay=15,
    acks_late=True,
    soft_time_limit=7200,
    time_limit=7500,
)
def run_rl_episode(
    self: Task,
    run_id: str,
    episode_number: int,
    dataset_rows: list[dict],
    dataset_columns: list[str],
    *,
    target_column: str,
    task_type: str,
    seed: int = 42,
    fast_surrogate: bool = True,
    q_table: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Execute single RL training episode for preprocessing optimization."""
    log.info("run_rl_episode: run_id=%s episode=%d", run_id, episode_number)
    try:
        conn = get_connection()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rl_episodes WHERE run_id = %s AND episode_number = %s",
                (run_id, episode_number),
            )
            if cur.fetchone():
                log.info("run_rl_episode: idempotent skip run_id=%s episode=%d", run_id, episode_number)
                return {
                    "status": "skipped",
                    "episode_number": episode_number,
                    "q_table": q_table or {},
                }

        df = pd.DataFrame(dataset_rows, columns=dataset_columns)
        X = df.drop(columns=[target_column])
        y = df[target_column].to_numpy()

        if fast_surrogate:
            reward_fn = fast_surrogate_reward_fn(cv_folds=3, seed=seed)
        else:
            cfg = OptunaSearchConfig(n_trials=10, cv_folds=3, seed=seed)
            reward_fn = full_stack_reward_fn(cfg, stacking_cv_folds=3, seed=seed)

        env = PreprocessingEnv(reward_fn, seed=seed)
        actions = build_action_space()
        agent = QLearningAgent(
            actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=max(0.3 * (0.98**episode_number), 0.01),
            epsilon_decay=0.98,
            seed=seed,
        )
        if q_table:
            agent.q_table = {k: list(v) for k, v in q_table.items()}

        state_features = env.reset(X, y, task_type)
        state_key = discretize_state(state_features)

        def step_fn(_state, action):
            result = env.step(action)
            return result.reward, result.info

        record = agent.run_episode(episode_number, state_key, step_fn)

        recorder = LineageRecorder(conn)
        action = actions[record.action_index]
        recorder.record_rl_episode(
            episode_number=episode_number,
            state={"state_key": state_key, "meta_features": state_features},
            action={"index": record.action_index, "action": str(action)},
            reward=record.reward,
            run_id=run_id,
        )

        conn.commit()
        return {
            "status": "done",
            "episode_number": episode_number,
            "reward": record.reward,
            "action": str(action),
            "q_table": agent.q_table,
        }
    except Exception as exc:
        log.exception("run_rl_episode: error run_id=%s episode=%d", run_id, episode_number)
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))
    finally:
        conn.close()


@celery_app.task(
    name="app.workers.tasks.run_maml_adaptation",
    bind=True,
    max_retries=2,
    default_retry_delay=15,
    acks_late=True,
    soft_time_limit=3600,
    time_limit=3900,
)
def run_maml_adaptation(
    self: Task,
    run_id: str,
    batch_rows: list[dict],
    batch_columns: list[str],
    reference_rows: list[dict],
    reference_columns: list[str],
    *,
    target_column: str,
    task_type: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute drift check and MAML fast-adaptation step on data batch."""
    log.info("run_maml_adaptation: run_id=%s", run_id)
    try:
        conn = get_connection()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))

    try:
        batch_df = pd.DataFrame(batch_rows, columns=batch_columns)
        ref_df = pd.DataFrame(reference_rows, columns=reference_columns)
        batch_hash = hash_dataframe(batch_df)

        existing = _step_exists(conn, run_id, "maml_adaptation", batch_hash)
        if existing is not None:
            log.info("run_maml_adaptation: idempotent skip run_id=%s", run_id)
            return {"status": "skipped", "output_hash": existing}

        X_ref_num = ref_df.select_dtypes(include="number").to_numpy(dtype=float)
        n_features = X_ref_num.shape[1]
        learner = MAMLLearner(
            input_dim=n_features,
            task_type=task_type,
            inner_lr=0.01,
            inner_steps=5,
        )

        y_col = batch_df[target_column].to_numpy()
        batch_features = batch_df.drop(columns=[target_column])

        initial_state = AdaptiveState(
            feature_mask=np.ones(n_features),
            adapted_params=learner.meta_params,
            reference_df=ref_df,
        )

        cfg = load_pipeline_config()
        step_result = run_adaptive_step(
            batch_features,
            y_col,
            initial_state,
            learner,
            drift_config=cfg.drift_gate,
            genetic_config=cfg.meta_learning.genetic_selector,
            seed=seed,
        )

        output_hash = hash_dataframe(batch_df)
        recorder = LineageRecorder(conn)
        recorder.record_pipeline_step(
            run_id,
            step_type="maml_adaptation",
            input_hash=batch_hash,
            output_hash=output_hash,
            params={
                "drift_detected": step_result.drift_detected,
                "adapted_this_step": step_result.adapted_this_step,
                "seed": seed,
                "task_type": task_type,
            },
            seed=seed,
            transform_code_hash=hash_source(run_adaptive_step),
            description="MAML fast-adaptation step",
        )

        conn.commit()
        return {
            "status": "done",
            "output_hash": output_hash,
            "run_id": run_id,
            "drift_detected": step_result.drift_detected,
            "adapted_this_step": step_result.adapted_this_step,
            "diagnostics": step_result.diagnostics,
        }
    except Exception as exc:
        log.exception("run_maml_adaptation: error run_id=%s", run_id)
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))
    finally:
        conn.close()


@celery_app.task(
    name="app.workers.tasks.run_summarizer",
    bind=True,
    max_retries=3,
    default_retry_delay=20,
    acks_late=True,
    soft_time_limit=300,
    time_limit=360,
)
def run_summarizer(
    self: Task,
    run_id: str,
    metrics: list[dict],
) -> dict[str, Any]:
    """Invoke agent-orchestrator summarizer service and persist report in lineage."""
    log.info("run_summarizer: run_id=%s n_metrics=%d", run_id, len(metrics))
    try:
        conn = get_connection()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))

    try:
        existing = _step_exists(conn, run_id, "summarizer", run_id)
        if existing is not None:
            log.info("run_summarizer: idempotent skip run_id=%s", run_id)
            return {"status": "skipped", "output_hash": existing}

        url = f"{_AGENT_ORCHESTRATOR_URL.rstrip('/')}/summarize"
        try:
            resp = httpx.post(
                url,
                json={"run_id": run_id, "metrics": metrics},
                timeout=_SUMMARIZER_TIMEOUT_S,
            )
            resp.raise_for_status()
            report = resp.json()
        except httpx.HTTPStatusError as exc:
            log.error("run_summarizer: orchestrator status %d for run %s", exc.response.status_code, run_id)
            raise self.retry(exc=exc, countdown=_backoff(self.request.retries))
        except httpx.RequestError as exc:
            raise self.retry(exc=exc, countdown=_backoff(self.request.retries))

        output_hash = "sha256:" + hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()

        recorder = LineageRecorder(conn)
        recorder.record_pipeline_step(
            run_id,
            step_type="summarizer",
            input_hash=run_id,
            output_hash=output_hash,
            params={"n_metrics": len(metrics)},
            seed=None,
            transform_code_hash="agent-orchestrator:summarize",
            description="bounded LLM summarizer (agent-orchestrator)",
        )
        recorder.record_metric(
            run_id,
            name="summary_n_emitted_claims",
            value=float(len(report.get("emitted_claims", []))),
        )
        recorder.record_metric(
            run_id,
            name="summary_n_flagged_claims",
            value=float(len(report.get("flagged_claims", []))),
        )

        conn.commit()
        return {"status": "done", "output_hash": output_hash, "run_id": run_id, "report": report}
    except Exception as exc:
        log.exception("run_summarizer: error run_id=%s", run_id)
        raise self.retry(exc=exc, countdown=_backoff(self.request.retries))
    finally:
        conn.close()
