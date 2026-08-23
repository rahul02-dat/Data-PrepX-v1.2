"""Smoke test lineage persistence against live Postgres instance."""

from __future__ import annotations

import sys

import pandas as pd

from app.pipeline.config import PipelineConfig
from app.pipeline.db import get_connection
from app.pipeline.lineage import LineageRecorder


class _FakeTrial:
    def __init__(self, family: str, number: int, params: dict, score: float):
        self.model_family = family
        self.trial_number = number
        self.params = params
        self.score = score


def main() -> int:
    conn = get_connection()
    conn.autocommit = True
    recorder = LineageRecorder(conn)

    print("1. Registering a dataset...")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
    dataset_id, content_hash = recorder.register_dataset(df, {"a": "float64", "b": "float64"})
    print(f"   dataset_id={dataset_id} content_hash={content_hash[:16]}...")

    print("2. Creating a run...")
    run_id, run_key, created = recorder.get_or_create_run(
        dataset_id=dataset_id,
        dataset_content_hash=content_hash,
        config=PipelineConfig(),
        git_sha="smoke-test-sha",
    )
    print(f"   run_id={run_id} created={created}")

    print("3. Recording individual hyperparameter trial (Phase 4)...")
    trial_id = recorder.record_hyperparameter_trial(
        run_id,
        model_family="xgboost",
        trial_number=0,
        params={"max_depth": 5, "learning_rate": 0.1},
        score=0.87,
    )
    print(f"   trial_id={trial_id}")

    print("4. Bulk-recording a study's trials (Phase 4)...")
    trials = [
        _FakeTrial("xgboost", 1, {"max_depth": 3}, 0.81),
        _FakeTrial("linear", 0, {"C": 1.0}, 0.75),
    ]
    trial_ids = recorder.record_study_trials(run_id, trials)
    print(f"   trial_ids={trial_ids}")

    print("5. Recording a metric with CI bounds (Phase 4)...")
    metric_id = recorder.record_metric(
        run_id, name="stacking_cv_score", value=0.91, ci_low=0.88, ci_high=0.94
    )
    print(f"   metric_id={metric_id}")

    print("6. Recording a metric with no CI bounds...")
    metric_id_2 = recorder.record_metric(run_id, name="default_baseline_score", value=0.75)
    print(f"   metric_id={metric_id_2}")

    print("7. Recording an RL episode WITH a run_id (Phase 5)...")
    episode_id = recorder.record_rl_episode(
        episode_number=0,
        state={"missingness_rate": 0.1, "mean_abs_skew": 0.5},
        action={"imputer": "mice", "outlier_method": "lof", "threshold_bin": 2},
        reward=0.034,
        run_id=run_id,
    )
    print(f"   episode_id={episode_id}")

    print("8. Recording an RL episode WITHOUT a run_id (surrogate reward case)...")
    episode_id_2 = recorder.record_rl_episode(
        episode_number=1,
        state={"missingness_rate": 0.3},
        action={"imputer": "knn", "outlier_method": "none", "threshold_bin": 0},
        reward=-0.01,
        run_id=None,
    )
    print(f"   episode_id={episode_id_2}")

    print("\n9. Reading everything back to confirm it actually persisted...")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM hyperparameters WHERE run_id = %s", (run_id,))
        n_trials = cur.fetchone()[0]
        print(f"   hyperparameters rows for this run: {n_trials} (expected 3)")
        assert n_trials == 3, f"expected 3 hyperparameter rows, got {n_trials}"

        cur.execute("SELECT count(*) FROM metrics WHERE run_id = %s", (run_id,))
        n_metrics = cur.fetchone()[0]
        print(f"   metrics rows for this run: {n_metrics} (expected 2)")
        assert n_metrics == 2, f"expected 2 metric rows, got {n_metrics}"

        cur.execute("SELECT count(*) FROM rl_episodes")
        n_episodes = cur.fetchone()[0]
        print(f"   rl_episodes rows total: {n_episodes} (expected 2)")
        assert n_episodes == 2, f"expected 2 rl_episodes rows, got {n_episodes}"

        cur.execute("SELECT run_id FROM rl_episodes WHERE episode_number = 1")
        null_run_id = cur.fetchone()[0]
        print(f"   surrogate episode's run_id: {null_run_id} (expected None)")
        assert null_run_id is None, f"expected NULL run_id, got {null_run_id}"

    print("\nALL CHECKS PASSED -- the new lineage SQL is valid against real Postgres.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
