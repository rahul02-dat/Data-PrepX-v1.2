from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from app.pipeline.estimation.optuna_search import (
    OptunaSearchConfig,
    TaskType,
    cv_splitter,
    scoring_for_task,
)
from app.pipeline.estimation.stacking import run_stacking
from app.pipeline.rl_optimizer.environment import RewardFn


# Production reward (per project decision): every episode calls the full Phase 4 Optuna+stacking
# pipeline (4 model families, config.n_trials each, cross-validated) and uses the resulting
# stack's CV score as the reward signal. This is the most accurate reward available, and the
# most expensive -- see docs/adr/0005-rl-reward-cost.md for the actual per-episode cost this
# implies and why it was chosen anyway. Do not use this for quick local iteration on the RL code
# itself; use fast_surrogate_reward_fn for that (see below), and switch to this only for a real
# training run.
def full_stack_reward_fn(
    config: OptunaSearchConfig, *, stacking_cv_folds: int = 5, seed: int = 42
) -> RewardFn:
    def _reward(X: np.ndarray, y: np.ndarray, task: TaskType) -> float:
        result = run_stacking(
            X, y, task, config=config, seed=seed, stacking_cv_folds=stacking_cv_folds
        )
        return result.stacking_cv_score

    return _reward


# The planner's explicitly recommended alternative (CLAUDE.md §5.1, planner Phase 5 Notes/Risks:
# "Use a cheap surrogate metric ... during RL training, and only validate the final learned
# policy against the full pipeline"). NOT the default -- provided so the two-tier design is at
# least implemented and testable even though the project's current choice is full-stack-always.
# A single untuned RandomForest with reduced cv is ~50x cheaper than full_stack_reward_fn per
# the Phase 4 benchmark's own timing (see docs/research/optuna_stacking_benchmark.md).
def fast_surrogate_reward_fn(*, cv_folds: int = 3, seed: int = 42) -> RewardFn:
    def _reward(X: np.ndarray, y: np.ndarray, task: TaskType) -> float:
        model = (
            RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
            if task == "classification"
            else RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1)
        )
        splitter = cv_splitter(task, cv_folds, seed)
        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring_for_task(task), n_jobs=1)
        return float(np.mean(scores))

    return _reward
