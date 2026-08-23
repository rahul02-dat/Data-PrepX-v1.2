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


def full_stack_reward_fn(
    config: OptunaSearchConfig, *, stacking_cv_folds: int = 5, seed: int = 42
) -> RewardFn:
    """Build reward function executing Bayesian HPO and stacked ensemble evaluation."""
    def _reward(X: np.ndarray, y: np.ndarray, task: TaskType) -> float:
        result = run_stacking(
            X, y, task, config=config, seed=seed, stacking_cv_folds=stacking_cv_folds
        )
        return result.stacking_cv_score

    return _reward


def fast_surrogate_reward_fn(*, cv_folds: int = 3, seed: int = 42) -> RewardFn:
    """Build fast surrogate reward function evaluating a single random forest."""
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
