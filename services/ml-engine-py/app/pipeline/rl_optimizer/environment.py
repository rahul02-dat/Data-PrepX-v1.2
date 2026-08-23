from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.pipeline.config import ImputationConfig, OutlierDetectionConfig
from app.pipeline.estimation.optuna_search import TaskType
from app.pipeline.imputation import impute
from app.pipeline.outliers import IS_OUTLIER_COLUMN, detect_outliers
from app.pipeline.rl_optimizer.meta_features import MetaFeatures, compute_meta_features

ImputerChoice = Literal["mice", "knn"]
OutlierChoice = Literal["isolation_forest", "lof", "none"]

THRESHOLD_BINS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
_IMPUTER_COST = {"mice": 0.02, "knn": 0.005}
_OUTLIER_COST = {"isolation_forest": 0.005, "lof": 0.01, "none": 0.0}


@dataclass(frozen=True)
class Action:
    imputer: ImputerChoice
    outlier_method: OutlierChoice
    threshold_bin: int

    def as_dict(self) -> dict[str, Any]:
        """Convert Action to dictionary representation."""
        return {
            "imputer": self.imputer,
            "outlier_method": self.outlier_method,
            "threshold_bin": self.threshold_bin,
            "contamination": (
                None if self.outlier_method == "none" else THRESHOLD_BINS[self.threshold_bin]
            ),
        }

    def compute_cost(self) -> float:
        """Compute relative compute cost penalty for action choices."""
        return _IMPUTER_COST[self.imputer] + _OUTLIER_COST[self.outlier_method]


def build_action_space() -> list[Action]:
    """Construct discrete action space combining imputation and outlier detection options."""
    actions: list[Action] = []
    for imputer in ("mice", "knn"):
        for outlier_method in ("isolation_forest", "lof"):
            for bin_idx in range(len(THRESHOLD_BINS)):
                actions.append(Action(imputer, outlier_method, bin_idx))
        actions.append(Action(imputer, "none", 0))
    return actions


def apply_action(
    X: pd.DataFrame, y: np.ndarray, action: Action, *, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply imputation and outlier detection defined by action to dataset."""
    imputation_config = ImputationConfig(method=action.imputer)
    imputed = impute(X, imputation_config, seed=seed).dataframe

    if action.outlier_method == "none":
        return imputed.to_numpy(dtype=float), y

    outlier_config = OutlierDetectionConfig(
        method=action.outlier_method,
        contamination=THRESHOLD_BINS[action.threshold_bin],
    )
    outlier_result = detect_outliers(imputed, outlier_config, seed=seed)
    keep_mask = ~outlier_result.dataframe[IS_OUTLIER_COLUMN].to_numpy()

    if keep_mask.sum() < max(10, int(0.5 * len(imputed))):
        return imputed.to_numpy(dtype=float), y

    X_kept = imputed.loc[keep_mask, X.columns].to_numpy(dtype=float)
    y_kept = np.asarray(y)[keep_mask]
    return X_kept, y_kept


RewardFn = Callable[[np.ndarray, np.ndarray, TaskType], float]


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    info: dict[str, Any]


class PreprocessingEnv:
    """Contextual bandit environment for preprocessing policy optimization."""

    def __init__(self, reward_fn: RewardFn, *, seed: int | None = None):
        self._reward_fn = reward_fn
        self._seed = seed
        self._X: pd.DataFrame | None = None
        self._y: np.ndarray | None = None
        self._task: TaskType | None = None
        self._baseline_score: float | None = None

    def reset(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task: TaskType,
        *,
        reference_df: pd.DataFrame | None = None,
    ) -> MetaFeatures:
        """Reset environment, calculate baseline score, and return dataset meta-features."""
        self._X = X

        self._y = y
        self._task = task

        baseline_X = X.fillna(X.mean(numeric_only=True))
        for col in baseline_X.columns:
            if baseline_X[col].isna().any():
                baseline_X[col] = baseline_X[col].fillna(baseline_X[col].mode().iloc[0])
        self._baseline_score = self._reward_fn(baseline_X.to_numpy(dtype=float), y, task)

        return compute_meta_features(X, target=y, reference_df=reference_df)

    def step(self, action: Action) -> StepResult:
        """Execute selected preprocessing action and compute reward delta over baseline."""
        if self._X is None or self._y is None or self._task is None:
            raise RuntimeError("call reset() before step()")

        X_processed, y_processed = apply_action(self._X, self._y, action, seed=self._seed)
        score = self._reward_fn(X_processed, y_processed, self._task)
        cost_penalty = action.compute_cost()
        reward = (score - self._baseline_score) - cost_penalty

        info = {
            "score": score,
            "baseline_score": self._baseline_score,
            "cost_penalty": cost_penalty,
            "rows_kept": len(y_processed),
            "rows_total": len(self._y),
        }
        return StepResult(reward=reward, done=True, info=info)
