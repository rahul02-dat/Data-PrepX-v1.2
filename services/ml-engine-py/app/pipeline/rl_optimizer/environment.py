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

# Contamination values the outlier-threshold action dimension chooses between (CLAUDE.md §5.1:
# "outlier-threshold bin"). Kept as an explicit, documented list rather than a continuous
# parameter so the action space stays discrete and finite, matching the tabular-Q design.
THRESHOLD_BINS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)

# A rough, explicit compute-cost ranking used for the reward's cost penalty (CLAUDE.md §5.1:
# "reward = Delta(validation metric) ... minus a compute-cost penalty"). MICE is iterative and
# markedly more expensive than KNN; LOF is O(n^2)-ish and costlier than Isolation Forest; "none"
# is free. These are relative weights, not measured wall-clock numbers -- recalibrate against
# real timings once Phase 8's async execution makes per-action cost directly measurable.
_IMPUTER_COST = {"mice": 0.02, "knn": 0.005}
_OUTLIER_COST = {"isolation_forest": 0.005, "lof": 0.01, "none": 0.0}


@dataclass(frozen=True)
class Action:
    imputer: ImputerChoice
    outlier_method: OutlierChoice
    threshold_bin: int  # index into THRESHOLD_BINS; ignored (but still 0) when method is "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "imputer": self.imputer,
            "outlier_method": self.outlier_method,
            "threshold_bin": self.threshold_bin,
            "contamination": (
                None if self.outlier_method == "none" else THRESHOLD_BINS[self.threshold_bin]
            ),
        }

    def compute_cost(self) -> float:
        return _IMPUTER_COST[self.imputer] + _OUTLIER_COST[self.outlier_method]


# Build the full discrete action space: {mice, knn} x {isolation_forest, lof, none} x
# threshold_bin (threshold_bin only varies when an outlier method is actually selected).
def build_action_space() -> list[Action]:
    actions: list[Action] = []
    for imputer in ("mice", "knn"):
        for outlier_method in ("isolation_forest", "lof"):
            for bin_idx in range(len(THRESHOLD_BINS)):
                actions.append(Action(imputer, outlier_method, bin_idx))
        actions.append(Action(imputer, "none", 0))
    return actions


# Apply one action's imputation + outlier handling to a feature dataframe, returning the
# resulting (X, y) with flagged-outlier rows dropped (a chosen outlier method with no
# behavioral effect wouldn't be a meaningful action for the agent to learn between).
def apply_action(
    X: pd.DataFrame, y: np.ndarray, action: Action, *, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
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

    # Guard against an action flagging away nearly everything on a small/unlucky sample --
    # falling back to "keep all rows" is safer than returning a near-empty training set that
    # would make the reward computation itself fail or be meaningless.
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
    """Gymnasium-style single-step environment (CLAUDE.md §5.1). One episode = one dataset:
    reset() observes its meta-features as state, step(action) applies that action's
    imputation/outlier pipeline, scores it with `reward_fn`, and terminates immediately.
    There is no natural multi-step transition here (applying a pipeline doesn't produce a "next
    dataset" to act on again), so this is a contextual bandit expressed as a 1-step MDP --
    Q-learning's max_a' Q(s',a') term is simply 0 on every update, which QLearningAgent.update
    handles via next_state_key=None rather than needing a separate code path.
    """

    def __init__(self, reward_fn: RewardFn, *, seed: int | None = None):
        self._reward_fn = reward_fn
        self._seed = seed
        self._X: pd.DataFrame | None = None
        self._y: np.ndarray | None = None
        self._task: TaskType | None = None
        self._baseline_score: float | None = None

    # Observe a new dataset's meta-features and cache the no-op-imputed baseline score
    def reset(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task: TaskType,
        *,
        reference_df: pd.DataFrame | None = None,
    ) -> MetaFeatures:
        self._X = X
        self._y = y
        self._task = task

        # No-op baseline: plain mean/mode fill, deliberately NOT one of the two imputers the
        # agent can choose between, so its score is a genuine "did nothing clever" baseline
        # rather than accidentally matching one of the real actions.
        baseline_X = X.fillna(X.mean(numeric_only=True))
        for col in baseline_X.columns:
            if baseline_X[col].isna().any():
                baseline_X[col] = baseline_X[col].fillna(baseline_X[col].mode().iloc[0])
        self._baseline_score = self._reward_fn(baseline_X.to_numpy(dtype=float), y, task)

        return compute_meta_features(X, target=y, reference_df=reference_df)

    def step(self, action: Action) -> StepResult:
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
