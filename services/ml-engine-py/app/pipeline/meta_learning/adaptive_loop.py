from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.pipeline.config import DriftGateConfig, GeneticSelectorConfig
from app.pipeline.meta_learning.genetic_selector import (
    GeneticSelectionResult,
    run_genetic_selection,
)
from app.pipeline.meta_learning.maml import MAMLLearner, ParamDict
from app.pipeline.validation_gates import DriftGate


@dataclass
class AdaptiveState:
    feature_mask: np.ndarray
    adapted_params: ParamDict
    reference_df: pd.DataFrame
    n_adaptations: int = 0
    n_batches_seen: int = 0


@dataclass(frozen=True)
class AdaptiveStepResult:
    state: AdaptiveState
    drift_detected: bool
    adapted_this_step: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _numeric_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract numeric columns as a float array and return array with column names."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("adaptive_loop requires at least one numeric column")
    return df[numeric_cols].to_numpy(dtype=float), numeric_cols


def apply_feature_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply feature mask by zeroing out unselected feature columns."""
    return X * mask


def run_adaptive_step(
    current_batch: pd.DataFrame,
    y: np.ndarray,
    state: AdaptiveState,
    maml_learner: MAMLLearner,
    *,
    drift_config: DriftGateConfig | None = None,
    genetic_config: GeneticSelectorConfig | None = None,
    seed: int | None = None,
) -> AdaptiveStepResult:
    """Evaluate drift against reference and trigger MAML adaptation when detected."""
    drift_config = drift_config or DriftGateConfig()

    genetic_config = genetic_config or GeneticSelectorConfig()

    gate = DriftGate(drift_config)
    gate_result = gate.evaluate(current_batch, reference_df=state.reference_df)
    drift_detected = not gate_result.passed

    X, _numeric_cols = _numeric_matrix(current_batch)
    if X.shape[0] != len(y):
        raise ValueError(
            f"current_batch has {X.shape[0]} rows but y has {len(y)} entries; they must match"
        )

    if not drift_detected:
        reused_state = AdaptiveState(
            feature_mask=state.feature_mask,
            adapted_params=state.adapted_params,
            reference_df=state.reference_df,
            n_adaptations=state.n_adaptations,
            n_batches_seen=state.n_batches_seen + 1,
        )
        return AdaptiveStepResult(
            state=reused_state,
            drift_detected=False,
            adapted_this_step=False,
            diagnostics={
                "gate_result": gate_result.details,
                "reason": "no drift detected; reused existing feature set/model",
            },
        )

    def _fitness(X_arr: np.ndarray, y_arr: np.ndarray, mask: np.ndarray) -> float:
        X_masked = apply_feature_mask(X_arr, mask)
        adapted = maml_learner.adapt(X_masked, y_arr)
        preds = maml_learner.predict(X_masked, adapted)
        if maml_learner.task_type == "classification":
            return float((preds == y_arr).mean())
        return float(-np.mean((preds - y_arr) ** 2))

    ga_result: GeneticSelectionResult = run_genetic_selection(
        X, y, _fitness, genetic_config, seed=seed
    )

    X_selected = apply_feature_mask(X, ga_result.best_mask)
    adapted_params = maml_learner.adapt(X_selected, y)

    new_state = AdaptiveState(
        feature_mask=ga_result.best_mask,
        adapted_params=adapted_params,
        reference_df=current_batch,
        n_adaptations=state.n_adaptations + 1,
        n_batches_seen=state.n_batches_seen + 1,
    )

    return AdaptiveStepResult(
        state=new_state,
        drift_detected=True,
        adapted_this_step=True,
        diagnostics={
            "gate_result": gate_result.details,
            "ga_best_fitness": ga_result.best_fitness,
            "n_features_selected": int(ga_result.best_mask.sum()),
        },
    )
