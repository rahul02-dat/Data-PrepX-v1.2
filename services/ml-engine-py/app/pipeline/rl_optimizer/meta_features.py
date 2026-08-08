from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.pipeline.config import DriftGateConfig
from app.pipeline.validation_gates import DriftGate


@dataclass(frozen=True)
class MetaFeatures:
    """RL state (CLAUDE.md §5.1): dataset meta-features the agent conditions its action on.
    All fields are plain floats so the vector is trivially usable as a tabular-Q state key
    (after binning, see environment.py) or as linear-function-approximation features."""

    missingness_rate: float
    mean_abs_skew: float
    mean_cardinality_ratio: float
    class_imbalance_ratio: float
    drift_score: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.missingness_rate,
                self.mean_abs_skew,
                self.mean_cardinality_ratio,
                self.class_imbalance_ratio,
                self.drift_score,
            ],
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "missingness_rate": self.missingness_rate,
            "mean_abs_skew": self.mean_abs_skew,
            "mean_cardinality_ratio": self.mean_cardinality_ratio,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "drift_score": self.drift_score,
        }


# Fraction of all cells that are missing, across the whole dataframe
def _missingness_rate(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    return float(df.isna().sum().sum() / df.size)


# Mean absolute skewness across numeric columns (0 for a dataset with no numeric columns)
def _mean_abs_skew(df: pd.DataFrame) -> float:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return 0.0
    skews = numeric.skew(numeric_only=True).abs()
    skews = skews.replace([np.inf, -np.inf], np.nan).dropna()
    if skews.empty:
        return 0.0
    return float(skews.mean())


# Mean (n_unique / n_rows) across categorical/object columns; 0 if there are none
def _mean_cardinality_ratio(df: pd.DataFrame) -> float:
    categorical = df.select_dtypes(exclude=[np.number])
    if categorical.shape[1] == 0 or len(df) == 0:
        return 0.0
    ratios = [categorical[c].nunique(dropna=True) / len(df) for c in categorical.columns]
    return float(np.mean(ratios))


# max_class_count / min_class_count for a classification target; 1.0 (perfectly balanced) if
# the target is absent, empty, or has fewer than 2 observed classes
def class_imbalance_ratio(y: pd.Series | np.ndarray | None) -> float:
    if y is None:
        return 1.0
    counts = pd.Series(y).dropna().value_counts()
    if len(counts) < 2:
        return 1.0
    return float(counts.max() / counts.min())


# Drift score vs. a reference distribution, reusing the Phase 2 DriftGate's PSI computation
# directly rather than re-implementing it -- one drift definition for the whole project.
def drift_score(df: pd.DataFrame, reference_df: pd.DataFrame | None) -> float:
    if reference_df is None or len(reference_df) == 0:
        return 0.0
    gate = DriftGate(DriftGateConfig())
    result = gate.evaluate(df, reference_df=reference_df)
    if not result.details.get("per_column"):
        return 0.0
    psi_values = [v.get("psi", 0.0) for v in result.details["per_column"].values()]
    psi_values = [v for v in psi_values if np.isfinite(v)]
    return float(np.mean(psi_values)) if psi_values else 0.0


# Compute the full RL state vector for a dataset (and optional target / drift reference)
def compute_meta_features(
    df: pd.DataFrame,
    *,
    target: pd.Series | np.ndarray | None = None,
    reference_df: pd.DataFrame | None = None,
) -> MetaFeatures:
    return MetaFeatures(
        missingness_rate=_missingness_rate(df),
        mean_abs_skew=_mean_abs_skew(df),
        mean_cardinality_ratio=_mean_cardinality_ratio(df),
        class_imbalance_ratio=class_imbalance_ratio(target),
        drift_score=drift_score(df, reference_df),
    )
