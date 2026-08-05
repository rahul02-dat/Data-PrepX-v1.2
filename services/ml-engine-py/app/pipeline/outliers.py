from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from app.pipeline.config import OutlierDetectionConfig

ANOMALY_SCORE_COLUMN = "_anomaly_score"
IS_OUTLIER_COLUMN = "_is_outlier"


@dataclass(frozen=True)
class OutlierResult:
    # Original columns plus anomaly indicators. Flagged rows are retained for RL threshold learning.
    dataframe: pd.DataFrame
    params: dict[str, Any]
    diagnostics: dict[str, Any]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# Detect outliers via Isolation Forest, LOF, or a no-op, without dropping any rows.
def detect_outliers(
    df: pd.DataFrame, config: OutlierDetectionConfig | None = None, *, seed: int | None = None
) -> OutlierResult:
    config = config or OutlierDetectionConfig()

    if len(df) == 0:
        raise ValueError("cannot run outlier detection on an empty dataframe")

    numeric_cols = _numeric_columns(df)
    out = df.copy()

    if config.method == "none" or not numeric_cols:
        out[ANOMALY_SCORE_COLUMN] = 0.0
        out[IS_OUTLIER_COLUMN] = False
        diagnostics = {
            "method": config.method,
            "numeric_columns_used": numeric_cols,
            "n_flagged": 0,
            "reason": "no numeric columns available" if not numeric_cols else None,
        }
        params = {"method": config.method, "seed": seed, "numeric_columns": numeric_cols}
        return OutlierResult(dataframe=out, params=params, diagnostics=diagnostics)

    numeric_block = df[numeric_cols].to_numpy(dtype=float)
    if np.isnan(numeric_block).any():
        raise ValueError(
            "outlier detection requires fully imputed numeric columns; "
            "run imputation before detect_outliers"
        )

    if config.method == "isolation_forest":
        model = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=seed,
        )
        model.fit(numeric_block)
        # Higher score = more anomalous.
        raw_scores = -model.decision_function(numeric_block)
        predictions = model.predict(numeric_block)  # -1 = outlier, 1 = inlier
        is_outlier = predictions == -1
        diagnostics = {
            "method": "isolation_forest",
            "n_estimators": config.n_estimators,
            "contamination": config.contamination,
        }
    else:  # "lof"
        model = LocalOutlierFactor(
            n_neighbors=min(config.lof_n_neighbors, len(df) - 1),
            contamination=config.contamination,
        )
        predictions = model.fit_predict(numeric_block)  # -1 = outlier, 1 = inlier
        raw_scores = -model.negative_outlier_factor_
        is_outlier = predictions == -1
        diagnostics = {
            "method": "lof",
            "lof_n_neighbors": min(config.lof_n_neighbors, len(df) - 1),
            "contamination": config.contamination,
        }

    out[ANOMALY_SCORE_COLUMN] = raw_scores
    out[IS_OUTLIER_COLUMN] = is_outlier
    diagnostics["numeric_columns_used"] = numeric_cols
    diagnostics["n_flagged"] = int(is_outlier.sum())
    diagnostics["flag_rate"] = float(is_outlier.sum() / len(df))

    params = {
        "method": config.method,
        "contamination": config.contamination,
        "n_estimators": config.n_estimators,
        "lof_n_neighbors": config.lof_n_neighbors,
        "seed": seed,
        "numeric_columns": numeric_cols,
    }

    return OutlierResult(dataframe=out, params=params, diagnostics=diagnostics)
