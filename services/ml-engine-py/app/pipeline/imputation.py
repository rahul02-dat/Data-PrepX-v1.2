from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401  (registers IterativeImputer)
)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from app.pipeline.config import ImputationConfig

# Logs method choice and diagnostics for RL agent action space.


@dataclass(frozen=True)
class ImputationResult:
    dataframe: pd.DataFrame
    # Deterministic reconstruction params.
    params: dict[str, Any]
    # Diagnostics for audit and benchmark.
    diagnostics: dict[str, Any]


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _impute_numeric(
    df: pd.DataFrame, numeric_cols: list[str], config: ImputationConfig, seed: int | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not numeric_cols:
        return df, {"numeric_columns_imputed": 0}

    numeric_block = df[numeric_cols].to_numpy(dtype=float)
    missing_before = int(np.isnan(numeric_block).sum())

    if config.method == "mice":
        imputer = IterativeImputer(
            max_iter=config.mice_max_iter,
            tol=config.mice_tol,
            random_state=seed,
        )
        imputed = imputer.fit_transform(numeric_block)
        diagnostics: dict[str, Any] = {
            "method": "mice",
            "n_iter": int(imputer.n_iter_),
            "converged": bool(imputer.n_iter_ < config.mice_max_iter),
            "max_iter": config.mice_max_iter,
            "tol": config.mice_tol,
            "missing_values_before": missing_before,
        }
    else:  # "knn"
        imputer = KNNImputer(n_neighbors=config.knn_n_neighbors)
        imputed = imputer.fit_transform(numeric_block)
        diagnostics = {
            "method": "knn",
            "n_neighbors": config.knn_n_neighbors,
            "missing_values_before": missing_before,
        }

    out = df.copy()
    out[numeric_cols] = imputed
    diagnostics["missing_values_after"] = int(out[numeric_cols].isna().sum().sum())
    return out, diagnostics


def _impute_categorical(
    df: pd.DataFrame, categorical_cols: list[str], config: ImputationConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not categorical_cols:
        return df, {"categorical_columns_imputed": 0}

    missing_before = int(df[categorical_cols].isna().sum().sum())
    imputer = SimpleImputer(strategy=config.categorical_strategy)
    imputed = imputer.fit_transform(df[categorical_cols])

    out = df.copy()
    out[categorical_cols] = imputed
    diagnostics = {
        "categorical_strategy": config.categorical_strategy,
        "categorical_columns_imputed": len(categorical_cols),
        "missing_values_before": missing_before,
        "missing_values_after": int(out[categorical_cols].isna().sum().sum()),
    }
    return out, diagnostics


# Impute missing values: MICE or KNN on numeric columns, most-frequent on categorical columns.
def impute(
    df: pd.DataFrame, config: ImputationConfig | None = None, *, seed: int | None = None
) -> ImputationResult:
    config = config or ImputationConfig()

    if len(df) == 0:
        raise ValueError("cannot impute an empty dataframe")

    numeric_cols, categorical_cols = _split_columns(df)

    working = df
    numeric_diag: dict[str, Any] = {"numeric_columns_imputed": 0}
    categorical_diag: dict[str, Any] = {"categorical_columns_imputed": 0}

    if numeric_cols:
        working, numeric_diag = _impute_numeric(working, numeric_cols, config, seed)
        numeric_diag["numeric_columns_imputed"] = len(numeric_cols)
    if categorical_cols:
        working, categorical_diag = _impute_categorical(working, categorical_cols, config)

    params = {
        "method": config.method,
        "mice_max_iter": config.mice_max_iter,
        "mice_tol": config.mice_tol,
        "knn_n_neighbors": config.knn_n_neighbors,
        "categorical_strategy": config.categorical_strategy,
        "seed": seed,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    diagnostics = {"numeric": numeric_diag, "categorical": categorical_diag}

    return ImputationResult(dataframe=working, params=params, diagnostics=diagnostics)
