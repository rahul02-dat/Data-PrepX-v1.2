from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

TaskType = Literal["classification", "regression"]

# Numeric targets with more distinct values than this (relative to n_rows) are treated as
# regression; fewer, or non-numeric, is treated as classification. This mirrors the
# heuristic pandas/sklearn tooling commonly uses (e.g. a numeric column with only a
# handful of repeated values is almost always a class label, not a continuous target).
_MAX_CLASSIFICATION_UNIQUE_RATIO = 0.05
_MAX_CLASSIFICATION_UNIQUE_ABSOLUTE = 20

_READERS: dict[str, Any] = {
    ".csv": pd.read_csv,
    # Extension points for later formats (planner: "for now CSV, later various formats").
    # Uncomment as each is actually needed and tested -- don't wire in an untested reader.
    # ".parquet": pd.read_parquet,
    # ".xlsx": pd.read_excel,
    # ".json": pd.read_json,
}


@dataclass(frozen=True)
class LoadedDataset:
    X: np.ndarray
    y: np.ndarray
    task: TaskType
    target_column: str
    feature_columns: list[str]


# Infer classification vs. regression from the target column's dtype and cardinality
def infer_task_type(y: pd.Series) -> TaskType:
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    n_unique = y.nunique(dropna=True)
    n_rows = len(y)
    if n_unique <= _MAX_CLASSIFICATION_UNIQUE_ABSOLUTE:
        return "classification"
    if n_rows > 0 and (n_unique / n_rows) <= _MAX_CLASSIFICATION_UNIQUE_RATIO:
        return "classification"
    return "regression"


# Load a tabular dataset file, auto-detecting target column and task type unless given
def load_dataset(
    path: str | Path,
    *,
    target_column: str | None = None,
    task: TaskType | None = None,
) -> LoadedDataset:
    path = Path(path)
    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        supported = ", ".join(sorted(_READERS))
        raise ValueError(
            f"unsupported dataset format {suffix!r} for {path}; supported: {supported}. "
            "Add a reader to _READERS in dataset_loading.py once that format is actually needed."
        )

    df = reader(path)
    if df.empty:
        raise ValueError(f"{path} loaded an empty dataframe; cannot build a dataset from it")

    resolved_target = target_column or df.columns[-1]
    if resolved_target not in df.columns:
        raise ValueError(
            f"target_column {resolved_target!r} not found in {path}; "
            f"available columns: {list(df.columns)}"
        )

    y_series = df[resolved_target]
    feature_columns = [c for c in df.columns if c != resolved_target]
    X_df = df[feature_columns]

    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(X_df[c])]
    if non_numeric:
        raise ValueError(
            f"feature columns {non_numeric} in {path} are non-numeric; encode categorical "
            "features (e.g. one-hot) before passing to the estimation module -- this loader "
            "intentionally does not silently encode, since the encoding choice affects "
            "downstream lineage hashes."
        )
    if X_df.isna().any().any():
        raise ValueError(
            f"{path} has missing values in feature columns; run Phase 3 imputation "
            "(app.pipeline.imputation.impute) before passing to the estimation module."
        )

    resolved_task = task or infer_task_type(y_series)

    if resolved_task == "classification" and not pd.api.types.is_numeric_dtype(y_series):
        y_array = y_series.astype("category").cat.codes.to_numpy()
    else:
        y_array = y_series.to_numpy(dtype=float)

    return LoadedDataset(
        X=X_df.to_numpy(dtype=float),
        y=y_array,
        task=resolved_task,
        target_column=resolved_target,
        feature_columns=feature_columns,
    )


@dataclass(frozen=True)
class AutoPreprocessReport:
    """What load_dataset_with_auto_preprocess actually did to the data. Printed by the CLI so
    the transformation isn't silent even though (unlike the Phase 2/3 pipeline) it isn't
    recorded to lineage -- this path is for quick exploration, not a reproducible run."""

    rows_dropped_missing_target: int
    feature_columns_imputed: list[str]
    feature_columns_one_hot_encoded: list[str]
    imputation_diagnostics: dict[str, Any]


# Load a dataset file, auto-imputing and one-hot-encoding FEATURE columns only. The target
# column is never imputed or used as an imputation predictor for features: rows with a missing
# target are dropped (imputing a label is fabricating ground truth; including the target as a
# MICE/KNN predictor for features leaks it into "imputed" values a real held-out row wouldn't
# have). This is a convenience path for quick exploration -- it does not go through
# lineage.py, so it is not reproducible/versioned the way a real pipeline run must be
# (CLAUDE.md §2). Use the Phase 2/3 pipeline directly (gates -> impute()) if you need that.
def load_dataset_with_auto_preprocess(
    path: str | Path,
    *,
    target_column: str | None = None,
    task: TaskType | None = None,
    na_values: list[str] | None = None,
) -> tuple[LoadedDataset, AutoPreprocessReport]:
    from app.pipeline.config import ImputationConfig
    from app.pipeline.imputation import impute

    path = Path(path)
    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        supported = ", ".join(sorted(_READERS))
        raise ValueError(
            f"unsupported dataset format {suffix!r} for {path}; supported: {supported}"
        )

    df = reader(path, na_values=na_values) if na_values else reader(path)
    if df.empty:
        raise ValueError(f"{path} loaded an empty dataframe; cannot build a dataset from it")

    resolved_target = target_column or df.columns[-1]
    if resolved_target not in df.columns:
        raise ValueError(
            f"target_column {resolved_target!r} not found in {path}; "
            f"available columns: {list(df.columns)}"
        )

    # Drop rows with a missing target BEFORE touching features at all: never fabricate a
    # label, and never let a to-be-dropped row's features influence imputation of kept rows.
    n_before = len(df)
    df = df.dropna(subset=[resolved_target]).reset_index(drop=True)
    rows_dropped = n_before - len(df)

    y_series = df[resolved_target]
    feature_columns = [c for c in df.columns if c != resolved_target]
    X_df = df[feature_columns].copy()

    # Impute FEATURES ONLY -- the target is never passed in, so it cannot leak into imputed
    # feature values, and cannot itself be imputed.
    imputed_cols: list[str] = []
    imputation_diagnostics: dict[str, Any] = {}
    if X_df.isna().any().any():
        impute_result = impute(X_df, ImputationConfig())
        X_df = impute_result.dataframe
        imputation_diagnostics = impute_result.diagnostics
        imputed_cols = [c for c in feature_columns if df[c].isna().any()]

    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(X_df[c])]
    encoded_cols: list[str] = []
    if non_numeric:
        X_df = pd.get_dummies(X_df, columns=non_numeric, drop_first=True)
        encoded_cols = non_numeric
    feature_columns = list(X_df.columns)

    resolved_task = task or infer_task_type(y_series)
    if resolved_task == "classification" and not pd.api.types.is_numeric_dtype(y_series):
        y_array = y_series.astype("category").cat.codes.to_numpy()
    else:
        y_array = y_series.to_numpy(dtype=float)

    dataset = LoadedDataset(
        X=X_df.to_numpy(dtype=float),
        y=y_array,
        task=resolved_task,
        target_column=resolved_target,
        feature_columns=feature_columns,
    )
    report = AutoPreprocessReport(
        rows_dropped_missing_target=rows_dropped,
        feature_columns_imputed=imputed_cols,
        feature_columns_one_hot_encoded=encoded_cols,
        imputation_diagnostics=imputation_diagnostics,
    )
    return dataset, report