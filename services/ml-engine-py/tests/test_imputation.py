import numpy as np
import pandas as pd
import pytest

from app.pipeline.config import ImputationConfig
from app.pipeline.imputation import impute


def _make_correlated_dataset(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 2.0 * x1 + rng.normal(scale=0.3, size=n)
    x3 = -1.5 * x1 + 0.5 * x2 + rng.normal(scale=0.3, size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat})


def _inject_mcar(df: pd.DataFrame, cols: list[str], rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    truth = df.copy()
    mask = {}
    for col in cols:
        m = rng.random(len(out)) < rate
        mask[col] = m
        out.loc[m, col] = np.nan
    return out, truth, mask


def test_mice_fills_all_numeric_missing_values():
    df = _make_correlated_dataset()
    missing, _, _ = _inject_mcar(df, ["x1", "x2", "x3"], rate=0.2, seed=1)
    result = impute(missing, ImputationConfig(method="mice"), seed=42)
    assert result.dataframe[["x1", "x2", "x3"]].isna().sum().sum() == 0
    assert result.diagnostics["numeric"]["missing_values_before"] > 0
    assert result.diagnostics["numeric"]["missing_values_after"] == 0


def test_knn_fills_all_numeric_missing_values():
    df = _make_correlated_dataset()
    missing, _, _ = _inject_mcar(df, ["x1", "x2", "x3"], rate=0.2, seed=2)
    result = impute(missing, ImputationConfig(method="knn"), seed=42)
    assert result.dataframe[["x1", "x2", "x3"]].isna().sum().sum() == 0


def test_categorical_columns_imputed_with_most_frequent():
    df = _make_correlated_dataset()
    missing = df.copy()
    rng = np.random.default_rng(3)
    mask = rng.random(len(missing)) < 0.15
    missing.loc[mask, "cat"] = np.nan

    result = impute(missing, ImputationConfig(), seed=42)
    assert result.dataframe["cat"].isna().sum() == 0
    assert result.diagnostics["categorical"]["categorical_columns_imputed"] == 1


def test_mice_beats_mean_imputation_on_correlated_data():
    # MICE models column correlation, reconstructing values more accurately than mean imputation.
    df = _make_correlated_dataset(n=500, seed=10)
    missing, truth, mask = _inject_mcar(df, ["x2"], rate=0.25, seed=11)

    mice_result = impute(missing, ImputationConfig(method="mice"), seed=42)
    mice_error = np.sqrt(
        np.mean((mice_result.dataframe.loc[mask["x2"], "x2"] - truth.loc[mask["x2"], "x2"]) ** 2)
    )

    mean_filled = missing.copy()
    mean_filled["x2"] = mean_filled["x2"].fillna(mean_filled["x2"].mean())
    mean_error = np.sqrt(
        np.mean((mean_filled.loc[mask["x2"], "x2"] - truth.loc[mask["x2"], "x2"]) ** 2)
    )

    assert mice_error < mean_error


def test_mice_diagnostics_report_convergence_info():
    df = _make_correlated_dataset()
    missing, _, _ = _inject_mcar(df, ["x1", "x2"], rate=0.2, seed=4)
    result = impute(missing, ImputationConfig(method="mice", mice_max_iter=15), seed=42)
    diag = result.diagnostics["numeric"]
    assert "n_iter" in diag
    assert "converged" in diag
    assert diag["n_iter"] <= 15


def test_impute_is_deterministic_given_seed():
    df = _make_correlated_dataset()
    missing, _, _ = _inject_mcar(df, ["x1", "x2", "x3"], rate=0.2, seed=5)
    r1 = impute(missing, ImputationConfig(method="mice"), seed=7)
    r2 = impute(missing, ImputationConfig(method="mice"), seed=7)
    pd.testing.assert_frame_equal(r1.dataframe, r2.dataframe)


def test_impute_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        impute(pd.DataFrame())


def test_invalid_method_rejected_by_config():
    with pytest.raises(ValueError):
        ImputationConfig(method="not_a_method")
