import numpy as np
import pandas as pd
import pytest

from app.pipeline.config import OutlierDetectionConfig
from app.pipeline.outliers import IS_OUTLIER_COLUMN, detect_outliers


def _make_dataset_with_outliers(
    n_normal=280, n_outliers=20, seed=0
) -> tuple[pd.DataFrame, np.ndarray]:
    # Generates correlated normal points and multivariate outliers with reversed correlation
    # but identical marginals.
    rng = np.random.default_rng(seed)
    a_normal = rng.normal(loc=0.0, scale=1.0, size=n_normal)
    normal = np.column_stack(
        [
            a_normal,
            a_normal + rng.normal(scale=0.3, size=n_normal),
            a_normal + rng.normal(scale=0.3, size=n_normal),
        ]
    )

    a_outlier = rng.normal(loc=0.0, scale=1.0, size=n_outliers)
    outliers = np.column_stack(
        [
            a_outlier,
            -2.0 * a_outlier + rng.normal(scale=0.15, size=n_outliers),
            -2.0 * a_outlier + rng.normal(scale=0.15, size=n_outliers),
        ]
    )

    data = np.vstack([normal, outliers])
    is_true_outlier = np.array([False] * n_normal + [True] * n_outliers)

    order = rng.permutation(len(data))
    data = data[order]
    is_true_outlier = is_true_outlier[order]

    df = pd.DataFrame(data, columns=["a", "b", "c"])
    return df, is_true_outlier


def _iqr_flags(series: pd.Series, k: float = 1.5) -> np.ndarray:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return ((series < lower) | (series > upper)).to_numpy()


def test_isolation_forest_flags_injected_outliers_without_dropping_rows():
    # Isolation Forest is weaker than LOF on correlation-reversal anomalies.
    df, is_true_outlier = _make_dataset_with_outliers()
    result = detect_outliers(
        df, OutlierDetectionConfig(method="isolation_forest", contamination=0.07), seed=42
    )
    assert len(result.dataframe) == len(df)

    flagged = result.dataframe[IS_OUTLIER_COLUMN].to_numpy()
    recall = flagged[is_true_outlier].mean()
    assert recall > 0.2


def test_lof_flags_injected_outliers_without_dropping_rows():
    df, is_true_outlier = _make_dataset_with_outliers()
    result = detect_outliers(df, OutlierDetectionConfig(method="lof", contamination=0.07), seed=42)
    assert len(result.dataframe) == len(df)

    flagged = result.dataframe[IS_OUTLIER_COLUMN].to_numpy()
    recall = flagged[is_true_outlier].mean()
    assert recall > 0.5


def test_isolation_forest_and_lof_outperform_single_column_iqr_on_multivariate_outliers():
    # Compare recall against single-column IQR baseline.
    df, is_true_outlier = _make_dataset_with_outliers(seed=1)

    iqr_flagged = _iqr_flags(df["a"]) | _iqr_flags(df["b"]) | _iqr_flags(df["c"])
    iqr_recall = iqr_flagged[is_true_outlier].mean()

    if_result = detect_outliers(
        df, OutlierDetectionConfig(method="isolation_forest", contamination=0.07), seed=42
    )
    if_recall = if_result.dataframe[IS_OUTLIER_COLUMN].to_numpy()[is_true_outlier].mean()

    assert if_recall >= iqr_recall


def test_none_method_flags_nothing():
    df, _ = _make_dataset_with_outliers()
    result = detect_outliers(df, OutlierDetectionConfig(method="none"))
    assert result.dataframe[IS_OUTLIER_COLUMN].sum() == 0
    assert (result.dataframe["_anomaly_score"] == 0.0).all()


def test_rejects_dataframe_with_unimputed_missing_values():
    df, _ = _make_dataset_with_outliers()
    df.loc[0, "a"] = np.nan
    with pytest.raises(ValueError):
        detect_outliers(df, OutlierDetectionConfig(method="isolation_forest"))


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        detect_outliers(pd.DataFrame())


def test_invalid_method_rejected_by_config():
    with pytest.raises(ValueError):
        OutlierDetectionConfig(method="not_a_method")


def test_invalid_contamination_rejected_by_config():
    with pytest.raises(ValueError):
        OutlierDetectionConfig(contamination=0.9)


def test_isolation_forest_deterministic_given_seed():
    df, _ = _make_dataset_with_outliers()
    r1 = detect_outliers(df, OutlierDetectionConfig(method="isolation_forest"), seed=7)
    r2 = detect_outliers(df, OutlierDetectionConfig(method="isolation_forest"), seed=7)
    pd.testing.assert_frame_equal(r1.dataframe, r2.dataframe)
