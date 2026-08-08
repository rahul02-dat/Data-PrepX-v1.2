import numpy as np
import pandas as pd

from app.pipeline.rl_optimizer.meta_features import (
    class_imbalance_ratio,
    compute_meta_features,
    drift_score,
)


def test_missingness_rate_reflects_actual_missing_fraction():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    features = compute_meta_features(df)
    assert features.missingness_rate == 2 / 8


def test_missingness_rate_zero_for_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    features = compute_meta_features(df)
    assert features.missingness_rate == 0.0


def test_mean_abs_skew_zero_for_no_numeric_columns():
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    features = compute_meta_features(df)
    assert features.mean_abs_skew == 0.0


def test_mean_abs_skew_positive_for_skewed_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.exponential(size=500)})  # heavily right-skewed
    features = compute_meta_features(df)
    assert features.mean_abs_skew > 0.5


def test_mean_cardinality_ratio_zero_for_no_categorical_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    features = compute_meta_features(df)
    assert features.mean_cardinality_ratio == 0.0


def test_mean_cardinality_ratio_high_for_near_unique_categorical():
    df = pd.DataFrame({"id": [f"row_{i}" for i in range(100)]})
    features = compute_meta_features(df)
    assert features.mean_cardinality_ratio > 0.9


def test_class_imbalance_ratio_one_for_balanced_binary_target():
    y = np.array([0, 1] * 50)
    assert class_imbalance_ratio(y) == 1.0


def test_class_imbalance_ratio_high_for_imbalanced_target():
    y = np.array([0] * 95 + [1] * 5)
    assert class_imbalance_ratio(y) == 19.0


def test_class_imbalance_ratio_one_when_target_is_none():
    assert class_imbalance_ratio(None) == 1.0


def test_class_imbalance_ratio_one_for_single_class():
    y = np.array([1, 1, 1, 1])
    assert class_imbalance_ratio(y) == 1.0


def test_drift_score_zero_without_reference():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert drift_score(df, None) == 0.0


def test_drift_score_positive_for_shifted_distribution():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"a": rng.normal(loc=0, size=500)})
    current = pd.DataFrame({"a": rng.normal(loc=5, size=500)})
    score = drift_score(current, reference)
    assert score > 0.0


def test_drift_score_near_zero_for_identical_distributions():
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({"a": rng.normal(size=1000)})
    current = pd.DataFrame({"a": rng.normal(size=1000)})
    score = drift_score(current, reference)
    assert score < 0.25  # below the Phase 2 DriftGate's default PSI threshold


def test_compute_meta_features_returns_all_fields_as_finite_floats():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=100),
            "cat": rng.choice(["x", "y", "z"], size=100),
        }
    )
    y = rng.integers(0, 2, size=100)
    features = compute_meta_features(df, target=y)
    arr = features.as_array()
    assert arr.shape == (5,)
    assert np.isfinite(arr).all()
