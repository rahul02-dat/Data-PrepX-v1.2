import pytest
from sklearn.datasets import make_classification, make_regression

from app.pipeline.estimation.optuna_search import OptunaSearchConfig, run_all_families
from app.pipeline.estimation.stacking import build_stack, run_stacking

FAST_CONFIG = OptunaSearchConfig(n_trials=5, cv_folds=3, seed=42, n_startup_trials=2)
FAST_FAMILIES = ["random_forest", "linear"]


def _classification_data():
    return make_classification(n_samples=250, n_features=12, n_informative=8, random_state=1)


def _regression_data():
    return make_regression(n_samples=250, n_features=12, noise=8.0, random_state=1)


def test_build_stack_raises_on_empty_base_results():
    with pytest.raises(ValueError):
        build_stack("classification", {}, seed=0)


def test_run_stacking_classification_end_to_end():
    X, y = _classification_data()
    result = run_stacking(X, y, "classification", FAST_FAMILIES, FAST_CONFIG, seed=42)

    assert result.task == "classification"
    assert set(result.base_results.keys()) == set(FAST_FAMILIES)
    assert result.single_best_family in FAST_FAMILIES
    assert -1.0 <= result.stacking_cv_score <= 1.0
    # Fitted stack should be usable for prediction.
    preds = result.fitted_stack.predict(X)
    assert preds.shape == y.shape


def test_run_stacking_regression_end_to_end():
    X, y = _regression_data()
    result = run_stacking(X, y, "regression", FAST_FAMILIES, FAST_CONFIG, seed=42)

    assert result.task == "regression"
    assert result.stacking_cv_score <= 0  # neg RMSE
    preds = result.fitted_stack.predict(X)
    assert preds.shape == y.shape


def test_run_stacking_is_deterministic_given_seed():
    X, y = _classification_data()
    r1 = run_stacking(X, y, "classification", FAST_FAMILIES, FAST_CONFIG, seed=42)
    r2 = run_stacking(X, y, "classification", FAST_FAMILIES, FAST_CONFIG, seed=42)
    assert r1.stacking_cv_score == r2.stacking_cv_score
    assert r1.single_best_family == r2.single_best_family


def test_single_best_family_matches_max_of_base_results():
    X, y = _classification_data()
    base_results = run_all_families(X, y, "classification", FAST_FAMILIES, FAST_CONFIG)
    expected_best = max(base_results, key=lambda f: base_results[f].best_score)

    result = run_stacking(X, y, "classification", FAST_FAMILIES, FAST_CONFIG, seed=42)
    assert result.single_best_family == expected_best
