import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from app.pipeline.estimation.optuna_search import (
    OptunaSearchConfig,
    build_estimator,
    default_baseline_score,
    run_all_families,
    run_optuna_study,
    suggest_params,
)


def _classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=6, random_state=0)
    return X, y


def _regression_data():
    X, y = make_regression(n_samples=200, n_features=10, noise=5.0, random_state=0)  # type: ignore
    return X, y


FAST_CONFIG = OptunaSearchConfig(n_trials=5, cv_folds=3, seed=42, n_startup_trials=2)


@pytest.mark.parametrize("family", ["random_forest", "xgboost", "lightgbm", "linear"])
def test_run_optuna_study_classification_all_families(family):
    X, y = _classification_data()
    result = run_optuna_study(X, y, family, "classification", FAST_CONFIG)
    assert result.model_family == family
    assert result.task == "classification"
    assert len(result.trials) == FAST_CONFIG.n_trials
    assert result.estimator is not None
    # Fitted estimator should score above chance on the data it was fit on.
    assert result.estimator.score(X, y) > 0.5


@pytest.mark.parametrize("family", ["random_forest", "xgboost", "lightgbm", "linear"])
def test_run_optuna_study_regression_all_families(family):
    X, y = _regression_data()
    result = run_optuna_study(X, y, family, "regression", FAST_CONFIG)
    assert result.model_family == family
    assert result.task == "regression"
    assert len(result.trials) == FAST_CONFIG.n_trials
    # neg_root_mean_squared_error is always <= 0; best_score should not be absurdly negative.
    assert result.best_score <= 0


def test_run_optuna_study_is_deterministic_given_seed():
    X, y = _classification_data()
    r1 = run_optuna_study(X, y, "random_forest", "classification", FAST_CONFIG)
    r2 = run_optuna_study(X, y, "random_forest", "classification", FAST_CONFIG)
    assert r1.best_params == r2.best_params
    assert r1.best_score == r2.best_score


def test_trial_log_length_matches_n_trials():
    X, y = _classification_data()
    config = OptunaSearchConfig(n_trials=8, cv_folds=3, seed=1, n_startup_trials=2)
    result = run_optuna_study(X, y, "linear", "classification", config)
    assert len(result.trials) == 8
    trial_numbers = [t.trial_number for t in result.trials]
    assert trial_numbers == sorted(trial_numbers)


def test_run_all_families_returns_one_result_per_family():
    X, y = _classification_data()
    families = ["random_forest", "linear"]
    results = run_all_families(X, y, "classification", families, FAST_CONFIG)
    assert set(results.keys()) == set(families)
    for family, result in results.items():
        assert result.model_family == family


def test_suggest_params_raises_on_unknown_family():
    import optuna

    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError):
        suggest_params(trial, "not_a_family", "classification")


def test_build_estimator_raises_on_unknown_family():
    with pytest.raises(ValueError):
        build_estimator("not_a_family", "classification", {}, seed=0)


def test_default_baseline_score_is_finite():
    X, y = _classification_data()
    score = default_baseline_score(X, y, "random_forest", "classification", FAST_CONFIG)
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_optuna_tuned_score_at_least_as_good_as_default_on_easy_task():
    # On an easy, well-separated classification task, HPO should not be worse than defaults
    # by more than a small margin (it may occasionally match, not always strictly beat, at
    # only 5 trials -- the strict "beats defaults" claim is validated properly in the research
    # benchmark with a full trial budget across multiple seeds).
    X, y = make_classification(
        n_samples=300, n_features=15, n_informative=10, class_sep=1.5, random_state=3
    )
    config = OptunaSearchConfig(n_trials=15, cv_folds=3, seed=3, n_startup_trials=3)
    tuned = run_optuna_study(X, y, "random_forest", "classification", config)
    default = default_baseline_score(X, y, "random_forest", "classification", config)
    assert tuned.best_score >= default - 0.05
