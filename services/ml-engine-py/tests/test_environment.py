import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from app.pipeline.rl_optimizer.environment import (
    Action,
    PreprocessingEnv,
    apply_action,
    build_action_space,
)


def _fast_classification_reward(X, y, task):
    clf = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)
    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
    return float(scores.mean())


def _fast_regression_reward(X, y, task):
    reg = RandomForestRegressor(n_estimators=20, random_state=0, n_jobs=1)
    scores = cross_val_score(reg, X, y, cv=3, scoring="neg_root_mean_squared_error")
    return float(scores.mean())


def _classification_data(n=150, missing_rate=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})
    y = (X["f1"] + X["f2"] * 0.5 > 0).astype(int).to_numpy()
    mask = rng.random(n) < missing_rate
    X.loc[mask, "f1"] = np.nan
    return X, y


def test_build_action_space_has_expected_size():
    actions = build_action_space()
    # 2 imputers * (2 outlier methods * 4 bins + 1 "none") = 2 * 9 = 18
    assert len(actions) == 18


def test_build_action_space_has_no_duplicate_actions():
    actions = build_action_space()
    assert len(set(actions)) == len(actions)


def test_action_as_dict_has_null_contamination_for_none_method():
    action = Action("mice", "none", 0)
    assert action.as_dict()["contamination"] is None


def test_action_as_dict_has_contamination_for_real_outlier_method():
    action = Action("knn", "lof", 2)
    assert action.as_dict()["contamination"] == pytest.approx(0.10)


def test_action_compute_cost_orders_methods_sensibly():
    mice_none = Action("mice", "none", 0)
    knn_none = Action("knn", "none", 0)
    mice_lof = Action("mice", "lof", 0)
    assert knn_none.compute_cost() < mice_none.compute_cost()
    assert mice_none.compute_cost() < mice_lof.compute_cost()


def test_apply_action_none_outlier_keeps_all_rows():
    X, y = _classification_data()
    action = Action("mice", "none", 0)
    X_out, y_out = apply_action(X, y, action, seed=0)
    assert X_out.shape[0] == len(X)
    assert y_out.shape[0] == len(y)
    assert not np.isnan(X_out).any()


def test_apply_action_with_outlier_method_can_drop_rows():
    X, y = _classification_data(n=300, seed=1)
    action = Action("mice", "isolation_forest", 3)  # contamination=0.20
    X_out, y_out = apply_action(X, y, action, seed=0)
    assert X_out.shape[0] <= len(X)
    assert X_out.shape[0] == y_out.shape[0]


def test_apply_action_falls_back_to_keep_all_when_too_many_flagged(monkeypatch):
    import app.pipeline.rl_optimizer.environment as env_module

    X, y = _classification_data(n=50, seed=2)

    class _FlagEverything:
        def __init__(self, *a, **k):
            pass

        dataframe = None

    def _fake_detect_outliers(df, config, seed=None):
        out = df.copy()
        out["_is_outlier"] = True  # flag every row
        from app.pipeline.outliers import OutlierResult

        return OutlierResult(dataframe=out, params={}, diagnostics={})

    monkeypatch.setattr(env_module, "detect_outliers", _fake_detect_outliers)

    action = Action("mice", "isolation_forest", 0)
    X_out, y_out = apply_action(X, y, action, seed=0)
    assert X_out.shape[0] == len(X)  # fell back to keeping everything


def test_env_reset_returns_meta_features_and_caches_baseline():
    X, y = _classification_data()
    env = PreprocessingEnv(_fast_classification_reward, seed=0)
    features = env.reset(X, y, "classification")
    assert features.missingness_rate > 0
    assert env._baseline_score is not None


def test_env_step_before_reset_raises():
    env = PreprocessingEnv(_fast_classification_reward, seed=0)
    with pytest.raises(RuntimeError):
        env.step(Action("mice", "none", 0))


def test_env_step_returns_done_true_single_step_episode():
    X, y = _classification_data()
    env = PreprocessingEnv(_fast_classification_reward, seed=0)
    env.reset(X, y, "classification")
    result = env.step(Action("mice", "none", 0))
    assert result.done is True
    assert "score" in result.info
    assert "baseline_score" in result.info


def test_env_reward_reflects_delta_minus_cost_penalty():
    X, y = _classification_data()
    env = PreprocessingEnv(_fast_classification_reward, seed=0)
    env.reset(X, y, "classification")
    action = Action("mice", "none", 0)
    result = env.step(action)
    expected = (result.info["score"] - result.info["baseline_score"]) - action.compute_cost()
    assert result.reward == pytest.approx(expected)


def test_env_works_for_regression_task():
    rng = np.random.default_rng(5)
    n = 150
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = (X["f1"] * 2 + X["f2"] + rng.normal(scale=0.5, size=n)).to_numpy()
    X.loc[rng.random(n) < 0.1, "f1"] = np.nan

    env = PreprocessingEnv(_fast_regression_reward, seed=0)
    env.reset(X, y, "regression")
    result = env.step(Action("knn", "none", 0))
    assert result.done is True
    assert np.isfinite(result.reward)
