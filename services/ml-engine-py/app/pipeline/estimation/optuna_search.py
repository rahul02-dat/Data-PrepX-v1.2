from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

TaskType = Literal["classification", "regression"]
ModelFamily = Literal["xgboost", "lightgbm", "random_forest", "linear"]

_ALL_FAMILIES: tuple[ModelFamily, ...] = ("xgboost", "lightgbm", "random_forest", "linear")

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class OptunaSearchConfig:
    n_trials: int = 30
    cv_folds: int = 5
    seed: int = 42
    n_startup_trials: int = 5
    n_warmup_steps: int = 5


@dataclass(frozen=True)
class TrialRecord:
    """One Optuna trial, shaped to map directly onto the `hyperparameters` lineage table
    (CLAUDE.md §6: model_family, trial_number, params_json, score)."""

    model_family: str
    trial_number: int
    params: dict[str, Any]
    score: float


@dataclass(frozen=True)
class StudyResult:
    model_family: str
    task: TaskType
    best_params: dict[str, Any]
    best_score: float
    trials: list[TrialRecord] = field(default_factory=list)
    estimator: BaseEstimator | None = None


# Cross-validation splitter appropriate to the task type
def cv_splitter(task: TaskType, n_splits: int, seed: int):
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


# Scoring metric appropriate to the task type. Both directions are "higher is better":
# sklearn negates RMSE so neg_root_mean_squared_error also maximizes.
def scoring_for_task(task: TaskType) -> str:
    return "accuracy" if task == "classification" else "neg_root_mean_squared_error"


# Suggest a hyperparameter set for the given model family and task
def suggest_params(trial: optuna.Trial, family: ModelFamily, task: TaskType) -> dict[str, Any]:
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if family == "linear":
        if task == "classification":
            return {
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    raise ValueError(f"unknown model family: {family!r}")


# Construct an estimator instance for the given model family, task, and hyperparameters
def build_estimator(
    family: ModelFamily, task: TaskType, params: dict[str, Any], seed: int
) -> BaseEstimator:
    if family == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(random_state=seed, n_jobs=1, **params)
    if family == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        return cls(random_state=seed, n_jobs=1, verbosity=0, **params)
    if family == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        return cls(random_state=seed, n_jobs=1, verbose=-1, **params)
    if family == "linear":
        if task == "classification":
            return LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=2000, random_state=seed, **params
            )
        return ElasticNet(random_state=seed, max_iter=5000, **params)
    raise ValueError(f"unknown model family: {family!r}")


# Default (untuned) estimator for a family, used as the Phase 4 acceptance baseline
def default_estimator(family: ModelFamily, task: TaskType, seed: int) -> BaseEstimator:
    if family == "linear":
        params: dict[str, Any] = {}
    else:
        params = {}
    return build_estimator(family, task, params, seed)


# Run a single-model-family Optuna study: TPE sampler, median pruning, cross-validated objective
def run_optuna_study(
    X: np.ndarray,
    y: np.ndarray,
    family: ModelFamily,
    task: TaskType,
    config: OptunaSearchConfig | None = None,
) -> StudyResult:
    config = config or OptunaSearchConfig()
    splitter = cv_splitter(task, config.cv_folds, config.seed)
    scoring = scoring_for_task(task)
    trials_log: list[TrialRecord] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, family, task)
        estimator = build_estimator(family, task, params, config.seed)
        scores = cross_val_score(estimator, X, y, cv=splitter, scoring=scoring, n_jobs=1)
        mean_score = float(np.mean(scores))
        trials_log.append(TrialRecord(family, trial.number, dict(params), mean_score))
        return mean_score

    sampler = TPESampler(seed=config.seed, n_startup_trials=config.n_startup_trials)
    pruner = MedianPruner(n_warmup_steps=config.n_warmup_steps)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=config.n_trials)

    best_estimator = build_estimator(family, task, study.best_params, config.seed)
    best_estimator.fit(X, y)

    return StudyResult(
        model_family=family,
        task=task,
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        trials=trials_log,
        estimator=best_estimator,
    )


# Run one Optuna study per requested model family
def run_all_families(
    X: np.ndarray,
    y: np.ndarray,
    task: TaskType,
    families: tuple[ModelFamily, ...] | list[ModelFamily] | None = None,
    config: OptunaSearchConfig | None = None,
) -> dict[str, StudyResult]:
    families = tuple(families) if families else _ALL_FAMILIES
    return {family: run_optuna_study(X, y, family, task, config) for family in families}


# Cross-validated score of a family's untuned, default-hyperparameter estimator
def default_baseline_score(
    X: np.ndarray, y: np.ndarray, family: ModelFamily, task: TaskType, config: OptunaSearchConfig
) -> float:
    splitter = cv_splitter(task, config.cv_folds, config.seed)
    scoring = scoring_for_task(task)
    estimator = default_estimator(family, task, config.seed)
    scores = cross_val_score(estimator, X, y, cv=splitter, scoring=scoring, n_jobs=1)
    return float(np.mean(scores))
