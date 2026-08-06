from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_score

from app.pipeline.estimation.optuna_search import (
    ModelFamily,
    OptunaSearchConfig,
    StudyResult,
    TaskType,
    cv_splitter,
    run_all_families,
    scoring_for_task,
)


@dataclass(frozen=True)
class StackingResult:
    task: TaskType
    base_results: dict[str, StudyResult]
    stacking_cv_score: float
    single_best_family: str
    single_best_cv_score: float
    fitted_stack: BaseEstimator
    meta_learner_params: dict[str, Any] = field(default_factory=dict)


# Build an (unfitted) stack combining every base_results estimator as a base learner
def build_stack(
    task: TaskType,
    base_results: dict[str, StudyResult],
    seed: int,
    stacking_cv_folds: int = 5,
) -> BaseEstimator:
    if not base_results:
        raise ValueError("base_results must contain at least one fitted model family")

    estimators = [(family, clone(result.estimator)) for family, result in base_results.items()]

    if task == "classification":
        meta = LogisticRegression(max_iter=2000, random_state=seed)
        return StackingClassifier(
            estimators=estimators, final_estimator=meta, cv=stacking_cv_folds, n_jobs=1
        )

    meta = RidgeCV()
    return StackingRegressor(
        estimators=estimators, final_estimator=meta, cv=stacking_cv_folds, n_jobs=1
    )


# Run Optuna HPO per family, then cross-validate a stack of the tuned models against the
# single best-performing tuned family (Phase 4 acceptance criterion: stack beats best single model
# and beats the fixed-default baseline, see estimation benchmark).
def run_stacking(
    X: np.ndarray,
    y: np.ndarray,
    task: TaskType,
    families: tuple[ModelFamily, ...] | list[ModelFamily] | None = None,
    config: OptunaSearchConfig | None = None,
    seed: int = 42,
    stacking_cv_folds: int = 5,
) -> StackingResult:
    config = config or OptunaSearchConfig(seed=seed)
    base_results = run_all_families(X, y, task, families, config)

    stack = build_stack(task, base_results, seed, stacking_cv_folds=stacking_cv_folds)
    splitter = cv_splitter(task, config.cv_folds, seed)
    scoring = scoring_for_task(task)
    stack_scores = cross_val_score(stack, X, y, cv=splitter, scoring=scoring, n_jobs=1)
    stacking_cv_score = float(np.mean(stack_scores))

    single_best_family = max(base_results, key=lambda f: base_results[f].best_score)
    single_best_cv_score = base_results[single_best_family].best_score

    fitted_stack = clone(stack)
    fitted_stack.fit(X, y)

    return StackingResult(
        task=task,
        base_results=base_results,
        stacking_cv_score=stacking_cv_score,
        single_best_family=single_best_family,
        single_best_cv_score=single_best_cv_score,
        fitted_stack=fitted_stack,
    )
