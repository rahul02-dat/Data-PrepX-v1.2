from __future__ import annotations

from celery import chain, group
from celery.result import AsyncResult

from app.workers.tasks import (
    run_estimation,
    run_imputation,
    run_outlier_detection,
    run_summarizer,
    run_validation_gates,
)


def enqueue_pipeline(
    *,
    run_id: str,
    dataset_rows: list[dict],
    dataset_columns: list[str],
    target_column: str,
    task_type: str,
    imputation_method: str = "mice",
    outlier_method: str = "isolation_forest",
    seed: int = 42,
    n_trials: int = 30,
    cv_folds: int = 5,
    stacking_cv_folds: int = 5,
    reference_rows: list[dict] | None = None,
    reference_columns: list[str] | None = None,
) -> AsyncResult:
    """Build and enqueue the pipeline Celery DAG (gates -> preprocessing -> estimation)."""
    gates_sig = run_validation_gates.si(
        run_id,
        dataset_rows,
        dataset_columns,
        reference_rows=reference_rows,
        reference_columns=reference_columns,
    )

    imputation_sig = run_imputation.si(
        run_id,
        dataset_rows,
        dataset_columns,
        method=imputation_method,
        seed=seed,
    )
    outlier_sig = run_outlier_detection.si(
        run_id,
        dataset_rows,
        dataset_columns,
        method=outlier_method,
        seed=seed,
    )

    estimation_sig = run_estimation.si(
        run_id,
        dataset_rows,
        dataset_columns,
        target_column=target_column,
        task_type=task_type,
        seed=seed,
        n_trials=n_trials,
        cv_folds=cv_folds,
        stacking_cv_folds=stacking_cv_folds,
    )

    summarizer_sig = run_summarizer.si(run_id, [])

    preprocessing_group = group(imputation_sig, outlier_sig)
    pipeline = chain(
        gates_sig,
        preprocessing_group,
        estimation_sig,
        summarizer_sig,
    )

    return pipeline.apply_async()
