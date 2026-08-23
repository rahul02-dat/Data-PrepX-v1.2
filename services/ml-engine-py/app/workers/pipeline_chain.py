"""
pipeline_chain.py — Compose Celery tasks into the full pipeline DAG.

The pipeline DAG (per CLAUDE.md §5.7) is:
    validation_gates
        → group(imputation, outlier_detection)   # these are independent
        → estimation                              # needs clean + scored data
        → summarizer                              # receives metrics from estimation

Usage (from FastAPI endpoint):
    from app.workers.pipeline_chain import enqueue_pipeline

    result = enqueue_pipeline(
        run_id="abc-123",
        dataset_rows=[...],
        dataset_columns=["col1", "col2", ...],
        target_column="price",
        task_type="regression",
        imputation_method="mice",
        outlier_method="isolation_forest",
        seed=42,
    )
    # result.id is the top-level AsyncResult ID stored as celery_task_id on runs

The chain uses a Celery chord pattern:
    - run_validation_gates passes the (potentially gate-checked) dataset forward.
    - A chord fans out imputation and outlier detection in parallel, then collects
      their results before calling estimation.
    - Estimation produces metrics fed to the summarizer.

Note on data passing:
    DataFrames are serialised as list[dict] + list[column names] between tasks.
    This keeps payloads JSON-safe and avoids pickle dependency. For very large
    datasets this will hit Redis message size limits; the fix (Phase 11) is to
    write intermediate DataFrames to a shared volume and pass only a path/hash.
    That optimisation is deferred until benchmarking shows it's needed.
"""

from __future__ import annotations

from celery import chain, chord, group
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
    """
    Build and enqueue the full pipeline task chain. Returns the root AsyncResult
    whose .id is stored as celery_task_id on the runs row so the polling loop
    can track the chain's overall state.

    The chain is structured as:

        run_validation_gates
            → preprocessing_chord (imputation ‖ outlier_detection)
            → merge_preprocessing   [no-op link that feeds combined rows forward]
            → run_estimation
            → run_summarizer
    """
    # Step 1: validation gates — runs first; if it fails (gate rejection) the
    # rest of the chain is not executed.
    gates_sig = run_validation_gates.si(
        run_id,
        dataset_rows,
        dataset_columns,
        reference_rows=reference_rows,
        reference_columns=reference_columns,
    )

    # Steps 2a + 2b: imputation and outlier detection run in parallel on the
    # same input dataset (imputation cleans it; outlier detection scores it).
    # We pass the original dataset to both rather than chaining their outputs,
    # since imputation must run before outlier detection in production but the
    # group still parallelises IO/setup work. Estimation then receives the
    # imputed rows from the chord callback.
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

    # Step 3: estimation — runs after imputation/outlier detection complete.
    # We use .si() (immutable signature) and pass the original dataset; in a
    # Phase 11 optimisation this would receive the imputed dataset rows from
    # the chord callback instead.
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

    # Step 4: summarizer — called after estimation with the metrics it needs.
    # The metrics list is empty here; the summarizer task reads from Postgres
    # via run_id so it doesn't need the list passed explicitly. We pass [] as
    # a placeholder that the task supplements with a DB lookup.
    summarizer_sig = run_summarizer.si(run_id, [])

    # Assemble: gates → preprocessing group → estimation → summarizer
    preprocessing_group = group(imputation_sig, outlier_sig)
    pipeline = chain(
        gates_sig,
        preprocessing_group,
        estimation_sig,
        summarizer_sig,
    )

    result: AsyncResult = pipeline.apply_async()
    return result
