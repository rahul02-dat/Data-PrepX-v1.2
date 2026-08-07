"""
Run Phase 4 (Optuna HPO + stacking) against an arbitrary user-supplied dataset file.

Usage:
    cd services/ml-engine-py
    python3 -m app.pipeline.estimation.run_on_dataset --path /path/to/data.csv
    python3 -m app.pipeline.estimation.run_on_dataset \
        --path /path/to/data.csv --target-column price --task regression --n-trials 30

Target column and task type are auto-detected if not given (see dataset_loading.py).
Feature columns must already be numeric and fully imputed -- run Phase 2 gates and Phase 3
imputation first if that's not already true for your file; this script deliberately does not
silently encode or impute, since that would make the run's lineage/config_hash misleading.
"""

from __future__ import annotations

import argparse
import sys

from app.pipeline.estimation.dataset_loading import load_dataset, load_dataset_with_auto_preprocess
from app.pipeline.estimation.optuna_search import OptunaSearchConfig, default_baseline_score
from app.pipeline.estimation.stacking import run_stacking


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Path to the dataset file (CSV for now).")
    parser.add_argument(
        "--target-column",
        default=None,
        help="Name of the target column. Defaults to the last column in the file.",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default=None,
        help="Task type. Auto-detected from the target column if not given.",
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per family.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Outer cross-validation folds.")
    parser.add_argument(
        "--stacking-cv-folds",
        type=int,
        default=5,
        help="Internal stacking cv folds (lower for a faster, noisier run on a large file).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["xgboost", "lightgbm", "random_forest", "linear"],
        help="Model families to search over.",
    )
    parser.add_argument(
        "--auto-preprocess",
        action="store_true",
        help="Automatically impute missing values and one-hot encode categorical features.",
    )
    parser.add_argument(
        "--na-values",
        default=None,
        help="String to treat as missing value (e.g. '?').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.auto_preprocess:
        print(
            "WARNING: --auto-preprocess imputes/encodes features outside the lineage-tracked "
            "pipeline. This run is NOT reproducible/versioned the way a real DataPrepX run "
            "must be (CLAUDE.md section 2) -- use it for quick exploration only. The target "
            "column is never imputed; rows with a missing target are dropped, not fabricated."
        )
        na_vals = [args.na_values] if args.na_values else None
        dataset, report = load_dataset_with_auto_preprocess(
            args.path, target_column=args.target_column, task=args.task, na_values=na_vals
        )
        if report.rows_dropped_missing_target:
            print(f"Dropped {report.rows_dropped_missing_target} row(s) with a missing target.")
        if report.feature_columns_imputed:
            print(f"Imputed feature columns: {report.feature_columns_imputed}")
        if report.feature_columns_one_hot_encoded:
            print(f"One-hot encoded feature columns: {report.feature_columns_one_hot_encoded}")
    else:
        dataset = load_dataset(args.path, target_column=args.target_column, task=args.task)

    print(
        f"Loaded {args.path}: {dataset.X.shape[0]} rows, {dataset.X.shape[1]} features, "
        f"target={dataset.target_column!r}, task={dataset.task}"
    )

    config = OptunaSearchConfig(n_trials=args.n_trials, cv_folds=args.cv_folds, seed=args.seed)

    result = run_stacking(
        dataset.X,
        dataset.y,
        dataset.task,
        families=args.families,
        config=config,
        seed=args.seed,
        stacking_cv_folds=args.stacking_cv_folds,
    )

    baseline = default_baseline_score(dataset.X, dataset.y, "random_forest", dataset.task, config)

    print("\nPer-family tuned scores:")
    for family, study_result in result.base_results.items():
        print(f"  {family:15s} {study_result.best_score:.4f}  params={study_result.best_params}")

    print(f"\nStack CV score:              {result.stacking_cv_score:.4f}")
    print(
        f"Single best tuned family:    {result.single_best_family} "
        f"({result.single_best_cv_score:.4f})"
    )
    print(f"Default-hyperparameter RF baseline: {baseline:.4f}")

    if result.stacking_cv_score > baseline:
        print("\n-> Stack beats the untuned default baseline.")
    else:
        print("\n-> Stack did NOT beat the untuned default baseline on this run.")

    if result.stacking_cv_score >= result.single_best_cv_score:
        print("-> Stack matches or beats the single best tuned family.")
    else:
        print("-> Stack did NOT beat the single best tuned family on this run.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
