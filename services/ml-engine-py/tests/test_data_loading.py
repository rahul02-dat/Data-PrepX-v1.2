import numpy as np
import pandas as pd

from app.pipeline.estimation.dataset_loading import load_dataset_with_auto_preprocess


def _write_csv(tmp_path, df: pd.DataFrame, name: str = "data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_rows_with_missing_target_are_dropped_not_fabricated(tmp_path):
    df = pd.DataFrame(
        {
            "f1": np.arange(20, dtype=float),
            "target": [0, 1] * 9 + [np.nan, np.nan],
        }
    )
    path = _write_csv(tmp_path, df)

    dataset, report = load_dataset_with_auto_preprocess(path, target_column="target")

    assert report.rows_dropped_missing_target == 2
    assert dataset.X.shape[0] == 18
    assert dataset.y.shape[0] == 18
    # No NaN or fabricated placeholder should appear in y.
    assert not np.isnan(dataset.y).any()


def test_target_column_is_never_passed_to_imputer(tmp_path, monkeypatch):
    # Regression test for the original bug: impute() must never see the target column,
    # since including it as a MICE/KNN predictor leaks target information into "imputed"
    # feature values.
    df = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    path = _write_csv(tmp_path, df)

    from app.pipeline.imputation import impute as real_impute

    seen_columns: list[list[str]] = []

    def _spy_impute(df_arg, *args, **kwargs):
        seen_columns.append(list(df_arg.columns))
        return real_impute(df_arg, *args, **kwargs)

    monkeypatch.setattr("app.pipeline.imputation.impute", _spy_impute)

    load_dataset_with_auto_preprocess(path, target_column="target")

    assert len(seen_columns) == 1
    assert "target" not in seen_columns[0]
    assert seen_columns[0] == ["f1"]


def test_feature_imputation_and_encoding_report_is_accurate(tmp_path):
    df = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
            "target": [0, 1, 0, 1],
        }
    )
    path = _write_csv(tmp_path, df)

    dataset, report = load_dataset_with_auto_preprocess(path, target_column="target")

    assert report.rows_dropped_missing_target == 0
    assert report.feature_columns_imputed == ["f1"]
    assert report.feature_columns_one_hot_encoded == ["cat"]
    assert not np.isnan(dataset.X).any()


def test_no_missing_values_means_no_imputation_reported(tmp_path):
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    path = _write_csv(tmp_path, df)

    dataset, report = load_dataset_with_auto_preprocess(path, target_column="target")

    assert report.feature_columns_imputed == []
    assert report.rows_dropped_missing_target == 0