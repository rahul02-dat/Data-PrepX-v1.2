from __future__ import annotations

from unittest.mock import MagicMock

from app.pipeline.lineage import LineageRecorder


def _mock_conn_with_cursor(fetchone_return=None):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = fetchone_return
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


def test_record_hyperparameter_trial_inserts_expected_row():
    conn, cursor = _mock_conn_with_cursor(fetchone_return={"id": "trial-uuid-1"})
    recorder = LineageRecorder(conn)

    trial_id = recorder.record_hyperparameter_trial(
        "run-1",
        model_family="xgboost",
        trial_number=3,
        params={"max_depth": 5},
        score=0.87,
    )

    assert trial_id == "trial-uuid-1"
    insert_sql, args = cursor.execute.call_args_list[0].args
    assert "INSERT INTO hyperparameters" in insert_sql
    assert args[0] == "run-1"
    assert args[1] == "xgboost"
    assert args[2] == 3
    assert args[4] == 0.87


def test_record_study_trials_bulk_inserts_one_row_per_trial():
    conn, cursor = _mock_conn_with_cursor(fetchone_return={"id": "trial-uuid"})
    recorder = LineageRecorder(conn)

    class _FakeTrial:
        def __init__(self, family, number, params, score):
            self.model_family = family
            self.trial_number = number
            self.params = params
            self.score = score

    trials = [
        _FakeTrial("xgboost", 0, {"max_depth": 3}, 0.8),
        _FakeTrial("xgboost", 1, {"max_depth": 5}, 0.85),
        _FakeTrial("linear", 0, {"C": 1.0}, 0.7),
    ]

    trial_ids = recorder.record_study_trials("run-1", trials)

    assert len(trial_ids) == 3
    insert_calls = [
        c for c in cursor.execute.call_args_list if "INSERT INTO hyperparameters" in c.args[0]
    ]
    assert len(insert_calls) == 3


def test_record_metric_inserts_expected_row_with_ci():
    conn, cursor = _mock_conn_with_cursor(fetchone_return={"id": "metric-uuid-1"})
    recorder = LineageRecorder(conn)

    metric_id = recorder.record_metric(
        "run-1", name="stacking_cv_score", value=0.91, ci_low=0.88, ci_high=0.94
    )

    assert metric_id == "metric-uuid-1"
    insert_sql, args = cursor.execute.call_args_list[0].args
    assert "INSERT INTO metrics" in insert_sql
    assert args == ("run-1", "stacking_cv_score", 0.91, 0.88, 0.94)


def test_record_metric_defaults_ci_to_none():
    conn, cursor = _mock_conn_with_cursor(fetchone_return={"id": "metric-uuid-2"})
    recorder = LineageRecorder(conn)

    recorder.record_metric("run-1", name="default_baseline_score", value=0.75)

    _, args = cursor.execute.call_args_list[0].args
    assert args[3] is None
    assert args[4] is None


def test_record_rl_episode_inserts_expected_row():
    conn, cursor = _mock_conn_with_cursor(fetchone_return={"id": "episode-uuid-1"})
    recorder = LineageRecorder(conn)

    episode_id = recorder.record_rl_episode(
        episode_number=42,
        state={"missingness_rate": 0.1, "skew": 0.5},
        action={"imputer": "mice", "outlier_method": "lof", "threshold_bin": 2},
        reward=0.03,
        run_id="run-1",
    )

    assert episode_id == "episode-uuid-1"
    insert_sql, args = cursor.execute.call_args_list[0].args
    assert "INSERT INTO rl_episodes" in insert_sql
    assert args[0] == 42
    assert args[3] == 0.03
    assert args[4] == "run-1"


def test_record_rl_episode_allows_null_run_id_for_surrogate_reward():
    conn, cursor = _mock_conn_with_cursor(fetchone_return={"id": "episode-uuid-2"})
    recorder = LineageRecorder(conn)

    recorder.record_rl_episode(
        episode_number=1,
        state={"missingness_rate": 0.2},
        action={"imputer": "knn", "outlier_method": "none", "threshold_bin": 0},
        reward=-0.01,
        run_id=None,
    )

    _, args = cursor.execute.call_args_list[0].args
    assert args[4] is None
