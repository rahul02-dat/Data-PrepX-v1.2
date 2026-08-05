import pandas as pd

from app.pipeline.hashing import compute_run_key, hash_config, hash_dataframe, hash_source


def test_hash_config_is_order_independent():
    a = hash_config({"x": 1, "y": 2})
    b = hash_config({"y": 2, "x": 1})
    assert a == b


def test_hash_config_prefix_and_determinism():
    h = hash_config({"a": 1})
    assert h.startswith("sha256:")
    assert h == hash_config({"a": 1})


def test_hash_config_empty_and_none_agree():
    assert hash_config(None) == hash_config({})


def test_hash_config_differs_for_different_values():
    assert hash_config({"a": 1}) != hash_config({"a": 2})


def test_hash_config_matches_known_go_style_vector():
    import hashlib
    import json

    config = {"threshold": 0.4, "enabled": True}
    ordered = ["enabled", True, "threshold", 0.4]
    expected_bytes = json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    expected = "sha256:" + hashlib.sha256(expected_bytes).hexdigest()
    assert hash_config(config) == expected


def test_hash_dataframe_is_deterministic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert hash_dataframe(df) == hash_dataframe(df.copy())


def test_hash_dataframe_changes_with_values():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 4]})
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_hash_dataframe_changes_with_column_order():
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = df1[["b", "a"]]
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_hash_dataframe_insensitive_to_row_index_labels():
    df1 = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 11, 12])
    assert hash_dataframe(df1) == hash_dataframe(df2)


def test_compute_run_key_deterministic_and_sensitive_to_each_input():
    base = compute_run_key("d1", "c1", "g1")
    assert base == compute_run_key("d1", "c1", "g1")
    assert base != compute_run_key("d2", "c1", "g1")
    assert base != compute_run_key("d1", "c2", "g1")
    assert base != compute_run_key("d1", "c1", "g2")


def test_hash_source_deterministic_and_prefixed():
    def sample_fn(x):
        return x + 1

    h = hash_source(sample_fn)
    assert h.startswith("sha256:")
    assert h == hash_source(sample_fn)


def test_hash_source_differs_when_implementation_changes():
    def fn_a(x):
        return x + 1

    def fn_b(x):
        return x + 2

    assert hash_source(fn_a) != hash_source(fn_b)
