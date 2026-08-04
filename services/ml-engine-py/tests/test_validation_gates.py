import numpy as np
import pandas as pd

from app.pipeline import config as config_module
from app.pipeline.config import DriftGateConfig, MaxNullRateGateConfig, SchemaConformanceGateConfig
from app.pipeline.validation_gates import (
    DriftGate,
    MaxNullRateGate,
    SchemaConformanceGate,
    build_gates,
    run_gates,
)

# --- MaxNullRateGate ---------------------------------------------------------


def test_max_null_rate_gate_passes_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gate = MaxNullRateGate(MaxNullRateGateConfig(threshold=0.4, per_column=True))
    result = gate.evaluate(df)
    assert result.passed is True


def test_max_null_rate_gate_fails_on_offending_column():
    df = pd.DataFrame({"a": [1, None, None, None], "b": [1, 2, 3, 4]})
    gate = MaxNullRateGate(MaxNullRateGateConfig(threshold=0.4, per_column=True))
    result = gate.evaluate(df)
    assert result.passed is False
    assert "a" in result.details["offending_columns"]
    assert result.reason is not None


def test_max_null_rate_gate_overall_mode():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": [1, 2, 3, 4]})
    gate = MaxNullRateGate(MaxNullRateGateConfig(threshold=0.5, per_column=False))
    result = gate.evaluate(df)
    assert result.passed is True  # 1/8 = 0.125 overall


def test_max_null_rate_gate_rejects_empty_dataset():
    gate = MaxNullRateGate(MaxNullRateGateConfig())
    result = gate.evaluate(pd.DataFrame())
    assert result.passed is False


# --- SchemaConformanceGate ----------------------------------------------------


def test_schema_conformance_passes_matching_schema():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    expected = {col: str(dt) for col, dt in df.dtypes.items()}
    gate = SchemaConformanceGate(SchemaConformanceGateConfig())
    result = gate.evaluate(df, expected_schema=expected)
    assert result.passed is True


def test_schema_conformance_fails_on_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    expected = {"a": "int64", "b": "object"}
    gate = SchemaConformanceGate(SchemaConformanceGateConfig())
    result = gate.evaluate(df, expected_schema=expected)
    assert result.passed is False
    assert "b" in result.details["missing_columns"]


def test_schema_conformance_fails_on_unexpected_extra_column_by_default():
    df = pd.DataFrame({"a": [1], "b": [2]})
    expected = {"a": "int64"}
    gate = SchemaConformanceGate(SchemaConformanceGateConfig(allow_extra_columns=False))
    result = gate.evaluate(df, expected_schema=expected)
    assert result.passed is False
    assert "b" in result.details["extra_columns"]


def test_schema_conformance_allows_extra_column_when_configured():
    df = pd.DataFrame({"a": [1], "b": [2]})
    expected = {"a": "int64"}
    gate = SchemaConformanceGate(SchemaConformanceGateConfig(allow_extra_columns=True))
    result = gate.evaluate(df, expected_schema=expected)
    assert result.passed is True


def test_schema_conformance_fails_on_dtype_mismatch():
    df = pd.DataFrame({"a": [1.0, 2.0]})  # float64
    expected = {"a": "int64"}
    gate = SchemaConformanceGate(SchemaConformanceGateConfig(strict_dtypes=True))
    result = gate.evaluate(df, expected_schema=expected)
    assert result.passed is False
    assert "a" in result.details["dtype_mismatches"]


def test_schema_conformance_fails_closed_without_expected_schema():
    df = pd.DataFrame({"a": [1]})
    gate = SchemaConformanceGate(SchemaConformanceGateConfig())
    result = gate.evaluate(df)
    assert result.passed is False


# --- DriftGate -----------------------------------------------------------------


def test_drift_gate_fails_closed_with_no_reference():
    df = pd.DataFrame({"a": np.random.default_rng(0).normal(size=100)})
    gate = DriftGate(DriftGateConfig())
    result = gate.evaluate(df, reference_df=None)
    assert result.passed is False
    assert "reference" in result.reason.lower()


def test_drift_gate_psi_passes_identical_distributions():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"a": rng.normal(size=1000)})
    current = pd.DataFrame({"a": rng.normal(size=1000)})
    gate = DriftGate(DriftGateConfig(method="psi", psi_threshold=0.25))
    result = gate.evaluate(current, reference_df=reference)
    assert result.passed is True


def test_drift_gate_psi_fails_on_shifted_distribution():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"a": rng.normal(loc=0, scale=1, size=1000)})
    current = pd.DataFrame({"a": rng.normal(loc=5, scale=1, size=1000)})
    gate = DriftGate(DriftGateConfig(method="psi", psi_threshold=0.25))
    result = gate.evaluate(current, reference_df=reference)
    assert result.passed is False
    assert "a" in result.details["drifted_columns"]


def test_drift_gate_ks_method_fails_on_shifted_distribution():
    rng = np.random.default_rng(7)
    reference = pd.DataFrame({"a": rng.normal(loc=0, size=500)})
    current = pd.DataFrame({"a": rng.normal(loc=3, size=500)})
    gate = DriftGate(DriftGateConfig(method="ks", ks_p_value_threshold=0.05))
    result = gate.evaluate(current, reference_df=reference)
    assert result.passed is False


def test_drift_gate_no_shared_numeric_columns():
    reference = pd.DataFrame({"x": [1, 2, 3]})
    current = pd.DataFrame({"y": [1, 2, 3]})
    gate = DriftGate(DriftGateConfig())
    result = gate.evaluate(current, reference_df=reference)
    assert result.passed is False


# --- run_gates / build_gates orchestration -------------------------------------


def test_run_gates_aggregates_all_results_even_on_first_failure():
    df = pd.DataFrame({"a": [None, None, None, 1]})
    null_gate = MaxNullRateGate(MaxNullRateGateConfig(threshold=0.1, per_column=True))
    schema_gate = SchemaConformanceGate(SchemaConformanceGateConfig())
    chain = run_gates([null_gate, schema_gate], df)  # no expected_schema -> schema also fails
    assert chain.passed is False
    assert len(chain.results) == 2
    assert len(chain.failures) == 2


def test_build_gates_respects_config_toggles():
    config = config_module.PipelineConfig(
        max_null_rate_gate=MaxNullRateGateConfig(enabled=False),
        schema_conformance_gate=SchemaConformanceGateConfig(enabled=True),
        drift_gate=DriftGateConfig(enabled=False),
    )
    gates = build_gates(config)
    assert len(gates) == 1
    assert isinstance(gates[0], SchemaConformanceGate)


def test_run_gates_passes_when_all_gates_pass():
    df = pd.DataFrame({"a": [1, 2, 3]})
    expected = {"a": "int64"}
    gate = SchemaConformanceGate(SchemaConformanceGateConfig())
    chain = run_gates([gate], df, expected_schema=expected)
    assert chain.passed is True
    assert chain.failures == []
