from pathlib import Path

import pytest

from app.pipeline.config import DriftGateConfig, load_pipeline_config

REPO_CONFIG = Path(__file__).resolve().parents[3] / "config" / "gates.yaml"


def test_loads_repo_config_defaults():
    config = load_pipeline_config(REPO_CONFIG)
    assert config.max_null_rate_gate.enabled is True
    assert config.max_null_rate_gate.threshold == 0.4
    assert config.schema_conformance_gate.strict_dtypes is True
    assert config.drift_gate.method == "psi"


def test_missing_config_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_pipeline_config(missing)


def test_invalid_drift_method_rejected():
    with pytest.raises(ValueError):
        DriftGateConfig(method="not_a_method")


def test_partial_yaml_falls_back_to_dataclass_defaults(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text("max_null_rate_gate:\n  threshold: 0.9\n")
    config = load_pipeline_config(path)
    assert config.max_null_rate_gate.threshold == 0.9
    assert config.max_null_rate_gate.per_column is True  # dataclass default
    assert config.drift_gate.psi_threshold == 0.25  # dataclass default
