from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "gates.yaml"


@dataclass(frozen=True)
class MaxNullRateGateConfig:
    enabled: bool = True
    threshold: float = 0.4
    per_column: bool = True


@dataclass(frozen=True)
class SchemaConformanceGateConfig:
    enabled: bool = True
    allow_extra_columns: bool = False
    strict_dtypes: bool = True


@dataclass(frozen=True)
class DriftGateConfig:
    enabled: bool = True
    method: str = "psi"
    psi_threshold: float = 0.25
    ks_p_value_threshold: float = 0.05
    psi_bins: int = 10

    # Validate configuration parameters
    def __post_init__(self) -> None:
        if self.method not in ("psi", "ks"):
            raise ValueError(f"drift_gate.method must be 'psi' or 'ks', got {self.method!r}")


@dataclass(frozen=True)
class ImputationConfig:
    enabled: bool = True
    # "mice" (IterativeImputer, chained-equations-style) or "knn" (KNNImputer). Numeric columns
    method: str = "mice"
    mice_max_iter: int = 25
    mice_tol: float = 1e-2
    knn_n_neighbors: int = 5
    # Non-numeric columns always use most-frequent imputation; MICE/KNN require numeric input.
    categorical_strategy: str = "most_frequent"

    # Validate configuration parameters
    def __post_init__(self) -> None:
        if self.method not in ("mice", "knn"):
            raise ValueError(f"imputation.method must be 'mice' or 'knn', got {self.method!r}")
        if self.categorical_strategy not in ("most_frequent",):
            raise ValueError(
                f"imputation.categorical_strategy must be 'most_frequent', "
                f"got {self.categorical_strategy!r}"
            )


@dataclass(frozen=True)
class OutlierDetectionConfig:
    enabled: bool = True
    # "isolation_forest", "lof", or "none". Produces a continuous anomaly score per row
    method: str = "isolation_forest"
    contamination: float = 0.05
    n_estimators: int = 100
    lof_n_neighbors: int = 20

    # Validate configuration parameters
    def __post_init__(self) -> None:
        if self.method not in ("isolation_forest", "lof", "none"):
            raise ValueError(
                f"outlier_detection.method must be 'isolation_forest', 'lof', or 'none', "
                f"got {self.method!r}"
            )
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(
                f"outlier_detection.contamination must be in (0, 0.5), got {self.contamination}"
            )


@dataclass(frozen=True)
class PipelineConfig:
    max_null_rate_gate: MaxNullRateGateConfig = field(default_factory=MaxNullRateGateConfig)
    schema_conformance_gate: SchemaConformanceGateConfig = field(
        default_factory=SchemaConformanceGateConfig
    )
    drift_gate: DriftGateConfig = field(default_factory=DriftGateConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    outlier_detection: OutlierDetectionConfig = field(default_factory=OutlierDetectionConfig)

    # Convert dataclass configuration to dictionary
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Resolve path to gate configuration file
def config_path() -> Path:
    override = os.environ.get("GATE_CONFIG_PATH")
    return Path(override) if override else _DEFAULT_CONFIG_PATH


# Load and validate pipeline configuration
def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    resolved = path or config_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Gate config not found at {resolved}. Set GATE_CONFIG_PATH or restore "
            "config/gates.yaml."
        )
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig(
        max_null_rate_gate=MaxNullRateGateConfig(**raw.get("max_null_rate_gate", {})),
        schema_conformance_gate=SchemaConformanceGateConfig(
            **raw.get("schema_conformance_gate", {})
        ),
        drift_gate=DriftGateConfig(**raw.get("drift_gate", {})),
        imputation=ImputationConfig(**raw.get("imputation", {})),
        outlier_detection=OutlierDetectionConfig(**raw.get("outlier_detection", {})),
    )
