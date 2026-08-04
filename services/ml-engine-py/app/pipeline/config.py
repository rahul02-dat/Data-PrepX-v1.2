"""Central, versioned gate configuration (CLAUDE.md §8: "central YAML config, no
hardcoded thresholds -- everything the RL agent or gates use should be tunable and
itself versioned").

Loaded from config/gates.yaml at the repo root by default; override with the
GATE_CONFIG_PATH env var (used by tests and by non-default deployments -- CLAUDE.md
§8 also says dev/prod must not silently diverge on gate thresholds, so a different
path should come with an ADR, not just an env var flip in compose files).
"""

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

    def __post_init__(self) -> None:
        if self.method not in ("psi", "ks"):
            raise ValueError(f"drift_gate.method must be 'psi' or 'ks', got {self.method!r}")


@dataclass(frozen=True)
class PipelineConfig:
    max_null_rate_gate: MaxNullRateGateConfig = field(default_factory=MaxNullRateGateConfig)
    schema_conformance_gate: SchemaConformanceGateConfig = field(
        default_factory=SchemaConformanceGateConfig
    )
    drift_gate: DriftGateConfig = field(default_factory=DriftGateConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_path() -> Path:
    override = os.environ.get("GATE_CONFIG_PATH")
    return Path(override) if override else _DEFAULT_CONFIG_PATH


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    """Load and validate config/gates.yaml (or GATE_CONFIG_PATH) into a PipelineConfig.

    Fails loudly (FileNotFoundError / dataclass ValueError) rather than silently
    falling back to defaults, per CLAUDE.md's "no silent defaults for lineage-
    relevant fields" stance applied to gate config.
    """
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
    )
