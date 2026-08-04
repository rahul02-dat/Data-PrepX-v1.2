from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.pipeline.config import DriftGateConfig, MaxNullRateGateConfig, SchemaConformanceGateConfig


@dataclass(frozen=True)
class GateResult:
    gate_name: str
    passed: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateChainResult:
    passed: bool
    results: list[GateResult]

    # Return failed gate evaluation results
    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]


class Gate(ABC):
    name: str

    # Evaluate gate criteria against dataset
    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        expected_schema: dict[str, str] | None = None,
        reference_df: pd.DataFrame | None = None,
    ) -> GateResult:
        raise NotImplementedError


class MaxNullRateGate(Gate):
    name = "max_null_rate_gate"

    def __init__(self, config: MaxNullRateGateConfig):
        self._config = config

    # Evaluate null rate thresholds against dataset
    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        expected_schema: dict[str, str] | None = None,
        reference_df: pd.DataFrame | None = None,
    ) -> GateResult:
        if len(df) == 0:
            return GateResult(self.name, passed=False, reason="dataset has zero rows", details={})

        if self._config.per_column:
            null_rates = (df.isna().sum() / len(df)).to_dict()
            offenders = {
                col: rate for col, rate in null_rates.items() if rate > self._config.threshold
            }
            if offenders:
                return GateResult(
                    self.name,
                    passed=False,
                    reason=(
                        f"{len(offenders)} column(s) exceed max null rate "
                        f"{self._config.threshold}"
                    ),
                    details={"null_rates": null_rates, "offending_columns": offenders},
                )
            return GateResult(self.name, passed=True, details={"null_rates": null_rates})

        overall_rate = float(df.isna().sum().sum() / (df.shape[0] * max(df.shape[1], 1)))
        if overall_rate > self._config.threshold:
            return GateResult(
                self.name,
                passed=False,
                reason=f"overall null rate {overall_rate:.4f} exceeds threshold "
                f"{self._config.threshold}",
                details={"overall_null_rate": overall_rate},
            )
        return GateResult(self.name, passed=True, details={"overall_null_rate": overall_rate})


class SchemaConformanceGate(Gate):
    name = "schema_conformance_gate"

    def __init__(self, config: SchemaConformanceGateConfig):
        self._config = config

    # Evaluate schema conformance against expected schema
    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        expected_schema: dict[str, str] | None = None,
        reference_df: pd.DataFrame | None = None,
    ) -> GateResult:
        if not expected_schema:
            return GateResult(
                self.name,
                passed=False,
                reason="no expected_schema supplied; cannot evaluate schema conformance",
                details={},
            )

        actual_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        expected_cols = set(expected_schema.keys())
        actual_cols = set(actual_schema.keys())

        missing_cols = sorted(expected_cols - actual_cols)
        extra_cols = sorted(actual_cols - expected_cols)

        dtype_mismatches: dict[str, dict[str, str]] = {}
        if self._config.strict_dtypes:
            for col in expected_cols & actual_cols:
                if expected_schema[col] != actual_schema[col]:
                    dtype_mismatches[col] = {
                        "expected": expected_schema[col],
                        "actual": actual_schema[col],
                    }

        problems = []
        if missing_cols:
            problems.append(f"missing columns: {missing_cols}")
        if extra_cols and not self._config.allow_extra_columns:
            problems.append(f"unexpected extra columns: {extra_cols}")
        if dtype_mismatches:
            problems.append(f"dtype mismatches: {dtype_mismatches}")

        details = {
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "dtype_mismatches": dtype_mismatches,
        }

        if problems:
            return GateResult(self.name, passed=False, reason="; ".join(problems), details=details)
        return GateResult(self.name, passed=True, details=details)


class DriftGate(Gate):
    name = "drift_gate"

    def __init__(self, config: DriftGateConfig):
        self._config = config

    # Evaluate distribution drift against reference dataset
    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        expected_schema: dict[str, str] | None = None,
        reference_df: pd.DataFrame | None = None,
    ) -> GateResult:
        if reference_df is None or len(reference_df) == 0:
            return GateResult(
                self.name,
                passed=False,
                reason="no reference dataset configured; drift_gate requires an "
                "explicit user-supplied reference (see ADR 0002)",
                details={},
            )

        numeric_cols = [
            c
            for c in df.columns
            if c in reference_df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            return GateResult(
                self.name,
                passed=False,
                reason="no shared numeric columns between dataset and reference to compare",
                details={},
            )

        per_column: dict[str, dict[str, float]] = {}
        drifted_columns: list[str] = []

        for col in numeric_cols:
            current = df[col].dropna().to_numpy(dtype=float)
            reference = reference_df[col].dropna().to_numpy(dtype=float)
            if len(current) == 0 or len(reference) == 0:
                continue

            if self._config.method == "psi":
                score = _population_stability_index(reference, current, bins=self._config.psi_bins)
                per_column[col] = {"psi": score}
                if score > self._config.psi_threshold:
                    drifted_columns.append(col)
            else:
                _, p_value = stats.ks_2samp(reference, current)
                per_column[col] = {"ks_p_value": float(p_value)}
                if p_value < self._config.ks_p_value_threshold:
                    drifted_columns.append(col)

        if drifted_columns:
            return GateResult(
                self.name,
                passed=False,
                reason=f"drift detected ({self._config.method}) in columns: {drifted_columns}",
                details={"per_column": per_column, "drifted_columns": drifted_columns},
            )
        return GateResult(self.name, passed=True, details={"per_column": per_column})


# Calculate Population Stability Index metric
def _population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    epsilon = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference.mean(), current.mean()) else float("inf")

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.maximum(ref_counts / max(len(reference), 1), epsilon)
    cur_pct = np.maximum(cur_counts / max(len(current), 1), epsilon)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# Construct configured validation gate list
def build_gates(pipeline_config) -> list[Gate]:
    gates: list[Gate] = []
    if pipeline_config.max_null_rate_gate.enabled:
        gates.append(MaxNullRateGate(pipeline_config.max_null_rate_gate))
    if pipeline_config.schema_conformance_gate.enabled:
        gates.append(SchemaConformanceGate(pipeline_config.schema_conformance_gate))
    if pipeline_config.drift_gate.enabled:
        gates.append(DriftGate(pipeline_config.drift_gate))
    return gates


# Execute gate chain against dataset
def run_gates(
    gates: list[Gate],
    df: pd.DataFrame,
    *,
    expected_schema: dict[str, str] | None = None,
    reference_df: pd.DataFrame | None = None,
) -> GateChainResult:
    results = [
        gate.evaluate(df, expected_schema=expected_schema, reference_df=reference_df)
        for gate in gates
    ]
    return GateChainResult(passed=all(r.passed for r in results), results=results)
