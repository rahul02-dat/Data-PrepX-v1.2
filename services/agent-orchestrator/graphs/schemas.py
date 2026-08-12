from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ConfidenceLabel = Literal["high", "moderate", "inconclusive", "unknown"]


@dataclass(frozen=True)
class Metric:
    """One pre-computed, gate-approved statistic handed to agent-orchestrator by ml-engine-py.
    Mirrors the `metrics` table (CLAUDE.md §6): name, value, and optional CI bounds. This is the
    ONLY data agent-orchestrator ever sees -- never a raw dataframe (CLAUDE.md §2)."""

    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }

    @staticmethod
    def from_dict(d: dict) -> Metric:
        return Metric(
            name=d["name"], value=d["value"], ci_low=d.get("ci_low"), ci_high=d.get("ci_high")
        )


@dataclass(frozen=True)
class ComputedStat:
    """A Metric enriched with the derived quantities confidence scoring needs. Computed once,
    deterministically, in compute_stats -- never recomputed or restated by the LLM."""

    name: str
    value: float
    ci_low: float | None
    ci_high: float | None
    ci_width: float | None
    relative_ci_width: float | None  # None when value == 0 or no CI is available

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_width": self.ci_width,
            "relative_ci_width": self.relative_ci_width,
        }


@dataclass(frozen=True)
class GroundingFact:
    """A plain-language, deterministically-formatted rendering of one ComputedStat, handed to
    the LLM as the ONLY grounding material it is allowed to draft claims from."""

    metric_name: str
    text: str

    def to_dict(self) -> dict:
        return {"metric_name": self.metric_name, "text": self.text}


@dataclass(frozen=True)
class DraftedClaim:
    """One claim as drafted by the LLM: a natural-language statement plus the specific numeric
    value it asserts for a specific metric. This structure -- not free text -- is what makes
    verify_claim_against_stats possible: there is an exact (metric_name, stated_value) pair to
    check, not prose to parse."""

    metric_name: str
    stated_value: float
    statement: str

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "stated_value": self.stated_value,
            "statement": self.statement,
        }

    @staticmethod
    def from_dict(d: dict) -> DraftedClaim:
        return DraftedClaim(
            metric_name=d["metric_name"],
            stated_value=float(d["stated_value"]),
            statement=d["statement"],
        )


@dataclass(frozen=True)
class VerifiedClaim:
    """Output of verify_claim_against_stats: whether the LLM's stated value actually matches the
    real computed statistic, within tolerance."""

    metric_name: str
    stated_value: float
    actual_value: float
    statement: str
    verified: bool
    rejection_reason: str | None

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "stated_value": self.stated_value,
            "actual_value": self.actual_value,
            "statement": self.statement,
            "verified": self.verified,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class ScoredClaim:
    """A verified claim plus its confidence label/score, per CLAUDE.md §5.4: "Confidence scores
    derive from effect size vs. variance/CI width"."""

    metric_name: str
    statement: str
    actual_value: float
    confidence_label: ConfidenceLabel
    confidence_score: float | None  # None only when confidence_label == "unknown"

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "statement": self.statement,
            "actual_value": self.actual_value,
            "confidence_label": self.confidence_label,
            "confidence_score": self.confidence_score,
        }
