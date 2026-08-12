from __future__ import annotations

from dataclasses import dataclass, field

from config import SummarizerConfig
from .schemas import (
    ComputedStat,
    DraftedClaim,
    GroundingFact,
    Metric,
    ScoredClaim,
    VerifiedClaim,
)

# ---------------------------------------------------------------------------
# compute_stats: deterministic. Turns raw Metric rows into ComputedStat rows carrying the
# derived quantities (ci_width, relative_ci_width) every downstream node needs. No LLM call.
# ---------------------------------------------------------------------------


def compute_stats(metrics: list[Metric]) -> list[ComputedStat]:
    stats: list[ComputedStat] = []
    for m in metrics:
        ci_width: float | None = None
        relative_ci_width: float | None = None
        if m.ci_low is not None and m.ci_high is not None:
            ci_width = m.ci_high - m.ci_low
            if m.value != 0:
                relative_ci_width = abs(ci_width / m.value)
        stats.append(
            ComputedStat(
                name=m.name,
                value=m.value,
                ci_low=m.ci_low,
                ci_high=m.ci_high,
                ci_width=ci_width,
                relative_ci_width=relative_ci_width,
            )
        )
    return stats


# ---------------------------------------------------------------------------
# retrieve_grounding_facts: deterministic. Formats each ComputedStat into a plain-language
# fact string. This -- not raw data -- is the only material the LLM is allowed to see
# (CLAUDE.md §2: agent-orchestrator never receives raw dataframes).
# ---------------------------------------------------------------------------


def retrieve_grounding_facts(stats: list[ComputedStat]) -> list[GroundingFact]:
    facts: list[GroundingFact] = []
    for s in stats:
        if s.ci_low is not None and s.ci_high is not None:
            text = f"{s.name} = {s.value:.6g} (95% CI [{s.ci_low:.6g}, {s.ci_high:.6g}])"
        else:
            text = f"{s.name} = {s.value:.6g} (no confidence interval available)"
        facts.append(GroundingFact(metric_name=s.name, text=text))
    return facts


# ---------------------------------------------------------------------------
# verify_claim_against_stats: deterministic, NOT an LLM call (CLAUDE.md §5.4). Every numeric
# assertion the LLM drafted is re-checked against the real computed statistic before it can
# reach the final report.
# ---------------------------------------------------------------------------


def verify_claim_against_stats(
    claims: list[DraftedClaim],
    stats_by_name: dict[str, ComputedStat],
    *,
    config: SummarizerConfig,
) -> list[VerifiedClaim]:
    verified: list[VerifiedClaim] = []
    for claim in claims:
        stat = stats_by_name.get(claim.metric_name)

        if stat is None:
            verified.append(
                VerifiedClaim(
                    metric_name=claim.metric_name,
                    stated_value=claim.stated_value,
                    actual_value=float("nan"),
                    statement=claim.statement,
                    verified=False,
                    rejection_reason=f"claim references unknown metric {claim.metric_name!r}; "
                    "not present in the computed statistics supplied to this run",
                )
            )
            continue

        tolerance = config.verification_relative_tolerance * max(abs(stat.value), 1e-9)
        difference = abs(claim.stated_value - stat.value)
        matches = difference <= tolerance

        verified.append(
            VerifiedClaim(
                metric_name=claim.metric_name,
                stated_value=claim.stated_value,
                actual_value=stat.value,
                statement=claim.statement,
                verified=matches,
                rejection_reason=(
                    None
                    if matches
                    else (
                        f"stated value {claim.stated_value:.6g} does not match the computed "
                        f"statistic {stat.value:.6g} (tolerance {tolerance:.6g})"
                    )
                ),
            )
        )
    return verified


# ---------------------------------------------------------------------------
# score_confidence: deterministic. Confidence derives from relative CI width, per CLAUDE.md
# §5.4 ("effect size vs. variance/CI width"), never from the LLM's own phrasing or hedging.
# ---------------------------------------------------------------------------


def _confidence_for_relative_width(
    relative_ci_width: float | None, config: SummarizerConfig
) -> tuple[str, float | None]:
    if relative_ci_width is None:
        return "unknown", None
    # Monotonically decreasing score in [0, 1] as relative CI width grows.
    score = 1.0 / (1.0 + relative_ci_width)
    if relative_ci_width <= config.high_confidence_max_relative_ci_width:
        return "high", score
    if relative_ci_width >= config.inconclusive_min_relative_ci_width:
        return "inconclusive", score
    return "moderate", score


def score_confidence(
    verified_claims: list[VerifiedClaim],
    stats_by_name: dict[str, ComputedStat],
    *,
    config: SummarizerConfig,
) -> list[ScoredClaim]:
    scored: list[ScoredClaim] = []
    for claim in verified_claims:
        if not claim.verified:
            continue  # unverifiable claims never reach confidence scoring or the report
        stat = stats_by_name[claim.metric_name]
        label, score = _confidence_for_relative_width(stat.relative_ci_width, config)
        scored.append(
            ScoredClaim(
                metric_name=claim.metric_name,
                statement=claim.statement,
                actual_value=stat.value,
                confidence_label=label,
                confidence_score=score,
            )
        )
    return scored


# ---------------------------------------------------------------------------
# emit_or_flag: deterministic. Splits scored claims into the accepted report body and a
# flagged section -- inconclusive/unknown-confidence findings are never stated as fact
# (CLAUDE.md §5.4: "explicitly flagged as inconclusive rather than stated as fact").
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryReport:
    accepted_claims: list[ScoredClaim] = field(default_factory=list)
    flagged_claims: list[ScoredClaim] = field(default_factory=list)
    rejected_claims: list[VerifiedClaim] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accepted_claims": [c.to_dict() for c in self.accepted_claims],
            "flagged_claims": [c.to_dict() for c in self.flagged_claims],
            "rejected_claims": [c.to_dict() for c in self.rejected_claims],
        }


def emit_or_flag(
    scored_claims: list[ScoredClaim], verified_claims: list[VerifiedClaim]
) -> SummaryReport:
    accepted = [c for c in scored_claims if c.confidence_label in ("high", "moderate")]
    flagged = [c for c in scored_claims if c.confidence_label in ("inconclusive", "unknown")]
    rejected = [c for c in verified_claims if not c.verified]
    return SummaryReport(accepted_claims=accepted, flagged_claims=flagged, rejected_claims=rejected)
