import pytest

from config import SummarizerConfig
from graphs.nodes import (
    compute_stats,
    emit_or_flag,
    retrieve_grounding_facts,
    score_confidence,
    verify_claim_against_stats,
)
from graphs.schemas import DraftedClaim, Metric

CONFIG = SummarizerConfig()


def test_compute_stats_derives_ci_width_and_relative_width():
    metrics = [Metric(name="accuracy", value=0.9, ci_low=0.88, ci_high=0.92)]
    stats = compute_stats(metrics)
    assert stats[0].ci_width == pytest.approx(0.04)
    assert stats[0].relative_ci_width == pytest.approx(0.04 / 0.9)


def test_compute_stats_handles_missing_ci():
    metrics = [Metric(name="accuracy", value=0.9)]
    stats = compute_stats(metrics)
    assert stats[0].ci_width is None
    assert stats[0].relative_ci_width is None


def test_compute_stats_handles_zero_value_without_dividing_by_zero():
    metrics = [Metric(name="delta", value=0.0, ci_low=-0.1, ci_high=0.1)]
    stats = compute_stats(metrics)
    assert stats[0].ci_width == pytest.approx(0.2)
    assert stats[0].relative_ci_width is None


def test_retrieve_grounding_facts_includes_ci_when_present():
    metrics = [Metric(name="accuracy", value=0.9, ci_low=0.88, ci_high=0.92)]
    facts = retrieve_grounding_facts(compute_stats(metrics))
    assert "0.9" in facts[0].text
    assert "CI" in facts[0].text


def test_retrieve_grounding_facts_notes_missing_ci():
    metrics = [Metric(name="accuracy", value=0.9)]
    facts = retrieve_grounding_facts(compute_stats(metrics))
    assert "no confidence interval" in facts[0].text


def test_verify_claim_accepts_matching_value():
    stats = compute_stats([Metric(name="accuracy", value=0.9, ci_low=0.85, ci_high=0.95)])
    stats_by_name = {s.name: s for s in stats}
    claims = [DraftedClaim(metric_name="accuracy", stated_value=0.9, statement="Accuracy was 0.9.")]
    result = verify_claim_against_stats(claims, stats_by_name, config=CONFIG)
    assert result[0].verified is True
    assert result[0].rejection_reason is None


def test_verify_claim_rejects_value_outside_tolerance():
    stats = compute_stats([Metric(name="accuracy", value=0.9, ci_low=0.85, ci_high=0.95)])
    stats_by_name = {s.name: s for s in stats}
    claims = [
        DraftedClaim(metric_name="accuracy", stated_value=0.95, statement="Accuracy was 0.95.")
    ]
    result = verify_claim_against_stats(claims, stats_by_name, config=CONFIG)
    assert result[0].verified is False
    assert result[0].rejection_reason is not None


def test_verify_claim_rejects_unknown_metric():
    stats_by_name = {"accuracy": compute_stats([Metric(name="accuracy", value=0.9)])[0]}
    claims = [DraftedClaim(metric_name="f1_score", stated_value=0.5, statement="F1 was 0.5.")]
    result = verify_claim_against_stats(claims, stats_by_name, config=CONFIG)
    assert result[0].verified is False
    assert "unknown metric" in result[0].rejection_reason


def test_verify_claim_within_tolerance_still_passes():
    stats = compute_stats([Metric(name="accuracy", value=0.900, ci_low=0.85, ci_high=0.95)])
    stats_by_name = {s.name: s for s in stats}
    # 0.9005 differs by 0.0005, well within the default 2% relative tolerance of 0.018.
    claims = [DraftedClaim(metric_name="accuracy", stated_value=0.9005, statement="~0.9")]
    result = verify_claim_against_stats(claims, stats_by_name, config=CONFIG)
    assert result[0].verified is True


def test_score_confidence_tight_ci_is_high():
    stats = compute_stats([Metric(name="accuracy", value=0.9, ci_low=0.895, ci_high=0.905)])
    stats_by_name = {s.name: s for s in stats}
    verified = verify_claim_against_stats(
        [DraftedClaim(metric_name="accuracy", stated_value=0.9, statement="x")],
        stats_by_name,
        config=CONFIG,
    )
    scored = score_confidence(verified, stats_by_name, config=CONFIG)
    assert scored[0].confidence_label == "high"
    assert scored[0].confidence_score is not None


def test_score_confidence_wide_ci_is_inconclusive():
    stats = compute_stats([Metric(name="accuracy", value=0.9, ci_low=0.3, ci_high=1.5)])
    stats_by_name = {s.name: s for s in stats}
    verified = verify_claim_against_stats(
        [DraftedClaim(metric_name="accuracy", stated_value=0.9, statement="x")],
        stats_by_name,
        config=CONFIG,
    )
    scored = score_confidence(verified, stats_by_name, config=CONFIG)
    assert scored[0].confidence_label == "inconclusive"


def test_score_confidence_missing_ci_is_unknown_not_high():
    stats = compute_stats([Metric(name="accuracy", value=0.9)])
    stats_by_name = {s.name: s for s in stats}
    verified = verify_claim_against_stats(
        [DraftedClaim(metric_name="accuracy", stated_value=0.9, statement="x")],
        stats_by_name,
        config=CONFIG,
    )
    scored = score_confidence(verified, stats_by_name, config=CONFIG)
    assert scored[0].confidence_label == "unknown"
    assert scored[0].confidence_score is None


def test_score_confidence_skips_unverified_claims():
    stats = compute_stats([Metric(name="accuracy", value=0.9, ci_low=0.85, ci_high=0.95)])
    stats_by_name = {s.name: s for s in stats}
    verified = verify_claim_against_stats(
        [DraftedClaim(metric_name="accuracy", stated_value=5.0, statement="wrong")],
        stats_by_name,
        config=CONFIG,
    )
    scored = score_confidence(verified, stats_by_name, config=CONFIG)
    assert scored == []


def test_emit_or_flag_splits_by_confidence_and_verification():
    stats = compute_stats(
        [
            Metric(name="high_conf", value=0.9, ci_low=0.895, ci_high=0.905),
            Metric(name="low_conf", value=0.9, ci_low=0.3, ci_high=1.5),
        ]
    )
    stats_by_name = {s.name: s for s in stats}
    claims = [
        DraftedClaim(metric_name="high_conf", stated_value=0.9, statement="a"),
        DraftedClaim(metric_name="low_conf", stated_value=0.9, statement="b"),
        DraftedClaim(metric_name="high_conf", stated_value=99.0, statement="wrong"),
    ]
    verified = verify_claim_against_stats(claims, stats_by_name, config=CONFIG)
    scored = score_confidence(verified, stats_by_name, config=CONFIG)
    report = emit_or_flag(scored, verified)

    assert len(report.accepted_claims) == 1
    assert report.accepted_claims[0].metric_name == "high_conf"
    assert len(report.flagged_claims) == 1
    assert report.flagged_claims[0].metric_name == "low_conf"
    assert len(report.rejected_claims) == 1
    assert report.rejected_claims[0].statement == "wrong"
