import json

from config import SummarizerConfig
from graphs.summarizer_graph import run_summarizer

CONFIG = SummarizerConfig()


def _fake_generate(claims: list[dict]):
    def _gen(prompt: str) -> str:
        return json.dumps(claims)

    return _gen


def test_full_graph_accepts_a_correct_high_confidence_claim():
    metrics = [{"name": "accuracy", "value": 0.9, "ci_low": 0.895, "ci_high": 0.905}]
    claims = [{"metric_name": "accuracy", "stated_value": 0.9, "statement": "Accuracy was 0.9."}]

    report = run_summarizer(metrics, config=CONFIG, generate_fn=_fake_generate(claims))

    assert len(report["accepted_claims"]) == 1
    assert report["accepted_claims"][0]["confidence_label"] == "high"
    assert report["flagged_claims"] == []
    assert report["rejected_claims"] == []


def test_full_graph_flags_an_inconclusive_claim_instead_of_asserting_it():
    metrics = [{"name": "accuracy", "value": 0.9, "ci_low": 0.2, "ci_high": 1.6}]
    claims = [{"metric_name": "accuracy", "stated_value": 0.9, "statement": "Accuracy was 0.9."}]

    report = run_summarizer(metrics, config=CONFIG, generate_fn=_fake_generate(claims))

    assert report["accepted_claims"] == []
    assert len(report["flagged_claims"]) == 1
    assert report["flagged_claims"][0]["confidence_label"] == "inconclusive"


def test_full_graph_rejects_a_hallucinated_numeric_claim():
    metrics = [{"name": "accuracy", "value": 0.9, "ci_low": 0.895, "ci_high": 0.905}]
    # The "model" states a value far from the real computed statistic.
    claims = [{"metric_name": "accuracy", "stated_value": 0.42, "statement": "Accuracy was 0.42."}]

    report = run_summarizer(metrics, config=CONFIG, generate_fn=_fake_generate(claims))

    assert report["accepted_claims"] == []
    assert report["flagged_claims"] == []
    assert len(report["rejected_claims"]) == 1
    assert report["rejected_claims"][0]["verified"] is False


def test_full_graph_rejects_claim_about_a_metric_not_supplied():
    metrics = [{"name": "accuracy", "value": 0.9, "ci_low": 0.895, "ci_high": 0.905}]
    claims = [{"metric_name": "f1_score", "stated_value": 0.8, "statement": "F1 was 0.8."}]

    report = run_summarizer(metrics, config=CONFIG, generate_fn=_fake_generate(claims))

    assert len(report["rejected_claims"]) == 1
    assert "unknown metric" in report["rejected_claims"][0]["rejection_reason"]


def test_full_graph_handles_malformed_llm_output_without_crashing():
    metrics = [{"name": "accuracy", "value": 0.9, "ci_low": 0.895, "ci_high": 0.905}]

    def _bad_gen(prompt: str) -> str:
        return "I think accuracy is pretty good honestly."

    report = run_summarizer(metrics, config=CONFIG, generate_fn=_bad_gen)

    assert report["accepted_claims"] == []
    assert report["flagged_claims"] == []
    assert report["rejected_claims"] == []
    assert "draft_error" in report


def test_full_graph_handles_multiple_metrics_independently():
    metrics = [
        {"name": "accuracy", "value": 0.9, "ci_low": 0.895, "ci_high": 0.905},
        {"name": "f1_score", "value": 0.5, "ci_low": 0.1, "ci_high": 0.9},
    ]
    claims = [
        {"metric_name": "accuracy", "stated_value": 0.9, "statement": "Accuracy was 0.9."},
        {"metric_name": "f1_score", "stated_value": 0.5, "statement": "F1 was 0.5."},
    ]

    report = run_summarizer(metrics, config=CONFIG, generate_fn=_fake_generate(claims))

    assert len(report["accepted_claims"]) == 1
    assert report["accepted_claims"][0]["metric_name"] == "accuracy"
    assert len(report["flagged_claims"]) == 1
    assert report["flagged_claims"][0]["metric_name"] == "f1_score"
