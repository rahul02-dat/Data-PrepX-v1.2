import pytest

from graphs.schemas import GroundingFact
from prompts.summarizer_prompt import ClaimParseError, build_prompt, parse_drafted_claims


def test_build_prompt_includes_all_facts():
    facts = [GroundingFact(metric_name="accuracy", text="accuracy = 0.9")]
    prompt = build_prompt(facts)
    assert "accuracy = 0.9" in prompt
    assert "JSON array" in prompt


def test_parse_drafted_claims_valid_json():
    raw = '[{"metric_name": "accuracy", "stated_value": 0.9, "statement": "Accuracy was 0.9."}]'
    claims = parse_drafted_claims(raw)
    assert len(claims) == 1
    assert claims[0].metric_name == "accuracy"
    assert claims[0].stated_value == 0.9


def test_parse_drafted_claims_strips_markdown_fences():
    raw = '```json\n[{"metric_name": "accuracy", "stated_value": 0.9, "statement": "x"}]\n```'
    claims = parse_drafted_claims(raw)
    assert len(claims) == 1


def test_parse_drafted_claims_rejects_non_json():
    with pytest.raises(ClaimParseError):
        parse_drafted_claims("Accuracy is definitely 0.9, trust me.")


def test_parse_drafted_claims_rejects_non_array_top_level():
    with pytest.raises(ClaimParseError):
        parse_drafted_claims('{"metric_name": "accuracy", "stated_value": 0.9, "statement": "x"}')


def test_parse_drafted_claims_rejects_missing_fields():
    with pytest.raises(ClaimParseError):
        parse_drafted_claims('[{"metric_name": "accuracy", "statement": "x"}]')


def test_parse_drafted_claims_rejects_non_numeric_stated_value():
    with pytest.raises(ClaimParseError):
        parse_drafted_claims(
            '[{"metric_name": "accuracy", "stated_value": "high", "statement": "x"}]'
        )


def test_parse_drafted_claims_empty_array_is_valid():
    assert parse_drafted_claims("[]") == []
