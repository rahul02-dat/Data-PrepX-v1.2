from __future__ import annotations

import json
import re

from graphs.schemas import DraftedClaim, GroundingFact

SYSTEM_INSTRUCTIONS = """You are a data-analysis summarizer. You will be given a list of \
GROUNDING FACTS -- statistics that have already been computed and verified. Your only job is \
to draft short, plain-language claims about them.

Rules:
- Use ONLY the numbers given in the grounding facts. Do not compute, estimate, or invent any \
number not explicitly present in a grounding fact.
- Every claim must reference exactly one metric_name from the grounding facts and restate its \
value as stated_value.
- Do not hedge, editorialize, or add confidence language (e.g. "likely", "strongly") -- \
confidence is scored separately, deterministically, by the calling system.
- Output ONLY a JSON array, no prose before or after, no markdown code fences. Each element:
  {"metric_name": "<one of the given metric names>", "stated_value": <number>, "statement": \
"<one sentence>"}
"""


# Build the full prompt sent to Ollama for one draft_claim call
def build_prompt(facts: list[GroundingFact]) -> str:
    facts_block = "\n".join(f"- {f.text}" for f in facts)
    return f"{SYSTEM_INSTRUCTIONS}\n\nGROUNDING FACTS:\n{facts_block}\n\nJSON array:"


# Strip markdown code fences a model may wrap its JSON output in, despite instructions
def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped


class ClaimParseError(ValueError):
    pass


# Parse a raw LLM response into DraftedClaim objects. Any malformed entry raises ClaimParseError
# rather than silently guessing -- an ungrounded claim slipping through is worse than a failed
# draft.
def parse_drafted_claims(raw_response: str) -> list[DraftedClaim]:
    cleaned = _strip_code_fences(raw_response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ClaimParseError(f"LLM response was not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ClaimParseError("LLM response JSON must be a top-level array of claims")

    claims: list[DraftedClaim] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ClaimParseError(f"claim at index {i} is not a JSON object")
        missing = {"metric_name", "stated_value", "statement"} - item.keys()
        if missing:
            raise ClaimParseError(f"claim at index {i} is missing required field(s): {missing}")
        try:
            stated_value = float(item["stated_value"])
        except (TypeError, ValueError) as exc:
            raise ClaimParseError(f"claim at index {i} has a non-numeric stated_value") from exc
        claims.append(
            DraftedClaim(
                metric_name=str(item["metric_name"]),
                stated_value=stated_value,
                statement=str(item["statement"]),
            )
        )
    return claims
