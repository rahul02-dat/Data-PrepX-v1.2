from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from config import SummarizerConfig, load_summarizer_config
from graphs.schemas import ComputedStat, Metric
from ollama_client import GenerateFn, OllamaClient
from prompts.summarizer_prompt import ClaimParseError, build_prompt, parse_drafted_claims

from .nodes import (
    compute_stats,
    emit_or_flag,
    retrieve_grounding_facts,
    score_confidence,
    verify_claim_against_stats,
)


class SummarizerState(TypedDict, total=False):

    metrics: list[dict]
    computed_stats: list[dict]
    grounding_facts: list[dict]
    drafted_claims: list[dict]
    draft_error: str | None
    verified_claims: list[dict]
    scored_claims: list[dict]
    report: dict


# Build the six-node LangGraph state machine. `generate_fn` defaults to a real OllamaClient but
# is injectable so tests never need a live Ollama server -- draft_claim is the only node that
# calls it, every other node is pure and tested directly against nodes.py.
def build_summarizer_graph(
    *, config: SummarizerConfig | None = None, generate_fn: GenerateFn | None = None
):
    config = config or load_summarizer_config()
    generate_fn = generate_fn or OllamaClient(config).generate

    # Reconstruct ComputedStat objects from the state dict rather than closing over mutable
    # state. The compiled graph object may be reused across concurrent invocations (e.g. one
    # FastAPI app instance serving multiple requests); a shared mutable closure would let one
    # request's stats leak into another's verification step. State is the only channel LangGraph
    # itself guarantees is per-invocation.
    def _stats_by_name(state: SummarizerState) -> dict[str, ComputedStat]:
        return {s["name"]: ComputedStat(**s) for s in state["computed_stats"]}

    def _compute_stats_node(state: SummarizerState) -> dict[str, Any]:
        metrics = [Metric.from_dict(m) for m in state["metrics"]]
        stats = compute_stats(metrics)
        return {"computed_stats": [s.to_dict() for s in stats]}

    def _retrieve_grounding_facts_node(state: SummarizerState) -> dict[str, Any]:
        stats_by_name = _stats_by_name(state)
        facts = retrieve_grounding_facts(list(stats_by_name.values()))
        return {"grounding_facts": [f.to_dict() for f in facts]}

    def _draft_claim_node(state: SummarizerState) -> dict[str, Any]:
        from graphs.schemas import GroundingFact

        facts = [GroundingFact(**f) for f in state["grounding_facts"]]
        prompt = build_prompt(facts)
        raw_response = generate_fn(prompt)
        try:
            claims = parse_drafted_claims(raw_response)
        except ClaimParseError as exc:
            # A malformed draft is not a crash: emit_or_flag will simply have nothing to
            # verify, and the caller sees why via draft_error rather than a 500.
            return {"drafted_claims": [], "draft_error": str(exc)}
        return {"drafted_claims": [c.to_dict() for c in claims], "draft_error": None}

    def _verify_claim_node(state: SummarizerState) -> dict[str, Any]:
        from graphs.schemas import DraftedClaim

        stats_by_name = _stats_by_name(state)
        claims = [DraftedClaim.from_dict(c) for c in state.get("drafted_claims", [])]
        verified = verify_claim_against_stats(claims, stats_by_name, config=config)
        return {"verified_claims": [v.to_dict() for v in verified]}

    def _score_confidence_node(state: SummarizerState) -> dict[str, Any]:
        from graphs.schemas import VerifiedClaim

        stats_by_name = _stats_by_name(state)
        verified = [VerifiedClaim(**v) for v in state["verified_claims"]]
        scored = score_confidence(verified, stats_by_name, config=config)
        return {"scored_claims": [s.to_dict() for s in scored]}

    def _emit_or_flag_node(state: SummarizerState) -> dict[str, Any]:
        from graphs.schemas import ScoredClaim, VerifiedClaim

        scored = [ScoredClaim(**s) for s in state["scored_claims"]]
        verified = [VerifiedClaim(**v) for v in state["verified_claims"]]
        report = emit_or_flag(scored, verified)
        report_dict = report.to_dict()
        if state.get("draft_error"):
            report_dict["draft_error"] = state["draft_error"]
        return {"report": report_dict}

    graph = StateGraph(SummarizerState)
    graph.add_node("compute_stats", _compute_stats_node)
    graph.add_node("retrieve_grounding_facts", _retrieve_grounding_facts_node)
    graph.add_node("draft_claim", _draft_claim_node)
    graph.add_node("verify_claim_against_stats", _verify_claim_node)
    graph.add_node("score_confidence", _score_confidence_node)
    graph.add_node("emit_or_flag", _emit_or_flag_node)

    graph.set_entry_point("compute_stats")
    graph.add_edge("compute_stats", "retrieve_grounding_facts")
    graph.add_edge("retrieve_grounding_facts", "draft_claim")
    graph.add_edge("draft_claim", "verify_claim_against_stats")
    graph.add_edge("verify_claim_against_stats", "score_confidence")
    graph.add_edge("score_confidence", "emit_or_flag")
    graph.add_edge("emit_or_flag", END)

    return graph.compile()


# Run the full summarizer graph end to end on a list of Metric dicts, returning the report dict
def run_summarizer(
    metrics: list[dict],
    *,
    config: SummarizerConfig | None = None,
    generate_fn: GenerateFn | None = None,
) -> dict:
    config = config or load_summarizer_config()
    app = build_summarizer_graph(config=config, generate_fn=generate_fn)
    result = app.invoke({"metrics": metrics})
    return result["report"]
