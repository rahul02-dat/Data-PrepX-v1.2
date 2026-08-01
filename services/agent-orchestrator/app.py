"""agent-orchestrator entrypoint.

Phase 0 scope only: prove the service boots and is health-checkable. The
LangGraph state machine (compute_stats -> retrieve_grounding_facts ->
draft_claim -> verify_claim_against_stats -> score_confidence -> emit_or_flag)
and the Ollama client are Phase 7 work (planner Phase 7; CLAUDE.md §5.4).

This service must never receive raw dataframes, only pre-computed,
gate-approved statistics from ml-engine-py -- that boundary is what makes the
bounded-reasoning guarantee enforceable at the architecture level. Do not add
a raw-data code path here even temporarily.
"""

from fastapi import FastAPI

app = FastAPI(title="dataprepx-agent-orchestrator", version="0.0.1")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "agent-orchestrator"}
