from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graphs.summarizer_graph import run_summarizer

app = FastAPI(title="dataprepx-agent-orchestrator", version="0.0.1")


# Service health check endpoint
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "agent-orchestrator"}


class MetricIn(BaseModel):
    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None


class SummarizeRequest(BaseModel):
    # Pre-computed, gate-approved metrics only (CLAUDE.md §2) -- never raw data.
    metrics: list[MetricIn]


# Run the Phase 7 bounded RAG summarizer graph over already-computed metrics
@app.post("/v1/summarize")
def summarize(req: SummarizeRequest) -> dict:
    if not req.metrics:
        raise HTTPException(status_code=400, detail="metrics must be a non-empty list")
    metrics_payload = [m.model_dump() for m in req.metrics]
    return run_summarizer(metrics_payload)
