from fastapi import FastAPI

app = FastAPI(title="dataprepx-agent-orchestrator", version="0.0.1")


# Service health check endpoint
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "agent-orchestrator"}
