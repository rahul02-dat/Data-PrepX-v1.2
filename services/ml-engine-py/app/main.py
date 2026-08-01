"""ml-engine-py entrypoint.

Phase 0 scope only: prove the service boots and can be health-checked by the
gateway. Pipeline logic (validation gates, imputation, RL, MAML, Optuna,
stacking) is out of scope until Phases 2-6 per docs/01_IMPLEMENTATION_PLANNER.md.
"""

from fastapi import FastAPI

app = FastAPI(title="dataprepx-ml-engine", version="0.0.1")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "ml-engine-py"}
