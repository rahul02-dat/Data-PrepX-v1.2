from fastapi import FastAPI

app = FastAPI(title="dataprepx-ml-engine", version="0.0.1")


# Health check endpoint
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "ml-engine-py"}
