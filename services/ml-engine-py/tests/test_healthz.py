from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz_returns_ok():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ml-engine-py"}


def test_healthz_rejects_post():
    response = client.post("/healthz")
    assert response.status_code == 405
