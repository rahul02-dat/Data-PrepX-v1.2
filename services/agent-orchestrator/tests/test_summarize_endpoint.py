from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def test_summarize_rejects_empty_metrics():
    response = client.post("/v1/summarize", json={"metrics": []})
    assert response.status_code == 400


def test_summarize_end_to_end_with_stubbed_ollama(monkeypatch):
    def fake_run_summarizer(metrics, *, config=None, generate_fn=None):
        return {"accepted_claims": [], "flagged_claims": [], "rejected_claims": []}

    monkeypatch.setattr(app_module, "run_summarizer", fake_run_summarizer)

    response = client.post(
        "/v1/summarize",
        json={"metrics": [{"name": "accuracy", "value": 0.9, "ci_low": 0.89, "ci_high": 0.91}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {"accepted_claims": [], "flagged_claims": [], "rejected_claims": []}
