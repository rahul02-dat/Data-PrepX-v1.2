import httpx
import pytest

from config import SummarizerConfig
from ollama_client import OllamaClient


def test_generate_returns_response_text(monkeypatch):
    config = SummarizerConfig(ollama_url="http://fake-ollama:11434")
    client = OllamaClient(config)

    def fake_post(url, json, timeout):
        assert url == "http://fake-ollama:11434/api/generate"
        assert json["model"] == config.ollama_model
        assert json["stream"] is False

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": "[]"}

        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    assert client.generate("some prompt") == "[]"


def test_generate_propagates_http_errors(monkeypatch):
    config = SummarizerConfig(ollama_url="http://fake-ollama:11434")
    client = OllamaClient(config)

    def fake_post(url, json, timeout):
        request = httpx.Request("POST", url)
        response = httpx.Response(status_code=500, request=request)

        class FakeResponse:
            def raise_for_status(self):
                raise httpx.HTTPStatusError("boom", request=request, response=response)

        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        client.generate("some prompt")
