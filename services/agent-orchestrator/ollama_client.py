from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from config import SummarizerConfig


class GenerateFn(Protocol):
    def __call__(self, prompt: str) -> str: ...


@dataclass
class OllamaClient:
    config: SummarizerConfig

    # Send a prompt to Ollama and return the raw text response
    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.config.ollama_url}/api/generate",
            json={"model": self.config.ollama_model, "prompt": prompt, "stream": False},
            timeout=self.config.ollama_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()["response"]
