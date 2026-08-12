from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SummarizerConfig:
    """Thresholds for the Phase 7 bounded RAG summarizer (CLAUDE.md §5.4).

    Confidence is derived from a claim's relative CI width (ci_width / |value|), not from the
    LLM's own wording. Smaller relative CI width -> higher confidence. Claims with no CI
    available cannot be scored and are always flagged rather than guessed at.
    """

    # Below this relative CI width, a claim is "high" confidence.
    high_confidence_max_relative_ci_width: float = 0.15
    # Above this relative CI width, a claim is "inconclusive" and must be flagged, never
    # asserted as fact (CLAUDE.md §5.4: "near-threshold findings are explicitly flagged").
    inconclusive_min_relative_ci_width: float = 0.40
    # Relative tolerance allowed between the LLM's stated numeric value and the actual computed
    # statistic before verify_claim_against_stats rejects the claim outright.
    verification_relative_tolerance: float = 0.02
    # Ollama model used for the draft_claim node. See docs/adr/0007-ollama-model-choice.md.
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not (
            0.0
            < self.high_confidence_max_relative_ci_width
            < self.inconclusive_min_relative_ci_width
        ):
            raise ValueError(
                "high_confidence_max_relative_ci_width must be positive and less than "
                "inconclusive_min_relative_ci_width"
            )
        if not (0.0 < self.verification_relative_tolerance < 1.0):
            raise ValueError("verification_relative_tolerance must be in (0, 1)")


def load_summarizer_config() -> SummarizerConfig:
    return SummarizerConfig(
        ollama_model=os.environ.get("SUMMARIZER_OLLAMA_MODEL", SummarizerConfig.ollama_model),
        ollama_url=os.environ.get("OLLAMA_URL", SummarizerConfig.ollama_url),
    )
