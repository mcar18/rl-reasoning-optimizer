"""Base LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    tokens_used: int
    raw: Any = None
    model: str = ""


class LLMBackend(ABC):
    """Abstract base class for LLM backends (local or API)."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send prompt to LLM and return response with token count."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
