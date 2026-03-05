"""API-based LLM backend (OpenAI / Anthropic compatible)."""

from __future__ import annotations

import os
from typing import Any

from .base_backend import LLMBackend, LLMResponse


def _get_client():
    """Lazy import to avoid requiring openai/anthropic when using local only."""
    try:
        from openai import OpenAI
        return ("openai", OpenAI)
    except ImportError:
        pass
    return (None, None)


class APIBackend(LLMBackend):
    """OpenAI API (or compatible) backend. Uses env OPENAI_API_KEY or API_KEY."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""
        self._base_url = base_url
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        provider, client_cls = _get_client()
        if client_cls is None:
            raise ImportError("Install openai: pip install openai")
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = client_cls(**kwargs)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choice = resp.choices[0] if resp.choices else None
        text = choice.message.content if choice else ""
        tokens_usage = getattr(resp, "usage", None)
        total = getattr(tokens_usage, "total_tokens", len(text.split())) if tokens_usage else len(text.split())
        return LLMResponse(
            text=text,
            tokens_used=total,
            raw=resp,
            model=self._model,
        )
