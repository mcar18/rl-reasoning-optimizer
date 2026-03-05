"""Local LLM backend via Ollama. Stub when Ollama not available."""

from __future__ import annotations

import json
from typing import Any

from .base_backend import LLMBackend, LLMResponse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class LocalBackend(LLMBackend):
    """Ollama local backend. Uses stub when Ollama is not running or requests missing."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        use_stub_if_unavailable: bool = True,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._use_stub = use_stub_if_unavailable
        self._available: bool | None = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not HAS_REQUESTS:
            self._available = False
            return False
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

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
        if self._use_stub and not self._check_available():
            return self._stub_response(prompt, max_tokens)
        try:
            return self._ollama_complete(prompt, max_tokens, temperature)
        except Exception:
            if self._use_stub:
                return self._stub_response(prompt, max_tokens)
            raise

    def _stub_response(self, prompt: str, max_tokens: int) -> LLMResponse:
        """Return a deterministic stub when Ollama is not available (for local runs without API)."""
        # Stub: echo a placeholder so tests and training can run
        stub_text = "FINAL: 42"
        tokens_estimate = min(len(prompt.split()) + len(stub_text.split()), max_tokens)
        return LLMResponse(
            text=stub_text,
            tokens_used=tokens_estimate,
            raw=None,
            model=self._model,
        )

    def _ollama_complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        assert HAS_REQUESTS
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        r = requests.post(url, json=payload, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        text = data.get("response", "")
        # Ollama may return eval_count as tokens used
        tokens_used = data.get("eval_count") or len(text.split())
        return LLMResponse(
            text=text,
            tokens_used=tokens_used,
            raw=data,
            model=self._model,
        )
