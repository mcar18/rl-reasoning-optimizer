"""LLM backends: base, local (Ollama), API (OpenAI/Anthropic compatible)."""

from .base_backend import LLMBackend, LLMResponse
from .local_backend import LocalBackend
from .api_backend import APIBackend

__all__ = ["LLMBackend", "LLMResponse", "LocalBackend", "APIBackend"]
