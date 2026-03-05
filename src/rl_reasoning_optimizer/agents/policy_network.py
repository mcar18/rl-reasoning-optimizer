"""Policy network: MLP on TF-IDF features of the question."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer


class PolicyNetwork(nn.Module):
    """MLP that maps question features (TF-IDF) to action logits."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        dims = [input_dim] + hidden_dims + [n_actions]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim). Returns logits (batch, n_actions)."""
        return self.mlp(x)


def build_tfidf_vectorizer(
    questions: list[str],
    max_features: int = 500,
    **kwargs: Any,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on question texts."""
    vec = TfidfVectorizer(max_features=max_features, **kwargs)
    vec.fit(questions)
    return vec


def question_to_features(
    question: str,
    vectorizer: TfidfVectorizer,
    length: int = 0,
    num_count: int = 0,
    max_len: int = 2000,
) -> np.ndarray:
    """Convert question to feature vector: TF-IDF + optional length/num_count (normalized)."""
    tfidf = vectorizer.transform([question]).toarray().flatten()
    extra = np.array([min(length, max_len) / max_len, min(num_count, 50) / 50.0], dtype=np.float32)
    return np.concatenate([tfidf, extra])
