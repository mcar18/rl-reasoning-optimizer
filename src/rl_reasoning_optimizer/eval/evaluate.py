"""Evaluation metrics: accuracy, avg tokens, reward, cost-adjusted accuracy."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def compute_metrics(
    results: list[dict[str, Any]],
    token_penalty_scale: float = 1e-4,
) -> dict[str, float]:
    """
    results: list of dicts with keys: correct, reward, tokens_used.
    Returns accuracy, avg_tokens, mean_reward, cost_adjusted_accuracy.
    """
    if not results:
        return {"accuracy": 0.0, "avg_tokens": 0.0, "mean_reward": 0.0, "cost_adjusted_accuracy": 0.0}
    n = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    accuracy = correct / n
    avg_tokens = sum(r.get("tokens_used", 0) for r in results) / n
    mean_reward = sum(r.get("reward", 0) for r in results) / n
    # Cost-adjusted: accuracy minus token cost (as fraction of 1)
    cost = token_penalty_scale * avg_tokens
    cost_adjusted_accuracy = max(0.0, accuracy - cost)
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "mean_reward": mean_reward,
        "cost_adjusted_accuracy": cost_adjusted_accuracy,
    }


def evaluate_agent(
    env: Any,
    get_features: Callable[[Any], np.ndarray],
    select_action: Callable[[np.ndarray], int],
    question_indices: list[int],
    deterministic: bool = True,
) -> list[dict[str, Any]]:
    """Run agent on given question indices; return list of result dicts."""
    results = []
    for idx in question_indices:
        state = env.reset(idx)
        features = get_features(state)
        action = select_action(features)
        step = env.step(state, action)
        info = step.info
        results.append({
            "correct": info.get("correct", False),
            "reward": step.reward,
            "tokens_used": info.get("tokens_used", 0),
            "strategy": info.get("strategy", ""),
        })
    return results
