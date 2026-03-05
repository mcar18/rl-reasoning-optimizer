"""Baselines: random prompt, best fixed strategy, epsilon-greedy bandit."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .evaluate import evaluate_agent, compute_metrics


def run_random_baseline(
    env: Any,
    get_features: Any,
    question_indices: list[int],
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Random prompt selection baseline."""
    rng = random.Random(seed)
    n_actions = env.n_actions

    def select_action(_features: np.ndarray) -> int:
        return rng.randint(0, n_actions - 1)

    return evaluate_agent(env, get_features, select_action, question_indices)


def run_best_fixed_baseline(
    env: Any,
    get_features: Any,
    question_indices: list[int],
    strategy_index: int,
) -> list[dict[str, Any]]:
    """Best fixed prompt: always choose the same strategy index."""

    def select_action(_features: np.ndarray) -> int:
        return strategy_index

    return evaluate_agent(env, get_features, select_action, question_indices)


def find_best_fixed_strategy(
    env: Any,
    get_features: Any,
    question_indices: list[int],
    token_penalty_scale: float = 1e-4,
    max_questions: int | None = None,
    verbose: bool = True,
) -> int:
    """Evaluate each strategy on (a subset of) questions; return index with best mean reward."""
    indices = question_indices
    if max_questions is not None and len(indices) > max_questions:
        indices = indices[:max_questions]
    n_actions = env.n_actions
    best_reward = float("-inf")
    best_idx = 0
    for a in range(n_actions):
        if verbose:
            print("  Strategy {}/{} ({} questions)...".format(a + 1, n_actions, len(indices)), flush=True)
        results = run_best_fixed_baseline(env, get_features, indices, a)
        m = compute_metrics(results, token_penalty_scale=token_penalty_scale)
        if m["mean_reward"] > best_reward:
            best_reward = m["mean_reward"]
            best_idx = a
    return best_idx
