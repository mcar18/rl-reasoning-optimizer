"""Contextual bandit baseline: epsilon-greedy over strategies."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


class EpsilonGreedyBandit:
    """Epsilon-greedy bandit: with prob epsilon random action, else best average reward per arm."""

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.counts = np.zeros(n_actions)
        self.sum_rewards = np.zeros(n_actions)

    def select_action(self, state_features: np.ndarray, deterministic: bool = False) -> int:
        """Ignore state_features for standard bandit; use epsilon-greedy on arms."""
        if deterministic or self.rng.random() >= self.epsilon:
            # Greedy: best average reward (avoid divide-by-zero for unseen arms)
            means = np.zeros(self.n_actions, dtype=float)
            np.divide(
                self.sum_rewards,
                self.counts,
                out=means,
                where=self.counts > 0,
            )
            best = np.max(means)
            candidates = np.where(means == best)[0]
            return int(self.rng.choice(candidates))
        return self.rng.randint(0, self.n_actions - 1)

    def update(self, action: int, reward: float) -> None:
        """Update counts and sum_rewards for the chosen action."""
        self.counts[action] += 1
        self.sum_rewards[action] += reward
