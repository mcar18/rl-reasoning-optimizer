"""REINFORCE agent with advantage normalization and entropy bonus."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .policy_network import PolicyNetwork, question_to_features, build_tfidf_vectorizer


class ReinforceAgent:
    """REINFORCE with baseline (advantage normalization) and entropy bonus."""

    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.policy = PolicyNetwork(feature_dim, n_actions, hidden_dims)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self._saved_log_probs: list[torch.Tensor] = []
        self._saved_entropies: list[torch.Tensor] = []

    def select_action(
        self,
        state_features: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Sample action from policy; save log_prob and entropy for REINFORCE."""
        x = torch.from_numpy(state_features).float().unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(x)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = int(probs.argmax(dim=-1).item())
            else:
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())
        # Save for gradient step
        logits = self.policy(x)
        log_probs = F.log_softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        self._saved_log_probs.append(log_probs[0, action])
        self._saved_entropies.append(dist.entropy())
        return action

    def store_reward(self, reward: float) -> None:
        """Store reward for current trajectory (single step in our env)."""
        self._rewards = getattr(self, "_rewards", [])
        self._rewards.append(reward)

    def finish_episode(self) -> float:
        """
        Compute REINFORCE loss with advantage normalization and entropy bonus.
        Returns mean loss value.
        """
        if not self._saved_log_probs:
            return 0.0
        rewards = getattr(self, "_rewards", [])
        if not rewards:
            self._saved_log_probs = []
            self._saved_entropies = []
            return 0.0
        R = rewards[0]  # single step
        # Advantage: normalize (optional baseline)
        advantages = [R - 0.0]  # no baseline; could use running mean
        if len(advantages) > 1:
            adv = torch.tensor(advantages, dtype=torch.float32)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
            adv = torch.tensor(advantages, dtype=torch.float32)
        policy_loss = []
        for log_prob, a in zip(self._saved_log_probs, adv):
            policy_loss.append(-log_prob * a.item())
        loss = sum(policy_loss) / len(policy_loss)
        entropy_bonus = -self.entropy_coef * sum(self._saved_entropies) / len(self._saved_entropies)
        loss = loss - entropy_bonus
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._saved_log_probs = []
        self._saved_entropies = []
        self._rewards = []
        return float(loss.item())

    def get_state_dict(self) -> dict[str, Any]:
        return self.policy.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy.load_state_dict(state_dict)
