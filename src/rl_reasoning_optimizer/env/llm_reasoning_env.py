"""MDP environment: state = question + features, action = strategy choice, reward = correctness - token penalty."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..backends.base_backend import LLMBackend
from ..prompts.prompt_library import list_strategies, render_prompt
from ..reward.penalties import compute_reward


@dataclass
class EnvState:
    """State: question id, text, and simple features."""

    question_id: str
    question: str
    length: int
    num_count: int
    previous_result: Any = None  # optional: last (reward, correct)


@dataclass
class EnvStep:
    """Result of one step: next state (same question, episode ends), reward, done, info."""

    state: EnvState
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


def _extract_features(question: str) -> tuple[int, int]:
    """Simple features: character length, count of numbers (digit sequences)."""
    length = len(question)
    numbers = re.findall(r"\d+", question)
    num_count = len(numbers)
    return length, num_count


class LLMReasoningEnv:
    """
    One question per episode. State = question + features. Action = strategy index.
    Transition: call LLM with chosen prompt; reward = correctness - token_penalty.
    """

    def __init__(
        self,
        backend: LLMBackend,
        questions: list[dict[str, Any]],
        token_penalty_scale: float = 1e-4,
        max_tokens: int = 1024,
    ) -> None:
        self.backend = backend
        self.questions = questions
        self.token_penalty_scale = token_penalty_scale
        self.max_tokens = max_tokens
        self.strategies = list_strategies()
        self.n_actions = len(self.strategies)

    def _get_question(self, index: int) -> dict[str, Any]:
        return self.questions[index]

    def reset(self, question_index: int) -> EnvState:
        """Reset to a specific question. Returns initial state."""
        q = self._get_question(question_index)
        qid = q.get("id", str(question_index))
        question = q.get("question", "")
        length, num_count = _extract_features(question)
        return EnvState(
            question_id=qid,
            question=question,
            length=length,
            num_count=num_count,
            previous_result=None,
        )

    def step(
        self,
        state: EnvState,
        action: int,
    ) -> EnvStep:
        """
        Execute one step: select strategy by index, call LLM, compute reward.
        Episode ends after one attempt (done=True).
        """
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action {action}; n_actions={self.n_actions}")
        strategy = self.strategies[action]
        prompt = render_prompt(strategy, state.question)
        response = self.backend.complete(prompt, max_tokens=self.max_tokens, temperature=0.0)
        # Find ground truth for this question
        q_index = next(
            (i for i, q in enumerate(self.questions) if q.get("id") == state.question_id),
            -1,
        )
        ground_truth = self.questions[q_index].get("answer", "") if q_index >= 0 else ""
        reward, correct = compute_reward(
            response.text,
            ground_truth,
            response.tokens_used,
            token_penalty_scale=self.token_penalty_scale,
            use_format_penalty=True,
        )
        info = {
            "strategy": strategy,
            "llm_output": response.text,
            "tokens_used": response.tokens_used,
            "correct": correct,
            "ground_truth": ground_truth,
        }
        next_state = EnvState(
            question_id=state.question_id,
            question=state.question,
            length=state.length,
            num_count=state.num_count,
            previous_result=(reward, correct),
        )
        return EnvStep(state=next_state, reward=reward, done=True, info=info)
