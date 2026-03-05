"""Reward penalties: token cost, formatting, and combined reward."""

from __future__ import annotations

from .scoring import compute_correctness


def token_penalty(tokens_used: int, scale: float = 1e-4) -> float:
    """Per-token penalty. Return negative value to subtract from reward."""
    return -scale * tokens_used


def format_penalty(llm_output: str, has_final: bool | None = None) -> float:
    """Penalty when FINAL: is missing. Optional pre-computed has_final."""
    if has_final is not None:
        return 0.0 if has_final else -0.1
    from .scoring import parse_final_answer
    return 0.0 if parse_final_answer(llm_output) is not None else -0.1


def compute_reward(
    llm_output: str,
    ground_truth: str,
    tokens_used: int,
    token_penalty_scale: float = 1e-4,
    use_format_penalty: bool = True,
) -> tuple[float, bool]:
    """
    Compute reward: +1 if correct, 0 if incorrect, minus token and optional format penalty.
    Returns (reward, correct).
    """
    correct = compute_correctness(llm_output, ground_truth)
    reward = 1.0 if correct else 0.0
    reward += token_penalty(tokens_used, token_penalty_scale)
    if use_format_penalty:
        reward += format_penalty(llm_output)
    return reward, correct
