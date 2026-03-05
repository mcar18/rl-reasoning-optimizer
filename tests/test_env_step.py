"""Tests for environment step (reset, step, reward)."""

import pytest
from src.rl_reasoning_optimizer.backends.base_backend import LLMBackend, LLMResponse
from src.rl_reasoning_optimizer.env import LLMReasoningEnv


class StubLLMBackend(LLMBackend):
    """Test double: always returns FINAL: 4 so env can score."""

    @property
    def model_name(self) -> str:
        return "stub"

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0, **kwargs):
        return LLMResponse(text="FINAL: 4", tokens_used=10, raw=None, model="stub")


@pytest.fixture
def sample_questions():
    return [
        {"id": "t1", "question": "What is 2 + 2?", "answer": "4"},
        {"id": "t2", "question": "What is 3 * 4?", "answer": "12"},
    ]


@pytest.fixture
def backend():
    return StubLLMBackend()


@pytest.fixture
def env(sample_questions, backend):
    return LLMReasoningEnv(
        backend=backend,
        questions=sample_questions,
        token_penalty_scale=1e-4,
        max_tokens=64,
    )


def test_env_reset(env):
    state = env.reset(0)
    assert state.question_id == "t1"
    assert "2 + 2" in state.question
    assert state.length > 0
    assert state.num_count >= 2


def test_env_step_returns_done(env):
    state = env.reset(0)
    step = env.step(state, 0)  # action 0 = first strategy
    assert step.done is True
    assert "tokens_used" in step.info
    assert "correct" in step.info
    assert "strategy" in step.info
    assert hasattr(step, "reward")


def test_env_n_actions(env):
    assert env.n_actions == 8  # number of strategies in prompt_library


def test_env_step_invalid_action(env):
    state = env.reset(0)
    with pytest.raises(ValueError):
        env.step(state, 99)
