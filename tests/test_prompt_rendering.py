"""Tests for prompt template rendering (Jinja2)."""

import pytest
from src.rl_reasoning_optimizer.prompts.prompt_library import (
    STRATEGY_NAMES,
    get_template,
    render_prompt,
    list_strategies,
)


def test_list_strategies():
    strategies = list_strategies()
    assert len(strategies) == 8
    assert "direct_answer" in strategies
    assert "chain_of_thought" in strategies
    assert "minimal_tokens" in strategies


def test_get_template():
    assert get_template("direct_answer") is not None
    assert get_template("chain_of_thought") is not None
    assert get_template("unknown_strategy") is None


def test_render_prompt_direct_answer():
    out = render_prompt("direct_answer", "What is 2+2?")
    assert "What is 2+2?" in out
    assert "FINAL" in out
    assert "direct" in out.lower() or "answer" in out.lower()


def test_render_prompt_chain_of_thought():
    out = render_prompt("chain_of_thought", "Solve this.")
    assert "Solve this." in out
    assert "FINAL" in out
    assert "step" in out.lower() or "reasoning" in out.lower()


def test_render_prompt_minimal_tokens():
    out = render_prompt("minimal_tokens", "Quick: 3*4?")
    assert "3*4?" in out
    assert "FINAL" in out


def test_render_prompt_unknown_raises():
    with pytest.raises(ValueError):
        render_prompt("not_a_strategy", "question")
