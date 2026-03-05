"""Tests for reward scoring (correctness, parse_final_answer, normalize)."""

import pytest
from src.rl_reasoning_optimizer.reward.scoring import (
    parse_final_answer,
    normalize_answer,
    compute_correctness,
)


def test_parse_final_answer_basic():
    assert parse_final_answer("Some text. FINAL: 42") == "42"
    assert parse_final_answer("FINAL: 100") == "100"
    assert parse_final_answer("FINAL: hello") == "hello"


def test_parse_final_answer_case_insensitive():
    assert parse_final_answer("final: 7") == "7"
    assert parse_final_answer("FINAL : 8") == "8"


def test_parse_final_answer_missing():
    assert parse_final_answer("No final here") is None
    assert parse_final_answer("") is None


def test_normalize_answer():
    assert normalize_answer("  42  ") == "42"
    assert normalize_answer("  Hello  World  ") == "hello world"
    assert normalize_answer("42.") == "42"
    assert normalize_answer("1,000") == "1,000"


def test_compute_correctness_exact():
    assert compute_correctness("FINAL: 42", "42") is True
    assert compute_correctness("FINAL: 100", "100") is True
    assert compute_correctness("FINAL: 42", "43") is False


def test_compute_correctness_numeric():
    assert compute_correctness("FINAL: 42.0", "42") is True
    assert compute_correctness("FINAL: 42", "42.0") is True
    assert compute_correctness("FINAL: 42", "41") is False


def test_compute_correctness_no_final():
    assert compute_correctness("I think the answer is 42", "42") is False
