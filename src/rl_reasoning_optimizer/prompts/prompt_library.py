"""Reasoning strategy prompt templates (Jinja2). Strategy id -> template string."""

from __future__ import annotations

from jinja2 import Template

STRATEGY_NAMES = [
    "direct_answer",
    "chain_of_thought",
    "decompose_problem",
    "scratchpad_reasoning",
    "verify_answer",
    "explain_then_answer",
    "minimal_tokens",
    "self_consistency",
]

PROMPT_TEMPLATES = {
    "direct_answer": Template(
        "Answer the following question with a single final answer.\n\n"
        "Question: {{ question }}\n\n"
        "Provide your answer in this exact format: FINAL: <answer>"
    ),
    "chain_of_thought": Template(
        "Solve the following step by step. Show your reasoning, then give the final answer.\n\n"
        "Question: {{ question }}\n\n"
        "Think step by step, then end with: FINAL: <answer>"
    ),
    "decompose_problem": Template(
        "Break the problem into smaller sub-problems. Solve each, then combine.\n\n"
        "Question: {{ question }}\n\n"
        "Sub-problems and solutions:\n\n"
        "Finally, give your answer as: FINAL: <answer>"
    ),
    "scratchpad_reasoning": Template(
        "Use the space below as a scratchpad. Work through the problem, then state the answer clearly.\n\n"
        "Question: {{ question }}\n\n"
        "Scratchpad:\n\n"
        "Answer: FINAL: <answer>"
    ),
    "verify_answer": Template(
        "First solve the problem. Then verify your answer is correct. If wrong, correct it.\n\n"
        "Question: {{ question }}\n\n"
        "Solution and verification:\n\n"
        "Final answer: FINAL: <answer>"
    ),
    "explain_then_answer": Template(
        "Explain what the question is asking in one sentence. Then solve it.\n\n"
        "Question: {{ question }}\n\n"
        "Explanation and solution:\n\n"
        "FINAL: <answer>"
    ),
    "minimal_tokens": Template(
        "Be concise. Answer only with the final value.\n\n"
        "{{ question }}\n\n"
        "FINAL: <answer>"
    ),
    "self_consistency": Template(
        "Solve the problem. Then briefly double-check your reasoning.\n\n"
        "Question: {{ question }}\n\n"
        "Solution:\n\n"
        "Double-check:\n\n"
        "FINAL: <answer>"
    ),
}


def get_template(strategy: str) -> Template | None:
    """Return the Jinja2 template for a strategy, or None if unknown."""
    return PROMPT_TEMPLATES.get(strategy)


def render_prompt(strategy: str, question: str, **kwargs: str) -> str:
    """Render the prompt for a strategy with the given question."""
    tpl = get_template(strategy)
    if tpl is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    return tpl.render(question=question, **kwargs)


def list_strategies() -> list[str]:
    """Return ordered list of strategy names."""
    return list(STRATEGY_NAMES)
