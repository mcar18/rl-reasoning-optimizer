"""Answer parsing and correctness scoring. Expects FINAL: <answer> in LLM output."""

from __future__ import annotations

import re


def parse_final_answer(llm_output: str) -> str | None:
    """Extract answer after 'FINAL:' (case-insensitive). Returns None if not found."""
    if not llm_output or not isinstance(llm_output, str):
        return None
    # Match "FINAL:" or "FINAL :" then capture rest of line or until next newline
    m = re.search(r"FINAL\s*:\s*(.+?)(?:\n|$)", llm_output, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def normalize_answer(s: str) -> str:
    """Normalize for comparison: strip, lowercase, collapse whitespace, remove punctuation for numbers."""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Remove trailing punctuation that might follow a number
    s = re.sub(r"[,.]$", "", s)
    return s


def compute_correctness(llm_output: str, ground_truth: str) -> bool:
    """Compare LLM output to ground truth. Uses numeric match when applicable."""
    pred = parse_final_answer(llm_output)
    if pred is None:
        return False
    pred_n = normalize_answer(pred)
    gt_n = normalize_answer(ground_truth)
    if pred_n == gt_n:
        return True
    # Try numeric comparison
    try:
        pv = float(pred_n.replace(",", ""))
        gv = float(gt_n.replace(",", ""))
        return abs(pv - gv) < 1e-6
    except ValueError:
        pass
    return False
