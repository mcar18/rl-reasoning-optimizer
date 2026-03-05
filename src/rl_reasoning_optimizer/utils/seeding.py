"""Deterministic seeding for reproducibility."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int) -> None:
    """Set random seeds for random, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def get_rng(seed: int | None = None) -> tuple[random.Random, np.random.Generator]:
    """Return a seeded random.Random and numpy Generator."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    return rng, np_rng
