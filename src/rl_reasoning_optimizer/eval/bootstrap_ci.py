"""Bootstrap confidence intervals for evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_ci(
    results: list[dict[str, Any]],
    metric_fn: Any,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """
    Compute bootstrap CI for a metric.
    metric_fn: function that takes list of result dicts and returns a scalar.
    Returns (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = metric_fn(results)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = [results[i] for i in idx]
        samples.append(metric_fn(resampled))
    alpha = 1 - confidence
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return point, low, high
