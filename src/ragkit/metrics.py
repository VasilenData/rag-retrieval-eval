"""Retrieval metrics. `ranked` is a list of document ids, best first; `gold` is the one correct id."""
from __future__ import annotations


def _cutoff(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def recall_at_k(ranked: list[int], gold: int, k: int) -> float:
    _cutoff(k)
    return 1.0 if gold in ranked[:k] else 0.0


def reciprocal_rank(ranked: list[int], gold: int, k: int = 10) -> float:
    _cutoff(k)
    for i, d in enumerate(ranked[:k], start=1):
        if d == gold:
            return 1.0 / i
    return 0.0


def summarise(runs: list[tuple[list[int], int]]) -> dict[str, float]:
    n = len(runs)
    if not n:
        raise ValueError("at least one evaluated query is required")
    return {
        "recall@1": sum(recall_at_k(r, g, 1) for r, g in runs) / n,
        "recall@5": sum(recall_at_k(r, g, 5) for r, g in runs) / n,
        "recall@10": sum(recall_at_k(r, g, 10) for r, g in runs) / n,
        "mrr@10": sum(reciprocal_rank(r, g) for r, g in runs) / n,
    }


def bootstrap_ci(values: list[float], iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval of the mean, so a 1-point difference is not oversold."""
    import random
    import math
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("bootstrap values must be nonempty and finite")
    _cutoff(iters)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]
