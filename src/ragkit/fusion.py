"""Reciprocal Rank Fusion: combine rankings without having to calibrate their scores."""
from __future__ import annotations


def rrf(rankings: list[list[int]], k: int = 60, top: int = 50) -> list[int]:
    """score(d) = sum over rankings of 1 / (k + rank). `k` damps the influence of top ranks."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][:top]
