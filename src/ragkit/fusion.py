"""Reciprocal Rank Fusion: combine rankings without having to calibrate their scores."""
from __future__ import annotations


def rrf(rankings: list[list[int]], k: int = 60, top: int = 50) -> list[int]:
    """score(d) = sum over rankings of 1 / (k + rank). `k` damps the influence of top ranks."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise ValueError("top must be a positive integer")
    score: dict[int, float] = {}
    for ranking in rankings:
        # One ranker gets one vote per document, even if its results repeat an ID.
        for rank, doc in enumerate(dict.fromkeys(ranking), start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][:top]
