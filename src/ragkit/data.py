"""Corpus and labelled questions from the SQuAD v1.1 dev set (CC BY-SA 4.0).

Each question is answered by one paragraph, which gives us ground truth for retrieval:
the correct paragraph for a question is the one it was written about.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def load(path: str = "data/dev-v1.1.json") -> dict:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build(raw: dict) -> tuple[list[str], list[tuple[str, int]]]:
    """Returns (paragraphs, [(question, gold_paragraph_id)]). Paragraph id = list index."""
    paragraphs, questions = [], []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            pid = len(paragraphs)
            paragraphs.append(para["context"])
            for qa in para["qas"]:
                questions.append((qa["question"], pid))
    return paragraphs, questions


def paragraph_articles(raw: dict) -> list[int]:
    """Article cluster for each paragraph id, in exactly the same order as `build`."""
    return [article_id for article_id, article in enumerate(raw["data"]) for _ in article["paragraphs"]]


def sample(questions: list[tuple[str, int]], n: int, seed: int = 7) -> list[tuple[str, int]]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1 or not questions:
        raise ValueError("sampling requires a positive n and a nonempty question set")
    if n >= len(questions):
        return questions
    return random.Random(seed).sample(questions, n)
