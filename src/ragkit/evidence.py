"""Reusable question-level scores and paired article-cluster comparisons; no model imports."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import tempfile
from pathlib import Path

from .metrics import cluster_bootstrap_ci, recall_at_k, reciprocal_rank

METRICS = ("recall@1", "recall@5", "recall@10", "mrr@10")


def question_records(questions, runs, paragraph_articles):
    """Attach every system's result to one shared question, preventing unpaired comparisons."""
    if not questions or not runs or any(len(rows) != len(questions) for rows in runs.values()):
        raise ValueError("every system must evaluate the same nonempty question set")
    records = []
    for index, (question, gold) in enumerate(questions):
        if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(paragraph_articles):
            raise ValueError("gold paragraph must identify a corpus paragraph")
        scores = {}
        for name, rows in runs.items():
            ranked, actual_gold = rows[index]
            if actual_gold != gold:
                raise ValueError("system results are not aligned to the same gold paragraphs")
            if (any(isinstance(doc, bool) or not isinstance(doc, int) or not 0 <= doc < len(paragraph_articles)
                    for doc in ranked) or len(ranked) != len(set(ranked))):
                raise ValueError("rankings must contain distinct valid corpus paragraph ids")
            scores[name] = {**{f"recall@{k}": recall_at_k(ranked, gold, k) for k in (1, 5, 10)},
                            "mrr@10": reciprocal_rank(ranked, gold)}
        records.append({"question_id": index, "question": question, "gold_paragraph_id": gold,
                        "article_id": paragraph_articles[gold], "scores": scores,
                        "ranked_paragraphs": {name: rows[index][0][:10] for name, rows in runs.items()}})
    return records


def compare_records(records, iters: int = 2000, seed: int = 7):
    """Candidate-minus-baseline differences; all pairwise intervals are exploratory/unadjusted."""
    if not isinstance(records, list) or not records:
        raise ValueError("question-level results are required; aggregate reports cannot be reconstructed")
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise ValueError("bootstrap iterations must be a positive integer")
    for row in records:
        if (not isinstance(row, dict) or not {"question_id", "article_id", "scores"} <= set(row)
                or isinstance(row["question_id"], bool) or not isinstance(row["question_id"], int)
                or row["question_id"] < 0 or not isinstance(row["scores"], dict)):
            raise ValueError("invalid question-level record schema")
    systems = list(records[0]["scores"])
    if len(systems) < 2 or any(not isinstance(name, str) or not name for name in systems):
        raise ValueError("at least two systems are required for paired comparison")
    ids = set()
    for row in records:
        if row["question_id"] in ids:
            raise ValueError("question ids must be unique")
        ids.add(row["question_id"])
        if set(row["scores"]) != set(systems):
            raise ValueError("every question must contain every compared system")
        if isinstance(row["article_id"], bool) or not isinstance(row["article_id"], int) or row["article_id"] < 0:
            raise ValueError("article ids must be integers")
        for scores in row["scores"].values():
            if (not isinstance(scores, dict) or set(scores) != set(METRICS)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                           or not 0 <= v <= 1 for v in scores.values())):
                raise ValueError("all retrieval metrics must be present, finite and in [0, 1]")
    clusters = [row["article_id"] for row in records]
    comparisons = []
    for baseline, candidate in itertools.combinations(systems, 2):
        metrics = {}
        for metric in METRICS:
            differences = [r["scores"][candidate][metric] - r["scores"][baseline][metric] for r in records]
            interval = cluster_bootstrap_ci(differences, clusters, iters, seed) if len(set(clusters)) >= 2 else None
            metrics[metric] = {"delta": math.fsum(differences) / len(records), "ci95": interval}
        comparisons.append({"baseline": baseline, "candidate": candidate, "metrics": metrics})
    return comparisons


def write_report(path: Path, report: dict) -> None:
    """Leave the previous complete evidence file intact if serialization/publication fails."""
    encoded = json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".evaluation-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Recompute paired comparisons from saved question-level evidence.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        parser.error("evaluation report must be a JSON object")
    try:
        comparisons = compare_records(report.get("per_question", []), args.iterations, args.seed)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"resampling_unit": "article", "iterations": args.iterations,
                      "seed": args.seed, "comparisons": comparisons}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
