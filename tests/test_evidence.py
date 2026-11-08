"""Paired, clustered uncertainty and complete evidence publication without downloading models."""
import copy
import json

import pytest

from ragkit.data import build, paragraph_articles
from ragkit.evidence import METRICS, compare_records, question_records, write_report
from ragkit.metrics import cluster_bootstrap_ci


def records(left, right, clusters):
    return [{"question_id": i, "article_id": cluster,
             "scores": {"baseline": dict.fromkeys(METRICS, a), "candidate": dict.fromkeys(METRICS, b)}}
            for i, (a, b, cluster) in enumerate(zip(left, right, clusters))]


def test_pairing_cancels_query_difficulty_for_identical_systems():
    rows = records([0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])
    for result in compare_records(rows)[0]["metrics"].values():
        assert result == {"delta": 0, "ci95": (0, 0)}


def test_repeating_questions_within_articles_does_not_create_independent_evidence():
    original = cluster_bootstrap_ci([0, 1], [0, 1], iters=500)
    repeated = cluster_bootstrap_ci([0] * 100 + [1] * 100, [0] * 100 + [1] * 100, iters=500)
    assert original == repeated == (0, 1)


def test_point_estimate_is_question_weighted_and_sign_reverses_with_system_order():
    rows = records([0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1])
    result = compare_records(rows)[0]["metrics"]["recall@1"]
    assert result["delta"] == 0.75
    reversed_rows = copy.deepcopy(rows)
    for row in reversed_rows:
        row["scores"] = dict(reversed(list(row["scores"].items())))
    reverse = compare_records(reversed_rows)[0]["metrics"]["recall@1"]
    assert reverse["delta"] == -result["delta"]
    assert reverse["ci95"] == pytest.approx((-result["ci95"][1], -result["ci95"][0]))
    assert compare_records(list(reversed(rows))) == compare_records(rows)


def test_one_article_reports_no_interval_instead_of_false_precision():
    rows = records([0, 0], [1, 1], [0, 0])
    assert compare_records(rows)[0]["metrics"]["recall@1"] == {"delta": 1, "ci95": None}
    with pytest.raises(ValueError, match="two"):
        cluster_bootstrap_ci([0, 1], [0, 0])
    with pytest.raises(ValueError, match="iterations"):
        compare_records(rows, iters=0)


@pytest.mark.parametrize("change", [
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0]["scores"].pop("candidate"),
    lambda rows: rows[0]["scores"]["candidate"].update({"mrr@10": float("nan")}),
    lambda rows: rows[0].update({"article_id": "unknown"}),
    lambda rows: rows[0].update({"question_id": True}),
])
def test_incomplete_or_invalid_evidence_cannot_be_compared(change):
    rows = records([0, 1], [1, 0], [0, 1])
    change(rows)
    with pytest.raises(ValueError):
        compare_records(rows)


def test_question_results_must_align_before_becoming_evidence():
    with pytest.raises(ValueError, match="aligned"):
        question_records([("q", 0)], {"system": [([1], 1)]}, [0, 1])
    with pytest.raises(ValueError, match="same nonempty"):
        question_records([("q", 0)], {"system": []}, [0])
    for ranking in ([0, 0], [-1], [2]):
        with pytest.raises(ValueError, match="distinct valid"):
            question_records([("q", 0)], {"system": [(ranking, 0)]}, [0, 1])


def test_publication_failure_preserves_the_previous_complete_report(tmp_path, monkeypatch):
    from ragkit import evidence
    path = tmp_path / "metrics.json"
    write_report(path, {"version": "old"})
    before = path.read_bytes()
    def fail(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(evidence.os, "replace", fail)
    with pytest.raises(OSError):
        write_report(path, {"version": "new"})
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp"))


def test_full_evaluator_records_reusable_article_evidence_with_fake_retrievers(tmp_path, monkeypatch):
    from ragkit import evaluate
    raw = {"data": [{"paragraphs": [{"context": f"article {i}",
                                     "qas": [{"question": f"q{i}a"}, {"question": f"q{i}b"}]}]}
                    for i in range(2)]}
    assert paragraph_articles(raw) == [0, 1]
    assert len(build(raw)[1]) == 4
    class FakeRetriever:
        def __init__(self, paragraphs):
            pass
        def search(self, query):
            return [0, 1]
        def embed_queries(self, texts):
            return [[1] for _ in texts]
        def search_vec(self, vector):
            return [1, 0]
        def rerank(self, query, candidates):
            return candidates
    monkeypatch.setattr(evaluate.data, "load", lambda: raw)
    for name in ("BM25", "Dense", "Reranker"):
        monkeypatch.setattr(evaluate, name, FakeRetriever)
    monkeypatch.setattr("sys.argv", ["evaluate", "--n", "4", "--out", str(tmp_path)])
    evaluate.main()
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["protocol"]["resampling_unit"] == "article"
    assert report["protocol"]["sampled_articles"] == 2
    assert len(report["per_question"]) == 4 and len(report["comparisons"]) == 6
    assert report["per_question"][0]["ranked_paragraphs"]["bm25"] == [0, 1]
    assert report["comparisons"] == json.loads(json.dumps(compare_records(report["per_question"])))


def test_offline_comparison_rejects_historical_aggregate_only_report(tmp_path, monkeypatch):
    from ragkit.evidence import main
    report = tmp_path / "legacy.json"
    report.write_text('{"systems": {}}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["evidence", str(report)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
