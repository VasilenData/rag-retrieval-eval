from ragkit.data import build, sample
from ragkit.fusion import rrf
from ragkit.metrics import bootstrap_ci, reciprocal_rank, recall_at_k, summarise
from ragkit.retrievers import BM25, tokenize


def test_recall_and_mrr():
    assert recall_at_k([3, 1, 2], 1, 1) == 0.0
    assert recall_at_k([3, 1, 2], 1, 2) == 1.0
    assert reciprocal_rank([3, 1, 2], 1) == 0.5
    assert reciprocal_rank([3, 2], 1) == 0.0


def test_summarise():
    m = summarise([([1, 2], 1), ([2, 1], 1)])
    assert m["recall@1"] == 0.5 and m["recall@5"] == 1.0 and m["mrr@10"] == 0.75


def test_rrf_prefers_documents_both_rankers_like():
    # doc 2 is in both lists (2nd and 1st); docs 1 and 3 appear in only one list
    assert rrf([[1, 2, 9], [2, 3, 8]])[0] == 2


def test_rrf_rewards_a_top_rank_over_two_middling_ones():
    # doc 1 is 1st and 3rd, doc 3 is 3rd and 1st, doc 2 is 2nd and 2nd.
    # Docs 1 and 3 tie (ordered by id) and both beat doc 2, because a top rank counts for more.
    assert rrf([[1, 2, 3], [3, 2, 1]]) == [1, 3, 2]


def test_rrf_is_deterministic_on_ties():
    assert rrf([[1, 2], [2, 1]]) == [1, 2]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the capital of France?") == ["capital", "france"]


def test_bm25_finds_keyword_match():
    docs = ["cats purr softly", "the river flows to the sea", "stock markets fell sharply"]
    assert BM25(docs).search("sea river", k=1) == [1]


def test_build_and_sample_are_deterministic():
    raw = {"data": [{"paragraphs": [{"context": "p0", "qas": [{"question": "q0"}, {"question": "q1"}]},
                                    {"context": "p1", "qas": [{"question": "q2"}]}]}]}
    paras, qs = build(raw)
    assert paras == ["p0", "p1"] and qs == [("q0", 0), ("q1", 0), ("q2", 1)]
    assert sample(qs, 2, seed=1) == sample(qs, 2, seed=1)


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50)
    assert lo < 0.5 < hi
