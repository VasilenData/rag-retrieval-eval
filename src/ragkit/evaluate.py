"""Run all retrieval systems on the same labelled questions and write metrics, latency and failures."""
from __future__ import annotations

import argparse
import json
import os
import time

from . import data
from .fusion import rrf
from .metrics import bootstrap_ci, recall_at_k, summarise
from .retrievers import BM25, Dense, Reranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of questions to evaluate")
    ap.add_argument("--rerank-top", type=int, default=20, help="candidates passed to the reranker")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    paragraphs, questions = data.build(data.load())
    qs = data.sample(questions, a.n)
    print(f"{len(paragraphs)} paragraphs, {len(questions)} questions, evaluating {len(qs)}", flush=True)

    t = time.perf_counter()
    bm25 = BM25(paragraphs)
    t_bm25_index = time.perf_counter() - t
    t = time.perf_counter()
    dense = Dense(paragraphs)
    t_dense_index = time.perf_counter() - t
    rer = Reranker(paragraphs)

    texts = [q for q, _ in qs]
    t = time.perf_counter()
    qvecs = dense.embed_queries(texts)
    t_embed = time.perf_counter() - t

    runs = {"bm25": [], "dense": [], "hybrid (BM25 + dense, RRF)": [], f"hybrid + rerank top {a.rerank_top}": []}
    lat = {k: 0.0 for k in runs}
    failures = []
    for (q, gold), v in zip(qs, qvecs):
        t = time.perf_counter(); r_b = bm25.search(q); lat["bm25"] += time.perf_counter() - t
        t = time.perf_counter(); r_d = dense.search_vec(v); lat["dense"] += time.perf_counter() - t
        h = "hybrid (BM25 + dense, RRF)"
        t = time.perf_counter(); r_h = rrf([r_b, r_d]); lat[h] += time.perf_counter() - t
        rk = f"hybrid + rerank top {a.rerank_top}"
        t = time.perf_counter(); r_r = rer.rerank(q, r_h[: a.rerank_top]); lat[rk] += time.perf_counter() - t
        for name, r in ((("bm25"), r_b), ("dense", r_d), (h, r_h), (rk, r_r)):
            runs[name].append((r, gold))
        if recall_at_k(r_r, gold, 1) == 0 and len(failures) < 8:
            failures.append({"question": q, "gold": gold, "top1": r_r[0], "gold_in_candidates": gold in r_h[: a.rerank_top]})

    n = len(qs)
    res = {}
    for name, rr in runs.items():
        m = summarise(rr)
        lo, hi = bootstrap_ci([recall_at_k(r, g, 1) for r, g in rr])
        res[name] = {**{k: round(v, 4) for k, v in m.items()}, "recall@1_ci95": [round(lo, 4), round(hi, 4)]}
    # latency: retrieval time per query; dense and hybrid-rerank add the one-off query embedding time
    emb_ms = 1000 * t_embed / n
    ms = {k: 1000 * v / n for k, v in lat.items()}
    ms["dense"] += emb_ms
    ms["hybrid (BM25 + dense, RRF)"] += ms["bm25"] + ms["dense"]
    ms[f"hybrid + rerank top {a.rerank_top}"] += ms["hybrid (BM25 + dense, RRF)"]
    for k in res:
        res[k]["latency_ms_per_query"] = round(ms[k], 1)
    out = {"questions": n, "paragraphs": len(paragraphs), "index_seconds": {"bm25": round(t_bm25_index, 1), "dense": round(t_dense_index, 1)}, "systems": res, "sample_failures": failures}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
