# Retrieval evaluation: BM25, vectors, fusion and reranking

[![ci](https://github.com/VasilenData/rag-retrieval-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/VasilenData/rag-retrieval-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The retrieval half of a RAG (retrieval-augmented generation) system, built and **measured** instead of assumed. Four retrieval strategies are compared on the same 2,000 labelled questions, with confidence intervals, latency and examples of failures.

```mermaid
flowchart LR
  Q[Question] --> B[BM25 keyword search]
  Q --> D[Embedding model -> Qdrant vector search]
  B --> F[Reciprocal Rank Fusion]
  D --> F
  F --> R[Cross-encoder reranker<br/>top 20]
  R --> A[Top 10 paragraphs]
```

## Setup

* **Corpus:** 2,067 paragraphs from 48 Wikipedia articles, and **10,570 questions** each written about one paragraph: the [SQuAD v1.1 dev set](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0). The paragraph a question was written about is the ground-truth answer to "which document should be retrieved?".
* **Sample:** 2,000 questions drawn with a fixed random seed.
* **Systems:** BM25 (`rank-bm25`); dense search with `BAAI/bge-small-en-v1.5` embeddings in **Qdrant**; hybrid fusion of the two with Reciprocal Rank Fusion; hybrid plus a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) on the top 20. Embeddings and reranking run through `fastembed` (ONNX, CPU only).
* **Metrics:** recall@k (is the right paragraph in the top k?) and MRR@10, with a bootstrap 95% confidence interval on recall@1.

## Results

| System | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Latency per query |
|---|---:|---:|---:|---:|---:|
| BM25 | 76.3% (74.5 to 78.1) | 91.4% | 94.4% | 0.828 | 2 ms |
| Dense vectors (Qdrant) | 69.4% (67.4 to 71.4) | 89.5% | 94.0% | 0.780 | 3 ms |
| Hybrid (BM25 + dense, RRF) | 77.0% (75.3 to 78.9) | 94.0% | 97.3% | 0.844 | 5 ms |
| **Hybrid + rerank top 20** | **86.3%** (84.8 to 87.8) | **97.2%** | **98.4%** | **0.911** | 99 ms |

Brackets are the 95% confidence interval for recall@1. Full numbers: [results/metrics.json](results/metrics.json).

## What this shows

1. **Keyword search beat vector search at the top spot.** BM25 got the right paragraph first 76% of the time against 69% for embeddings. That is not what most tutorials suggest, and it makes sense here: the questions were written while looking at the paragraph, so they share rare words with it. Measure on your own data before choosing a vector database.
2. **Fusion helps most at 10 results, not at 1.** Hybrid raised recall@10 from 94.4% to 97.3%; its observed recall@1 gain was only 0.7 percentage points. The historical aggregate report cannot support a paired uncertainty estimate for that difference, so I would not claim an established top-1 improvement.
3. **The reranker produced the largest observed gain on this sample**: recall@1 from 77% to 86%, with non-overlapping individual intervals; a paired or article-clustered comparison is not reported. The cost is latency, about 99 ms per question against 5 ms, because a cross-encoder reads each question together with 20 candidate paragraphs.
4. **Where it still fails:** the report lists sample failures. In the 8 failures sampled in `metrics.json`, 7 had the correct paragraph among the 20 candidates, so the remaining errors are mostly about ordering, not about finding candidates. Many are near-duplicates, for example two paragraphs from the same article about the same topic.

## Limits, stated plainly

* **Retrieval only.** There is no language model in this repository, so answer quality and hallucination are not evaluated. The retriever is the part you would plug into one.
* **A small, clean corpus.** 2,067 paragraphs is far easier than millions of messy documents, and the questions were written with the paragraph in view, which favours keyword matching. Treat the absolute numbers as specific to this dataset.
* **Query embedding was done in one batch**, so the dense and hybrid latencies amortise it. A single live query would cost more for that step.
* Latencies are from a cloud VM on CPU and will differ on other hardware.
* The index is rebuilt in memory on every run (about 26 seconds for the embeddings). A production system would persist it.

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m ragkit.evaluate --n 2000 --out results/local-eval  # downloads data/models; preserves historical results
pytest -q                                              # unit tests for metrics, fusion, BM25 and data handling
```

`src/ragkit/` holds the pieces: `retrievers.py` (the three retrievers), `fusion.py` (RRF), `metrics.py` (recall, MRR, bootstrap), `evaluate.py` (the experiment).

## Input and ranking contracts
Metrics reject nonpositive cutoffs and empty evaluations rather than interpreting negative Python slices as valid top-k results. Fusion gives each ranker at most one vote per document even if an upstream result repeats an ID. The CLI validates query count and rerank depth before loading data or models.

The historical intervals above resample individual evaluated questions; within-article correlation is not accounted for. The existing 2,000-question report was not rerun by these lightweight regression tests, and its aggregate scores cannot be converted into paired/clustered results without the original question-level predictions.

## Paired article-cluster evidence for future runs

New evaluations emit schema-version-2 reports containing each sampled question, its gold paragraph and article id, and every system's top-10 paragraph ids plus recall@1/5/10 and MRR@10. Systems are compared on the same question records. Confidence intervals resample entire articles with replacement, preserving all sampled questions in a selected article; the estimand remains a question-weighted mean, not an equally weighted average of article means. Pairwise intervals use candidate-minus-baseline differences inside the same resample. A sample with fewer than two articles reports `null` intervals instead of pretending it has independent clusters.

The report records sampling/bootstrap seeds, iteration count and sampled article count. It is written as one complete atomic snapshot so a failed publication cannot truncate the previous evidence. Recompute comparisons from a new report without loading models:

```bash
PYTHONPATH=src python -m ragkit.evidence results/local-eval/metrics.json --iterations 2000 --seed 7
```

These percentile intervals are exploratory and unadjusted for multiple comparisons. They describe variability under article resampling of the sampled SQuAD questions; they do not establish generalization to another corpus, production accuracy, or a calibrated winner-selection procedure. Historical metrics/latency values above remain unchanged. The new protocol is covered by paired-system, within-article duplication, row-order and publication-failure tests plus an end-to-end evaluation using fake retrievers; no new full-model benchmark is claimed.
