"""The retrieval systems being compared. Every retriever maps a query to a ranked list of paragraph ids."""
from __future__ import annotations

import re

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an the of in on at to for and or is are was were be been by with as from that this it its what which who whom when where how why did does do has have had".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    """Keyword search. Strong when the question shares rare words with the answer paragraph."""

    def __init__(self, docs: list[str]):
        from rank_bm25 import BM25Okapi
        self.index = BM25Okapi([tokenize(d) for d in docs])

    def search(self, query: str, k: int = 50) -> list[int]:
        scores = self.index.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


class Dense:
    """Meaning-based search: sentence embeddings stored in Qdrant (local mode, no server needed)."""

    def __init__(self, docs: list[str], model: str = "BAAI/bge-small-en-v1.5", batch: int = 32):
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient, models
        self.model = TextEmbedding(model_name=model)
        vecs = np.array(list(self.model.embed(docs, batch_size=batch)))
        self.client = QdrantClient(":memory:")
        self.client.create_collection("docs", vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE))
        self.client.upload_collection("docs", vectors=vecs, ids=list(range(len(docs))))

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return np.array(list(self.model.query_embed(queries)))

    def search_vec(self, vec: np.ndarray, k: int = 50) -> list[int]:
        hits = self.client.query_points("docs", query=vec.tolist(), limit=k).points
        return [int(h.id) for h in hits]


class Reranker:
    """A cross-encoder reads the question and a paragraph together, which is slower but more precise."""

    def __init__(self, docs: list[str], model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.docs = docs
        self.model = TextCrossEncoder(model_name=model)

    def rerank(self, query: str, candidates: list[int], top: int = 10) -> list[int]:
        scores = list(self.model.rerank(query, [self.docs[c] for c in candidates]))
        order = np.argsort(-np.array(scores), kind="stable")
        return [candidates[i] for i in order][:top]
