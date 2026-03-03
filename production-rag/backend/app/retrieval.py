"""Hybrid retrieval: dense (vector) + sparse (BM25), fused with RRF.

Reciprocal Rank Fusion combines two ranked lists using only their ranks, so it
is robust to the fact that cosine similarity and BM25 scores live on totally
different scales. Score for a doc = sum over lists of 1 / (rrf_k + rank).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None
    rrf_score: float


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    rrf_k: int,
    top_k: int,
) -> list[RetrievedChunk]:
    merged: dict[str, dict[str, Any]] = {}

    def register(items: list[dict[str, Any]], source: str) -> None:
        for rank, item in enumerate(items):
            entry = merged.setdefault(
                item["id"],
                {
                    "id": item["id"],
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "dense_rank": None,
                    "sparse_rank": None,
                    "dense_score": None,
                    "sparse_score": None,
                    "rrf": 0.0,
                },
            )
            entry[f"{source}_rank"] = rank
            entry[f"{source}_score"] = item["score"]
            entry["rrf"] += 1.0 / (rrf_k + rank + 1)
            if not entry["text"]:
                entry["text"] = item.get("text", "")
            if not entry["metadata"]:
                entry["metadata"] = item.get("metadata", {})

    register(dense, "dense")
    register(sparse, "sparse")

    ranked = sorted(
        merged.values(),
        key=lambda e: (e["rrf"], e["dense_score"] or 0.0),
        reverse=True,
    )[:top_k]

    return [
        RetrievedChunk(
            id=e["id"],
            text=e["text"],
            metadata=e["metadata"],
            dense_rank=e["dense_rank"],
            sparse_rank=e["sparse_rank"],
            dense_score=e["dense_score"],
            sparse_score=e["sparse_score"],
            rrf_score=e["rrf"],
        )
        for e in ranked
    ]


class HybridRetriever:
    def __init__(self, vectorstore, bm25, embedder, *, candidate_k: int, rrf_k: int) -> None:
        self._vs = vectorstore
        self._bm25 = bm25
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int) -> tuple[list[RetrievedChunk], float]:
        query_embedding = self._embedder.embed_query(query)
        dense = self._vs.query(query_embedding, self._candidate_k)
        sparse = self._bm25.query(query, self._candidate_k)
        fused = reciprocal_rank_fusion(dense, sparse, self._rrf_k, top_k)
        top_dense_score = dense[0]["score"] if dense else 0.0
        return fused, top_dense_score
