"""End-to-end RAG pipeline: cache -> retrieve -> gate -> generate -> verify."""

import time

from app.config import Settings
from app.generation import REFUSAL, build_context, generate_answer
from app.guardrails import check_faithfulness, relevance_gate
from app.retrieval import HybridRetriever, RetrievedChunk


def _citations(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "index": i + 1,
            "id": c.id,
            "source": c.metadata.get("source", "unknown"),
            "text": c.text,
            "dense_rank": c.dense_rank,
            "sparse_rank": c.sparse_rank,
            "dense_score": round(c.dense_score, 4) if c.dense_score is not None else None,
            "sparse_score": round(c.sparse_score, 4) if c.sparse_score is not None else None,
            "rrf_score": round(c.rrf_score, 5),
        }
        for i, c in enumerate(chunks)
    ]


class RagPipeline:
    def __init__(self, *, retriever: HybridRetriever, llm, cache, settings: Settings) -> None:
        self._retriever = retriever
        self._llm = llm
        self._cache = cache
        self._s = settings

    async def answer(self, query: str, top_k: int | None = None) -> dict:
        top_k = top_k or self._s.top_k
        gen_model = self._s.generation_model
        started = time.perf_counter()

        cached = self._cache.get(query, top_k, gen_model)
        if cached is not None:
            cached["cache_hit"] = True
            cached["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000, 1)
            return cached

        retrieval_started = time.perf_counter()
        chunks, top_dense = self._retriever.retrieve(query, top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        citations = _citations(chunks)

        # Guardrail 1: relevance gate (pre-generation).
        if not relevance_gate(top_dense, bool(chunks), self._s.relevance_threshold):
            return {
                "answer": REFUSAL,
                "cited_indices": [],
                "citations": citations,
                "guardrails": {
                    "relevance_gate_passed": False,
                    "top_dense_score": round(top_dense, 4),
                    "relevance_threshold": self._s.relevance_threshold,
                    "faithfulness": {
                        "checked": False,
                        "grounded": True,
                        "faithfulness_score": 1.0,
                        "unsupported_claims": [],
                        "threshold": self._s.faithfulness_threshold,
                    },
                    "refused": True,
                },
                "cache_hit": False,
                "timings_ms": {
                    "retrieval": round(retrieval_ms, 1),
                    "generation": 0.0,
                    "faithfulness": 0.0,
                    "total": round((time.perf_counter() - started) * 1000, 1),
                },
            }

        generation_started = time.perf_counter()
        answer, cited = await generate_answer(self._llm, query, chunks, model=gen_model)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        # Guardrail 2: faithfulness verification (post-generation).
        faith_started = time.perf_counter()
        context = build_context(chunks)
        faith = await check_faithfulness(
            self._llm, answer, context, model=self._s.guardrail_model
        )
        faith_ms = (time.perf_counter() - faith_started) * 1000
        below_threshold = faith.checked and faith.faithfulness_score < self._s.faithfulness_threshold

        response = {
            "answer": answer,
            "cited_indices": cited,
            "citations": citations,
            "guardrails": {
                "relevance_gate_passed": True,
                "top_dense_score": round(top_dense, 4),
                "relevance_threshold": self._s.relevance_threshold,
                "faithfulness": {
                    "checked": faith.checked,
                    "grounded": faith.grounded and not below_threshold,
                    "faithfulness_score": round(faith.faithfulness_score, 3),
                    "unsupported_claims": faith.unsupported_claims,
                    "threshold": self._s.faithfulness_threshold,
                },
                "refused": False,
            },
            "cache_hit": False,
            "timings_ms": {
                "retrieval": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "faithfulness": round(faith_ms, 1),
                "total": round((time.perf_counter() - started) * 1000, 1),
            },
        }
        self._cache.set(query, top_k, gen_model, response)
        return response
