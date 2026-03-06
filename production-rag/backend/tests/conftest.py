"""Shared test fixtures. Everything here runs offline: hashing embedder, fakeredis,
and a fake LLM, so no model downloads or API calls are needed."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from app.bm25_index import BM25Index  # noqa: E402
from app.cache import QueryCache  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embeddings import HashingEmbedder  # noqa: E402
from app.ingest import ingest_text  # noqa: E402
from app.pipeline import RagPipeline  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402
from app.vectorstore import VectorStore  # noqa: E402


class FakeLLM:
    """Routes by system prompt so one fake serves generation + all judges."""

    def __init__(self, answer: str = "Three consecutive runs are required [1].", faithful: bool = True):
        self.answer = answer
        self.faithful = faithful
        self.systems: list[str] = []

    async def complete(self, *, system, user, model=None, max_tokens=1024, temperature=0.0):
        self.systems.append(system)
        if "fact-checker" in system:
            grounded = "true" if self.faithful else "false"
            score = "1.0" if self.faithful else "0.2"
            return f'{{"grounded": {grounded}, "faithfulness_score": {score}, "unsupported_claims": []}}'
        if "generate exactly 3" in system:
            return '["question one", "question two", "question three"]'
        if "relevant and useful" in system:
            return '{"relevant": true}'
        return self.answer


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_backend="hashing",
        embedding_dim=256,
        chroma_persist_dir=str(tmp_path / "chroma"),
        chroma_collection="test",
        cache_enabled=False,
        relevance_threshold=0.02,
        faithfulness_threshold=0.6,
        top_k=3,
        candidate_k=10,
    )


@pytest.fixture
def embedder(settings: Settings) -> HashingEmbedder:
    return HashingEmbedder(settings.embedding_dim)


@pytest.fixture
def stores(settings: Settings):
    vectorstore = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    bm25 = BM25Index(str(Path(settings.chroma_persist_dir) / "bm25.json"))
    return vectorstore, bm25


@pytest.fixture
def pipeline(settings: Settings, embedder: HashingEmbedder, stores):
    vectorstore, bm25 = stores
    retriever = HybridRetriever(
        vectorstore, bm25, embedder, candidate_k=settings.candidate_k, rrf_k=settings.rrf_k
    )
    cache = QueryCache("redis://localhost:6379/0", ttl_seconds=60, enabled=False)
    return RagPipeline(retriever=retriever, llm=FakeLLM(), cache=cache, settings=settings)


@pytest.fixture
def seed_corpus(settings: Settings, embedder: HashingEmbedder, stores):
    vectorstore, bm25 = stores
    docs = {
        "cleaning.md": (
            "A minimum of three consecutive successful cleaning runs is required to "
            "consider a cleaning process validated. Swab sampling targets worst-case "
            "locations such as welds and gaskets."
        ),
        "calibration.md": (
            "The standard used to calibrate must have an accuracy at least four times "
            "better than the tolerance of the instrument. This is the 4:1 test accuracy "
            "ratio."
        ),
        "integrity.md": (
            "ALCOA stands for Attributable, Legible, Contemporaneous, Original and "
            "Accurate. Audit trails must record the previous value and the identity of "
            "the person making the change."
        ),
    }
    for name, text in docs.items():
        ingest_text(text, name, embedder, vectorstore, bm25, settings)
    return vectorstore, bm25
