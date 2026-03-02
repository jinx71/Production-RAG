"""Text embedders.

Two backends:
- FastEmbedEmbedder: production default. Lightweight ONNX models (no torch).
- HashingEmbedder: zero-dependency feature hashing. Deterministic and offline,
  used for tests / CI and as a no-network fallback.
"""

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from app.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder:
    """Feature-hashing embedder: maps tokens into a fixed-dim, L2-normalized vector.

    Lexically overlapping texts land close in cosine space, which is enough for
    plumbing tests and an offline fallback (it is not semantically aware).
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = digest % self.dimension
            sign = 1.0 if (digest >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (no torch dependency)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # local import: optional heavy dep

        self._model = TextEmbedding(model_name=model_name)
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dimension = int(len(probe))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._model.embed([text])))]


def get_embedder(settings: Settings) -> Embedder:
    backend = settings.embedding_backend.lower()
    if backend == "hashing":
        return HashingEmbedder(dimension=settings.embedding_dim)
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name=settings.embedding_model)
    raise ValueError(f"Unknown embedding backend: {settings.embedding_backend!r}")
