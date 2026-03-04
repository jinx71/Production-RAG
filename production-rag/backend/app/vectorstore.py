"""Dense vector store backed by an embedded (persistent) Chroma collection."""

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._name = collection_name
        self._collection = self._get_collection()

    def _get_collection(self):
        # cosine space => distance in [0, 2]; similarity = 1 - distance.
        return self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        out: list[dict[str, Any]] = []
        for i, _id in enumerate(ids):
            out.append(
                {
                    "id": _id,
                    "text": docs[i],
                    "metadata": metas[i] or {},
                    "score": 1.0 - float(dists[i]),  # cosine distance -> similarity
                }
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._get_collection()
