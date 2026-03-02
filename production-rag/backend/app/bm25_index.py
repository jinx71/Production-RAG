"""Sparse keyword retrieval via BM25 (Okapi).

The tokenized corpus is persisted as JSON; the BM25 index is cheap to rebuild
on load, so we keep only the raw texts/ids on disk.
"""

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, store_path: str) -> None:
        self._store_path = Path(store_path)
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._ids = data.get("ids", [])
            self._texts = data.get("texts", [])
            self._rebuild()

    def _persist(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps({"ids": self._ids, "texts": self._texts}),
            encoding="utf-8",
        )

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi([_tokenize(t) for t in self._texts]) if self._texts else None

    def add(self, ids: list[str], texts: list[str]) -> None:
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._rebuild()
        self._persist()

    def query(self, text: str, top_k: int) -> list[dict[str, Any]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[dict[str, Any]] = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            out.append({"id": self._ids[i], "text": self._texts[i], "score": float(scores[i])})
        return out

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids, self._texts, self._bm25 = [], [], None
        if self._store_path.exists():
            self._store_path.unlink()
