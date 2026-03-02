"""Query-response cache backed by Redis.

Caches the full answer payload keyed by (normalized query, top_k, model) so a
repeat question skips retrieval, generation, and guardrails entirely. If Redis is
unreachable the cache disables itself instead of crashing the request path.
"""

import hashlib
import json
from typing import Any

import redis


def _make_key(query: str, top_k: int, model: str) -> str:
    raw = f"{query.strip().lower()}|{top_k}|{model}"
    return "rag:q:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class QueryCache:
    def __init__(self, redis_url: str, *, ttl_seconds: int, enabled: bool, client=None) -> None:
        self._ttl = ttl_seconds
        self._enabled = enabled
        self._available = False
        self._client = None
        if not enabled:
            return
        try:
            self._client = client or redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def get(self, query: str, top_k: int, model: str) -> dict | None:
        if not self._available:
            return None
        try:
            raw = self._client.get(_make_key(query, top_k, model))
            self._client.incr("rag:stats:hits" if raw else "rag:stats:misses")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, query: str, top_k: int, model: str, value: dict[str, Any]) -> None:
        if not self._available:
            return
        try:
            self._client.set(_make_key(query, top_k, model), json.dumps(value), ex=self._ttl)
        except Exception:
            pass

    def stats(self) -> dict[str, Any]:
        if not self._available:
            return {"available": False, "hits": 0, "misses": 0}
        try:
            hits = int(self._client.get("rag:stats:hits") or 0)
            misses = int(self._client.get("rag:stats:misses") or 0)
            total = hits + misses
            return {
                "available": True,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 3) if total else 0.0,
            }
        except Exception:
            return {"available": False, "hits": 0, "misses": 0}
