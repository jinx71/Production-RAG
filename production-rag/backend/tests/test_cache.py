import fakeredis

from app.cache import QueryCache, _make_key


def _cache() -> QueryCache:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    return QueryCache("redis://unused", ttl_seconds=60, enabled=True, client=client)


def test_set_then_get_roundtrip():
    cache = _cache()
    assert cache.available
    payload = {"answer": "hello", "timings_ms": {"total": 1.0}}
    cache.set("question?", 3, "model-x", payload)
    assert cache.get("question?", 3, "model-x") == payload


def test_get_miss_returns_none():
    cache = _cache()
    assert cache.get("never stored", 3, "model-x") is None


def test_key_is_normalized():
    assert _make_key("  Hello ", 3, "m") == _make_key("hello", 3, "m")
    assert _make_key("hello", 3, "m") != _make_key("hello", 5, "m")


def test_stats_track_hits_and_misses():
    cache = _cache()
    cache.set("q", 3, "m", {"answer": "a"})
    cache.get("q", 3, "m")  # hit
    cache.get("other", 3, "m")  # miss
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_disabled_cache_is_noop():
    cache = QueryCache("redis://unused", ttl_seconds=60, enabled=False)
    assert not cache.available
    cache.set("q", 3, "m", {"answer": "a"})
    assert cache.get("q", 3, "m") is None
