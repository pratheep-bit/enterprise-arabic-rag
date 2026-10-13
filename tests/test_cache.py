import time
from app.utils.cache import QueryCache


def test_query_cache_set_and_get():
    cache = QueryCache(max_size=10, ttl_seconds=60)
    cache.set("key1", "val1")
    assert cache.get("key1") == "val1"
    assert cache.get("nonexistent") is None
    assert cache.size() == 1


def test_query_cache_lru_eviction():
    cache = QueryCache(max_size=2, ttl_seconds=60)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    # Access k1 so k2 becomes LRU
    assert cache.get("k1") == "v1"
    cache.set("k3", "v3")

    assert cache.get("k2") is None
    assert cache.get("k1") == "v1"
    assert cache.get("k3") == "v3"


def test_query_cache_ttl_expiration():
    cache = QueryCache(max_size=5, ttl_seconds=1)
    cache.set("short_lived", "data")
    assert cache.get("short_lived") == "data"
    time.sleep(1.1)
    assert cache.get("short_lived") is None


def test_query_cache_clear_and_stats():
    cache = QueryCache(max_size=5, ttl_seconds=60)
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_ratio"] == 0.5

    cache.clear()
    assert cache.size() == 0
