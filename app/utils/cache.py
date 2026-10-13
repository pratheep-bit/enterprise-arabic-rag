"""
cache.py — In-Memory Thread-Safe LRU/TTL Query Cache

Caches retrieval results and LLM question-answering outputs to reduce
latency and API token costs on recurring queries.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class QueryCache:
    """Thread-safe LRU Cache with TTL expiration for RAG queries."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key if present and not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            timestamp, value = self._cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with current timestamp, evicting oldest if capacity reached."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), value)
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Return the current number of cached items."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = round(self._hits / total, 4) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
            }
