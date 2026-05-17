"""
SwingAdvisorBot — Module 1: Data Layer
cache.py — TTL-based in-memory cache

Every API call costs either money (NewsAPI daily quota, Claude tokens)
or rate limit capacity (Kite 3 req/sec). Caching prevents wasteful
re-fetching of data that hasn't changed.

Design decisions:
  - Pure in-memory dict — no Redis dependency for a single-process bot.
  - TTL per cache key — different data types have different staleness
    tolerances (stocks: 3min, news: 15min, VIX: 5min, FRED: 60min).
  - Thread-safe via asyncio (single event loop, no threading concerns).
  - Explicit invalidation via clear() and delete() for token refresh flows.
  - Cache stats for pipeline health monitoring — the advisor wants
    to know if the cache is working or if we're hammering APIs.

TTL values are set in DataFetchConfig (config.py):
  cache_ttl_stocks  = 180   (3 minutes)
  cache_ttl_news    = 900   (15 minutes)
  cache_ttl_events  = 3600  (60 minutes)
  cache_ttl_vix     = 300   (5 minutes)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CacheEntry:
    """A single cached value with its expiration timestamp.

    Attributes:
        value: The cached data (any type — StockData list, NewsItem list, VIX float, etc.)
        expires_at: Unix timestamp when this entry becomes stale.
                    Compared against time.monotonic() for clock-skew safety.
        created_at: Unix timestamp when this entry was stored.
                    Used for cache age reporting in pipeline health checks.
    """

    value: Any
    expires_at: float
    created_at: float


@dataclass
class CacheStats:
    """Cache performance metrics for pipeline health monitoring.

    The advisor's pipeline health check (Step 7) reports these stats
    to verify the cache is reducing API load as expected.
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage.

        A healthy cache should maintain > 60% hit rate during market hours.
        Below 30% suggests TTLs are too short or the cache is being
        cleared too aggressively.
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return round((self.hits / total) * 100, 2)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats for health report inclusion."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate_pct": self.hit_rate,
        }


class TTLCache:
    """In-memory cache with per-key TTL expiration.

    Used by all fetchers to avoid redundant API calls within
    the staleness tolerance for each data type.

    Usage pattern in fetchers:
        cache = TTLCache()

        # Check cache first
        cached = cache.get("stocks:HDFCBANK")
        if cached is not None:
            return cached  # Fresh data, no API call needed

        # Cache miss — fetch from API
        data = await kite.ltp("NSE:HDFCBANK")

        # Store with appropriate TTL
        cache.set("stocks:HDFCBANK", data, ttl=config.cache_ttl_stocks)

    Key naming convention:
        "stocks:{ticker}"       → Individual stock data
        "stocks:batch"          → Batch stock fetch result
        "news:headlines"        → NewsAPI headlines
        "vix:current"           → India VIX value
        "sectors:all"           → Sector performance data
        "economic:{series_id}"  → FRED series data
        "instruments:map"       → Kite instrument token map

    Thread safety:
        This cache is designed for single-process asyncio usage.
        All fetchers run on the same event loop — no threading
        concerns. If multi-process scaling is needed, replace
        with Redis (but that's a Module 9 concern, not Module 1).
    """

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._stats: CacheStats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if it exists and hasn't expired.

        Returns None on cache miss or expired entry.
        Expired entries are lazily evicted on access — no background
        cleanup thread needed for our scale.

        Args:
            key: Cache key following the naming convention above.

        Returns:
            The cached value if fresh, None if missing or expired.
        """
        entry = self._store.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        if time.monotonic() > entry.expires_at:
            del self._store[key]
            self._stats.misses += 1
            self._stats.evictions += 1
            return None

        self._stats.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Store a value with a TTL in seconds.

        Overwrites any existing entry for the same key.
        TTL is converted to an absolute expiration timestamp
        using time.monotonic() for clock-skew immunity.

        Args:
            key: Cache key following the naming convention.
            value: Data to cache (any serializable type).
            ttl: Time-to-live in seconds. After this duration,
                 get() returns None and the entry is evicted.
        """
        now = time.monotonic()
        self._store[key] = CacheEntry(
            value=value,
            expires_at=now + ttl,
            created_at=now,
        )
        self._stats.sets += 1

    def delete(self, key: str) -> bool:
        """Explicitly remove a cache entry.

        Used when data is known to be invalid — e.g., after a Kite
        token refresh, the old instrument map must be invalidated.

        Args:
            key: Cache key to remove.

        Returns:
            True if the key existed and was removed, False if not found.
        """
        if key in self._store:
            del self._store[key]
            self._stats.evictions += 1
            return True
        return False

    def clear(self) -> None:
        """Remove all cached entries.

        Used during:
          - Kite re-authentication (all stock data may be stale)
          - Pipeline restart
          - Manual cache flush via MCP health endpoint
        """
        count = len(self._store)
        self._store.clear()
        self._stats.evictions += count

    def has(self, key: str) -> bool:
        """Check if a key exists and is still fresh (not expired).

        Does NOT count as a hit or miss in stats — this is a
        peek operation used by the pipeline to decide fetch strategy
        without triggering cache eviction side effects.

        Args:
            key: Cache key to check.

        Returns:
            True if the key exists and hasn't expired, False otherwise.
        """
        entry = self._store.get(key)
        if entry is None:
            return False
        if time.monotonic() > entry.expires_at:
            return False
        return True

    def get_age(self, key: str) -> Optional[float]:
        """Get the age of a cached entry in seconds.

        Returns None if the key doesn't exist or has expired.
        Used by the pipeline health check to report data freshness
        to the advisor — e.g., "Stock data is 45 seconds old."

        Args:
            key: Cache key to check age for.

        Returns:
            Age in seconds (float), or None if not cached/expired.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        now = time.monotonic()
        if now > entry.expires_at:
            return None
        return round(now - entry.created_at, 2)

    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Normally, expired entries are lazily evicted on access via get().
        This method does a full sweep — useful before a pipeline health
        check to get accurate cache size metrics.

        Returns:
            Number of expired entries removed.
        """
        now = time.monotonic()
        expired_keys = [
            key
            for key, entry in self._store.items()
            if now > entry.expires_at
        ]
        for key in expired_keys:
            del self._store[key]
        self._stats.evictions += len(expired_keys)
        return len(expired_keys)

    @property
    def size(self) -> int:
        """Number of entries currently in cache (including potentially expired)."""
        return len(self._store)

    @property
    def stats(self) -> CacheStats:
        """Access cache performance stats for health reporting."""
        return self._stats

    def get_stats_report(self) -> dict[str, Any]:
        """Generate a cache health report for pipeline monitoring.

        Returns a dict suitable for inclusion in PipelineHealthReport.
        The advisor pipeline uses this to verify the cache is working
        as expected and not causing stale data or excessive API calls.
        """
        self.cleanup_expired()
        return {
            "cache_size": self.size,
            "stats": self._stats.to_dict(),
            "keys": list(self._store.keys()),
        }


# ─────────────────────────────────────────────────────────────
# Module-level singleton — all fetchers share one cache instance
# Import as: from module1_data_layer.cache import cache
# ─────────────────────────────────────────────────────────────

cache = TTLCache()
