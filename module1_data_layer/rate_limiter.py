"""
SwingAdvisorBot — Module 1: Data Layer
rate_limiter.py — Async API rate limiting

Every external API has rate limits. Violating them means dropped
requests, temporary bans, or unreliable data — all unacceptable
for a system a senior finance advisor depends on.

Rate limits we enforce:
  Kite Connect: 3 requests/second
    → asyncio.Semaphore(3) with 1-second cooldown window
    → Used by stock_fetcher, vix_fetcher, sector_fetcher
    → If 15 stocks are fetched, they go out in batches of 3

  NewsAPI: 100 requests/day (free tier)
    → Daily counter with hard stop at 95 (5-request safety margin)
    → Cache TTL of 15 minutes prevents wasteful re-fetches
    → Used by news_fetcher only

  FRED API: 120 requests/minute
    → Generous limit, but we still enforce a semaphore(10) 
      for good citizenship and to prevent accidental bursts
    → Cache TTL of 60 minutes makes this rarely hit

Design decisions:
  - asyncio.Semaphore for concurrency control (not threading)
  - Sliding window for Kite (most critical rate limit)
  - Daily counter for NewsAPI (hard quota, no retries)
  - All limiters are module-level singletons, shared by all fetchers
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("swing_advisor.rate_limiter")


@dataclass
class RateLimiterStats:
    """Rate limiter performance metrics for pipeline monitoring.

    These stats help the advisor pipeline diagnose slow data fetches.
    If throttled_count is high, the pipeline is bumping against limits
    and the advisor may be getting delayed data.
    """

    requests_made: int = 0
    requests_throttled: int = 0
    total_wait_seconds: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Serialize for health report inclusion."""
        return {
            "requests_made": self.requests_made,
            "requests_throttled": self.requests_throttled,
            "total_wait_seconds": round(self.total_wait_seconds, 3),
        }


class SlidingWindowLimiter:
    """Sliding window rate limiter for APIs with per-second quotas.

    Used primarily for Kite Connect (3 req/sec). Tracks request
    timestamps in a deque-like list, and waits if the window is full.

    How it works:
      1. Before each request, check how many requests were made
         in the last `window_seconds`.
      2. If count >= max_requests, calculate sleep time until the
         oldest request in the window expires.
      3. Sleep, then proceed.

    This is more precise than a simple semaphore for bursty workloads.
    A semaphore allows max_requests concurrent, but doesn't enforce
    the per-second spread. This limiter does both.

    Example for Kite (3 req/sec, 1 second window):
      t=0.00  req1 → allowed (1 in window)
      t=0.01  req2 → allowed (2 in window)
      t=0.02  req3 → allowed (3 in window)
      t=0.03  req4 → WAIT until t=1.00 (oldest req1 exits window)
      t=1.00  req4 → allowed (window reset)
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 1.0,
        name: str = "default",
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._name = name
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()
        self._stats = RateLimiterStats()

    async def acquire(self) -> None:
        """Acquire permission to make an API request.

        Blocks (via asyncio.sleep) if the sliding window is full.
        Thread-safe via asyncio.Lock — multiple coroutines can
        call acquire() concurrently on the same limiter.

        Logs a professional briefing-style message when throttling
        occurs so the advisor pipeline knows about delays.
        """
        async with self._lock:
            now = time.monotonic()

            # Purge timestamps outside the current window
            self._timestamps = [
                ts
                for ts in self._timestamps
                if now - ts < self._window_seconds
            ]

            if len(self._timestamps) >= self._max_requests:
                # Window is full — calculate wait time
                oldest = self._timestamps[0]
                wait_time = self._window_seconds - (now - oldest)

                if wait_time > 0:
                    self._stats.requests_throttled += 1
                    self._stats.total_wait_seconds += wait_time
                    logger.info(
                        f"[{self._name}] Rate limit reached — "
                        f"{self._max_requests} requests in {self._window_seconds}s window. "
                        f"Throttling for {wait_time:.3f}s to respect API limits."
                    )
                    await asyncio.sleep(wait_time)

                    # Re-purge after sleep
                    now = time.monotonic()
                    self._timestamps = [
                        ts
                        for ts in self._timestamps
                        if now - ts < self._window_seconds
                    ]

            self._timestamps.append(time.monotonic())
            self._stats.requests_made += 1

    @property
    def stats(self) -> RateLimiterStats:
        """Access rate limiter stats for health reporting."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset stats counters. Called at the start of each pipeline run."""
        self._stats = RateLimiterStats()


class DailyQuotaLimiter:
    """Daily quota limiter for APIs with hard daily request caps.

    Used for NewsAPI (100 requests/day on free tier). Unlike the
    sliding window, this is a hard counter that resets at midnight IST.

    Safety margin: Stops 5 requests before the actual limit to leave
    room for manual/debug requests during the day.

    Behaviour when quota is exhausted:
      - Returns False from try_acquire() — caller must handle gracefully.
      - Does NOT block or sleep — there's no point waiting for a daily reset.
      - Logs a clear warning so the advisor pipeline can report degraded status.
    """

    def __init__(
        self,
        daily_limit: int,
        safety_margin: int = 5,
        name: str = "default",
    ) -> None:
        self._daily_limit = daily_limit
        self._safety_margin = safety_margin
        self._effective_limit = daily_limit - safety_margin
        self._name = name
        self._request_count: int = 0
        self._reset_day: float = time.monotonic()
        self._day_marker: int = self._get_day_marker()

    @staticmethod
    def _get_day_marker() -> int:
        """Get current day as integer for reset detection.

        Uses time.time() (wall clock) because daily resets must
        align with calendar days, not monotonic uptime.
        """
        return int(time.time() // 86400)

    def _check_daily_reset(self) -> None:
        """Reset counter if a new calendar day has started."""
        current_day = self._get_day_marker()
        if current_day != self._day_marker:
            logger.info(
                f"[{self._name}] New day detected — resetting daily quota counter. "
                f"Previous day used {self._request_count}/{self._effective_limit} requests."
            )
            self._request_count = 0
            self._day_marker = current_day

    def try_acquire(self) -> bool:
        """Attempt to use one request from the daily quota.

        Returns True if quota is available, False if exhausted.
        Does NOT block — the caller decides how to handle exhaustion
        (typically: return cached data or raise DataFetchError).

        Returns:
            True if the request is allowed, False if quota exhausted.
        """
        self._check_daily_reset()

        if self._request_count >= self._effective_limit:
            logger.warning(
                f"[{self._name}] Daily quota exhausted — "
                f"{self._request_count}/{self._effective_limit} requests used "
                f"(hard limit: {self._daily_limit}, safety margin: {self._safety_margin}). "
                f"Cache must serve remaining requests until midnight reset."
            )
            return False

        self._request_count += 1
        return True

    @property
    def remaining(self) -> int:
        """Number of requests remaining in today's quota."""
        self._check_daily_reset()
        return max(0, self._effective_limit - self._request_count)

    @property
    def usage_report(self) -> dict[str, int]:
        """Quota usage report for health monitoring."""
        self._check_daily_reset()
        return {
            "used": self._request_count,
            "remaining": self.remaining,
            "effective_limit": self._effective_limit,
            "hard_limit": self._daily_limit,
        }


class ConcurrencySemaphore:
    """Asyncio semaphore wrapper with stats tracking.

    Used for FRED API (120 req/min — generous, but we cap concurrent
    requests at 10 for good citizenship). Also usable as a general-
    purpose concurrency limiter for any async fetch operation.

    Unlike SlidingWindowLimiter, this only controls concurrency
    (how many requests run simultaneously), not throughput rate.
    """

    def __init__(self, max_concurrent: int, name: str = "default") -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._name = name
        self._stats = RateLimiterStats()

    async def acquire(self) -> None:
        """Acquire a concurrency slot. Blocks if all slots are in use."""
        await self._semaphore.acquire()
        self._stats.requests_made += 1

    def release(self) -> None:
        """Release a concurrency slot back to the pool."""
        self._semaphore.release()

    async def __aenter__(self) -> ConcurrencySemaphore:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.release()

    @property
    def stats(self) -> RateLimiterStats:
        """Access concurrency stats for health reporting."""
        return self._stats


# ─────────────────────────────────────────────────────────────
# Module-level singletons — shared by all fetchers
# Import as: from module1_data_layer.rate_limiter import kite_limiter
# ─────────────────────────────────────────────────────────────

# Kite Connect: 3 requests per second, 1-second sliding window
# Used by stock_fetcher.py, vix_fetcher.py, sector_fetcher.py
kite_limiter = SlidingWindowLimiter(
    max_requests=3,
    window_seconds=1.0,
    name="KiteConnect",
)

# NewsAPI: 100 requests per day, 5-request safety margin
# Used by news_fetcher.py only
news_limiter = DailyQuotaLimiter(
    daily_limit=100,
    safety_margin=5,
    name="NewsAPI",
)

# FRED API: 120 requests per minute — generous, cap concurrency at 10
# Used by economic_fetcher.py only
fred_limiter = ConcurrencySemaphore(
    max_concurrent=10,
    name="FRED",
)
