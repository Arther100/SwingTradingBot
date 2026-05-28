"""
SwingAdvisorBot — Module Backtest
data/historical_fetcher.py — Kite Connect daily OHLCV history fetcher

Fetches 1 year of daily OHLCV bars for NSE stocks from Kite Connect's
historical data API. All results are cached in SQLite (data_cache.py)
so Kite is only called once per ticker per new trading day.

Pipeline:
  1. Resolve NSE ticker → Kite instrument_token (via instruments list)
  2. Check SQLite cache — return cached bars if up-to-date
  3. Fetch missing date range from kite.historical_data(interval="day")
  4. Validate response (OHLCV consistency, chronological order)
  5. Store new bars in SQLite cache
  6. Return list[OHLCVBar] sorted oldest-to-newest

Rate limiting:
  Kite historical API: 3 requests/second
  asyncio.Semaphore(KITE_HISTORICAL_MAX_CONCURRENT) + KITE_HISTORICAL_DELAY_SEC
  55 stocks × 1 call ≈ 25 seconds total — acceptable for batch job

No look-ahead bias:
  This fetcher only provides raw OHLCV bars.
  Indicators and signals are computed downstream.
  The engine never accesses bars beyond the replay date.

Usage:
    from module_backtest.data.historical_fetcher import historical_fetcher

    bars = await historical_fetcher.fetch(
        ticker="HDFCBANK",
        from_date=date(2025, 5, 1),
        to_date=date(2026, 5, 21),
    )
    # Returns list[OHLCVBar] sorted oldest-to-newest
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from zoneinfo import ZoneInfo

from module_backtest.config import (
    KITE_HISTORICAL_DELAY_SEC,
    KITE_HISTORICAL_MAX_CONCURRENT,
    NIFTY_INSTRUMENT_TOKEN,
)
from module_backtest.models import OHLCVBar

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.backtest.historical_fetcher")

# Shared semaphore — enforces Kite's 3 req/s limit across all concurrent fetches
_kite_semaphore = asyncio.Semaphore(KITE_HISTORICAL_MAX_CONCURRENT)

# Ticker alias map — handles renamed/merged companies and special characters.
# If the original symbol is not found in Kite instruments, the alias is tried.
# Add entries bidirectionally so resolution works from either name.
_TICKER_ALIASES: dict[str, str] = {
    "LTIM":         "LTIMINDTREE",  # LTIMindtree (post-merger: L&T Infotech + Mindtree)
    "LTIMINDTREE":  "LTIM",         # reverse alias
    "M&M":          "MM",           # Mahindra — special char variant
    "MM":           "M&M",          # reverse
    "BAJAJ-AUTO":   "BAJAJAUTO",    # hyphen variant
    "BAJAJAUTO":    "BAJAJ-AUTO",   # reverse
}


class HistoricalFetcher:
    """Fetches daily OHLCV history from Kite Connect with caching.

    All Kite SDK calls are synchronous — they are run via
    loop.run_in_executor() to avoid blocking the async event loop.

    Cache is checked before every fetch. Only new dates are fetched
    incrementally so subsequent runs are near-instant.
    """

    def __init__(self) -> None:
        # Instrument token cache: ticker → token (loaded once per session)
        self._token_cache: dict[str, int] = {}
        # Raw instruments list — fetched once, reused for all lookups
        self._instruments: list[dict] = []
        # Single import guard for data_cache (avoids circular import at module level)
        self._cache: Optional[object] = None

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    async def fetch(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
    ) -> list[OHLCVBar]:
        """Fetch daily OHLCV bars for a single ticker.

        Returns bars sorted oldest-to-newest. Checks SQLite cache
        first — only calls Kite for the missing date range.

        Args:
            ticker:    NSE ticker symbol e.g. "HDFCBANK"
            from_date: Start date (inclusive)
            to_date:   End date (inclusive)

        Returns:
            list[OHLCVBar] sorted by date ascending. May be empty
            if Kite has no data (e.g. ticker not listed on NSE).

        Raises:
            Never raises — returns empty list on any failure and logs error.
        """
        try:
            return await self._fetch_with_cache(ticker, from_date, to_date)
        except Exception as exc:
            logger.error(
                f"[HistoricalFetcher] Failed to fetch {ticker}: {exc}",
                exc_info=True,
            )
            return []

    async def fetch_many(
        self,
        tickers: list[str],
        from_date: date,
        to_date: date,
    ) -> dict[str, list[OHLCVBar]]:
        """Fetch daily OHLCV for multiple tickers concurrently.

        Respects Kite rate limit via asyncio.Semaphore(3).

        Args:
            tickers:   List of NSE ticker symbols
            from_date: Start date (inclusive)
            to_date:   End date (inclusive)

        Returns:
            dict[ticker → list[OHLCVBar]] for all requested tickers.
            Missing or failed tickers are included as empty lists.
        """
        tasks = [
            self.fetch(ticker=t, from_date=from_date, to_date=to_date)
            for t in tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return {ticker: bars for ticker, bars in zip(tickers, results)}

    async def fetch_nifty(
        self,
        from_date: date,
        to_date: date,
    ) -> list[OHLCVBar]:
        """Fetch Nifty 50 daily OHLCV for benchmark comparison.

        Uses the fixed NIFTY_INSTRUMENT_TOKEN from config.
        No instrument lookup needed — token is hardcoded.
        """
        try:
            return await self._fetch_by_token(
                instrument_token=NIFTY_INSTRUMENT_TOKEN,
                ticker="NIFTY_50",
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as exc:
            logger.error(f"[HistoricalFetcher] Nifty fetch failed: {exc}")
            return []

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    async def _fetch_with_cache(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
    ) -> list[OHLCVBar]:
        """Check cache, fetch only missing range, merge and return."""
        cache = self._get_cache()

        # ── Step 1: Load what we already have ──────────────────
        cached_bars = cache.get_bars(ticker, from_date, to_date)
        cached_dates: set[date] = {b.date for b in cached_bars}

        # ── Step 2: Determine what's missing ───────────────────
        # Build expected trading day range (weekdays; exchange holidays
        # will simply have no data and that's fine)
        expected = _date_range(from_date, to_date)
        missing_dates = sorted(d for d in expected if d not in cached_dates)

        if not missing_dates:
            logger.debug(f"[HistoricalFetcher] {ticker}: full cache hit")
            return sorted(cached_bars, key=lambda b: b.date)

        # ── Step 3: Resolve instrument token ───────────────────
        token = await self._resolve_token(ticker)
        if token is None:
            logger.warning(
                f"[HistoricalFetcher] {ticker}: instrument token not found"
            )
            return sorted(cached_bars, key=lambda b: b.date)

        # ── Step 4: Fetch missing range from Kite ──────────────
        fetch_from = missing_dates[0]
        fetch_to = missing_dates[-1]

        logger.info(
            f"[HistoricalFetcher] {ticker}: fetching "
            f"{fetch_from} → {fetch_to} "
            f"({len(missing_dates)} missing days)"
        )

        new_bars = await self._fetch_by_token(
            instrument_token=token,
            ticker=ticker,
            from_date=fetch_from,
            to_date=fetch_to,
        )

        # ── Step 5: Persist new bars to cache ──────────────────
        if new_bars:
            cache.store_bars(ticker, new_bars)
            logger.info(
                f"[HistoricalFetcher] {ticker}: stored {len(new_bars)} bars"
            )

        # ── Step 6: Merge cached + new, sort, return ───────────
        all_bars = {b.date: b for b in cached_bars}
        for b in new_bars:
            all_bars[b.date] = b

        result = [b for b in all_bars.values() if from_date <= b.date <= to_date]
        result.sort(key=lambda b: b.date)

        logger.info(
            f"[HistoricalFetcher] {ticker}: {len(result)} bars total "
            f"({len(cached_bars)} cached + {len(new_bars)} new)"
        )
        return result

    async def _fetch_by_token(
        self,
        instrument_token: int,
        ticker: str,
        from_date: date,
        to_date: date,
    ) -> list[OHLCVBar]:
        """Call kite.historical_data() with rate limiting.

        Runs the blocking Kite SDK call in a thread executor.
        Applies semaphore + delay to respect 3 req/s limit.

        Returns:
            Validated list[OHLCVBar], may be empty on failure.
        """
        async with _kite_semaphore:
            try:
                kite = await self._get_kite_client()
                loop = asyncio.get_event_loop()

                # Convert date → datetime for Kite API (requires datetime)
                from_dt = datetime(from_date.year, from_date.month, from_date.day)
                to_dt = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)

                raw_data = await loop.run_in_executor(
                    None,
                    lambda: kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=from_dt,
                        to_date=to_dt,
                        interval="day",
                    ),
                )

                bars = self._parse_kite_response(raw_data, ticker)

                # Delay after each request to respect rate limit
                await asyncio.sleep(KITE_HISTORICAL_DELAY_SEC)

                return bars

            except Exception as exc:
                logger.warning(
                    f"[HistoricalFetcher] {ticker} token={instrument_token}: "
                    f"Kite call failed — {type(exc).__name__}: {exc}"
                )
                return []

    def _parse_kite_response(
        self,
        raw_data: list[dict],
        ticker: str,
    ) -> list[OHLCVBar]:
        """Parse and validate Kite historical_data() response.

        Kite returns list of dicts:
            {"date": datetime, "open": float, "high": float,
             "low": float, "close": float, "volume": int}

        Validates OHLCV consistency. Skips malformed bars.

        Returns:
            Validated list[OHLCVBar] sorted oldest-to-newest.
        """
        if not raw_data:
            return []

        bars: list[OHLCVBar] = []
        for raw in raw_data:
            try:
                # Kite returns date as datetime — extract just the date part
                raw_date = raw.get("date")
                if isinstance(raw_date, datetime):
                    bar_date = raw_date.date()
                elif isinstance(raw_date, date):
                    bar_date = raw_date
                else:
                    logger.warning(
                        f"[HistoricalFetcher] {ticker}: unexpected date type "
                        f"{type(raw_date)} — skipping bar"
                    )
                    continue

                bar = OHLCVBar(
                    date=bar_date,
                    open=float(raw["open"]),
                    high=float(raw["high"]),
                    low=float(raw["low"]),
                    close=float(raw["close"]),
                    volume=int(raw.get("volume", 0)),
                )
                bars.append(bar)

            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    f"[HistoricalFetcher] {ticker}: skipping malformed bar "
                    f"{raw} — {exc}"
                )
                continue

        # Sort ascending — oldest first
        bars.sort(key=lambda b: b.date)

        logger.debug(
            f"[HistoricalFetcher] {ticker}: parsed {len(bars)} valid bars "
            f"from {len(raw_data)} raw"
        )
        return bars

    async def _resolve_token(self, ticker: str) -> Optional[int]:
        """Resolve NSE ticker symbol to Kite instrument_token.

        Resolution order (CoT):
          Step 1 — Check in-memory token cache (hit = instant return)
          Step 2 — Load full instruments list once per session
          Step 3 — Exact match: tradingsymbol == ticker AND type == EQ
          Step 4 — EQ-suffix match: tradingsymbol == "{ticker}-EQ"
          Step 5 — Case-insensitive match on loaded token map
          Step 6 — Log warning with similar symbols for debugging

        Handles Kite accounts where segment is "NSE-EQ" instead of "NSE",
        and tickers stored as "TATAMOTORS-EQ" instead of "TATAMOTORS".

        Returns:
            instrument_token int, or None if ticker not found.
        """
        # Step 1: Cache hit
        if ticker in self._token_cache:
            return self._token_cache[ticker]

        try:
            kite = await self._get_kite_client()
            loop = asyncio.get_event_loop()

            # Step 2: Fetch instruments list once per session
            if not self._instruments:
                self._instruments = await loop.run_in_executor(
                    None, lambda: kite.instruments("NSE")
                )
                logger.info(
                    f"[InstrumentCache] Loaded {len(self._instruments)} "
                    f"NSE instruments"
                )

                # Build comprehensive token map in one pass
                for inst in self._instruments:
                    sym: str = inst.get("tradingsymbol", "")
                    itype: str = inst.get("instrument_type", "")
                    seg: str = inst.get("segment", "")

                    # Accept EQ instruments from any NSE segment variant
                    is_eq = itype == "EQ"
                    is_nse = seg in ("NSE", "NSE-EQ", "NSE_EQ")

                    if not (is_eq and is_nse):
                        continue

                    tok = inst["instrument_token"]
                    self._token_cache[sym] = tok

                    # Map base symbol when stored with -EQ suffix
                    # e.g. "TATAMOTORS-EQ" → also map "TATAMOTORS"
                    if sym.endswith("-EQ"):
                        base = sym[:-3]
                        if base not in self._token_cache:
                            self._token_cache[base] = tok

                logger.info(
                    f"[InstrumentCache] Mapped {len(self._token_cache)} "
                    f"ticker → token entries"
                )

            # Step 3: Exact match (already in token_cache from build)
            if ticker in self._token_cache:
                tok = self._token_cache[ticker]
                logger.debug(
                    f"[HistoricalFetcher] {ticker}: token={tok} (exact match)"
                )
                return tok

            # Step 4: EQ-suffix match
            eq_sym = f"{ticker}-EQ"
            if eq_sym in self._token_cache:
                tok = self._token_cache[eq_sym]
                self._token_cache[ticker] = tok  # prime for future calls
                logger.info(
                    f"[HistoricalFetcher] {ticker}: token={tok} (via {eq_sym})"
                )
                return tok

            # Step 5: Case-insensitive scan
            ticker_up = ticker.upper()
            for sym, tok in self._token_cache.items():
                if sym.upper() == ticker_up:
                    self._token_cache[ticker] = tok  # prime
                    logger.info(
                        f"[HistoricalFetcher] {ticker}: token={tok} "
                        f"(case-insensitive match via '{sym}')"
                    )
                    return tok

            # Step 5b: Alias fallback — try known ticker alias
            alias = _TICKER_ALIASES.get(ticker) or _TICKER_ALIASES.get(ticker.upper())
            if alias:
                # Exact alias match
                if alias in self._token_cache:
                    tok = self._token_cache[alias]
                    self._token_cache[ticker] = tok  # prime for future calls
                    logger.info(
                        f"[HistoricalFetcher] {ticker}: token={tok} "
                        f"(via alias '{alias}')"
                    )
                    return tok
                # Case-insensitive alias match
                alias_up = alias.upper()
                for sym, tok in self._token_cache.items():
                    if sym.upper() == alias_up:
                        self._token_cache[ticker] = tok
                        logger.info(
                            f"[HistoricalFetcher] {ticker}: token={tok} "
                            f"(via alias '{alias}' → '{sym}')"
                        )
                        return tok

            # Step 6: Not found — log with similar symbols for debugging
            prefix = ticker[:4].upper()
            similar = sorted({
                inst["tradingsymbol"]
                for inst in self._instruments
                if prefix in inst.get("tradingsymbol", "").upper()
            })[:8]
            logger.warning(
                f"[HistoricalFetcher] {ticker}: instrument token not found. "
                f"Similar NSE symbols: {similar}"
            )
            return None

        except Exception as exc:
            logger.error(
                f"[HistoricalFetcher] Failed to load instruments: {exc}"
            )
            return None

    async def _get_kite_client(self):
        """Return authenticated KiteConnect client from M1 auth manager."""
        from module1_data_layer.auth.kite_auth import kite_auth_manager
        return await kite_auth_manager.get_authenticated_client()

    def _get_cache(self):
        """Lazy-import data_cache to avoid circular imports."""
        if self._cache is None:
            from module_backtest.data.data_cache import historical_cache
            self._cache = historical_cache
        return self._cache


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def _date_range(from_date: date, to_date: date) -> list[date]:
    """Return weekday dates between from_date and to_date inclusive.

    NSE is closed on weekends. We use weekdays as a proxy for
    expected trading days — the cache simply won't have weekend
    dates, which is correct behaviour.
    """
    result: list[date] = []
    current = from_date
    while current <= to_date:
        if current.weekday() < 5:  # 0=Mon ... 4=Fri
            result.append(current)
        current += timedelta(days=1)
    return result


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

historical_fetcher = HistoricalFetcher()
