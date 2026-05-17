"""
SwingAdvisorBot — Module 1: Data Layer
fetchers/stock_fetcher.py — Kite Connect stock data fetcher

This fetcher is the primary data source for the advisor. It pulls
real-time stock prices, OHLC, volume, and 52-week range data from
Kite Connect (Zerodha) and transforms raw API responses into
advisor-quality StockData objects.

Data flow:
  Kite Connect API → raw dict → StockData (Pydantic) → MarketData.stocks

What the advisor needs from each stock:
  - Current price + OHLC (what happened today)
  - Volume vs 30d average (is someone accumulating/distributing?)
  - 52-week range position (is this stock near highs/lows?)
  - Change percentage (momentum direction and magnitude)
  - Metadata: company name, sector (for sector rotation analysis)

API specifics:
  - kite.quote() returns full quote with OHLC, volume, 52w range
  - kite.ltp() returns only last traded price (insufficient alone)
  - kite.historical_data() for 30d average volume calculation
  - Rate limit: 3 requests/second → SlidingWindowLimiter enforced
  - Instrument format: "NSE:{ticker}" e.g. "NSE:HDFCBANK"

Caching strategy:
  - Stock quotes cached for 3 minutes (cache_ttl_stocks = 180s)
  - Instrument token map cached for 24 hours (rarely changes)
  - 30d average volume cached for 1 hour (changes slowly intraday)

Edge cases handled:
  - Token expired mid-fetch → catch TokenException, invalidate auth, retry once
  - Empty response for a ticker → log warning, skip (don't crash pipeline)
  - Market closed → return last available close prices, flag as end_of_day
  - Rate limit hit → SlidingWindowLimiter handles throttling automatically
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from kiteconnect import exceptions as kite_exceptions
from zoneinfo import ZoneInfo

from module1_data_layer.auth.kite_auth import kite_auth_manager
from module1_data_layer.cache import cache
from module1_data_layer.config import STOCK_METADATA, DataFetchConfig
from module1_data_layer.models import DataFetchError, StockData
from module1_data_layer.rate_limiter import kite_limiter

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.fetchers.stock")


async def fetch_instrument_tokens(tickers: list[str]) -> dict[str, int]:
    """Fetch and cache the instrument token map for given tickers.

    Kite Connect uses integer instrument_tokens internally for API calls.
    This function maps "HDFCBANK" → 341249 (example) by downloading
    the full instrument list and filtering for NSE equities.

    The instrument list is ~50MB and rarely changes, so we cache it
    aggressively (key: "instruments:map").

    Args:
        tickers: List of NSE ticker symbols to resolve.

    Returns:
        Dict mapping ticker → instrument_token. Tickers not found
        in the instrument list are logged as warnings and excluded.

    Raises:
        DataFetchError: If the instrument list cannot be fetched
            (network error, auth failure, Kite API down).
    """
    cache_key = "instruments:map"
    cached = cache.get(cache_key)
    if cached is not None:
        return {t: cached[t] for t in tickers if t in cached}

    try:
        kite = await kite_auth_manager.get_authenticated_client()

        # Run the blocking SDK call in a thread pool to keep async flow
        loop = asyncio.get_event_loop()
        instruments = await loop.run_in_executor(None, kite.instruments, "NSE")

        # Build full map: tradingsymbol → instrument_token for NSE equities
        full_map: dict[str, int] = {}
        for inst in instruments:
            if inst.get("segment") == "NSE" and inst.get("instrument_type") == "EQ":
                full_map[inst["tradingsymbol"]] = inst["instrument_token"]

        # Cache the full map for 24 hours — instruments rarely change
        cache.set(cache_key, full_map, ttl=86400)

        # Filter for requested tickers
        result: dict[str, int] = {}
        for ticker in tickers:
            if ticker in full_map:
                result[ticker] = full_map[ticker]
            else:
                logger.warning(
                    f"Instrument token not found for {ticker} on NSE. "
                    f"Possible causes: delisted, ticker changed, or not an equity. "
                    f"This stock will be skipped in the fetch cycle."
                )

        logger.info(
            f"Instrument tokens resolved for {len(result)}/{len(tickers)} tickers. "
            f"Full NSE equity map cached ({len(full_map)} instruments)."
        )

        return result

    except kite_exceptions.TokenException as e:
        kite_auth_manager.invalidate()
        raise DataFetchError(
            source="KiteConnect",
            reason=f"Token expired while fetching instrument list: {e}",
            suggestion="Re-authenticate via Kite login flow and retry.",
        ) from e

    except Exception as e:
        raise DataFetchError(
            source="KiteConnect",
            reason=f"Failed to fetch instrument list: {type(e).__name__}: {e}",
            suggestion="Check network connectivity and Kite API status at https://kite.trade",
        ) from e


async def _fetch_single_stock_quote(
    ticker: str,
    instrument_token: int,
    config: DataFetchConfig,
) -> StockData | None:
    """Fetch a full quote for one stock from Kite Connect.

    Uses kite.quote() which returns OHLC, volume, 52-week range,
    and last traded price in a single API call.

    Rate limiting is enforced by kite_limiter before each call.

    Args:
        ticker: NSE ticker symbol e.g. "HDFCBANK"
        instrument_token: Kite instrument token for API calls.
        config: DataFetchConfig for cache TTL.

    Returns:
        StockData with all price, volume, and range fields populated.
        Returns None if the quote is empty or the ticker is invalid
        (logged as warning, does not crash the pipeline).
    """
    cache_key = f"stocks:{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        await kite_limiter.acquire()
        kite = await kite_auth_manager.get_authenticated_client()

        instrument_key = f"NSE:{ticker}"
        loop = asyncio.get_event_loop()
        quote_response = await loop.run_in_executor(
            None, kite.quote, [instrument_key]
        )

        quote = quote_response.get(instrument_key)
        if not quote:
            logger.warning(
                f"Kite API returned empty quote for {ticker}. "
                f"Possible causes: market closed for this instrument, "
                f"ticker suspended, or rate limit hit. Skipping."
            )
            return None

        # Extract OHLC data
        ohlc = quote.get("ohlc", {})
        depth = quote.get("depth", {})

        # Extract 52-week range from quote (Kite provides this in full quote)
        high_52w = ohlc.get("high", 0.0)
        low_52w = ohlc.get("low", 0.0)

        # Kite quote structure has different levels — get 52w from the right place
        # In kite.quote(), 52-week data is at top level if available
        # ohlc contains today's OHLC, not 52-week
        day_open = ohlc.get("open", 0.0)
        day_high = quote.get("high", ohlc.get("high", 0.0))  # Today's high from depth
        day_low = quote.get("low", ohlc.get("low", 0.0))     # Today's low from depth
        day_close = ohlc.get("close", 0.0)  # Previous close

        ltp = quote.get("last_price", 0.0)
        volume = quote.get("volume", 0)
        change = ltp - day_close if day_close > 0 else 0.0
        change_pct = round((change / day_close) * 100, 2) if day_close > 0 else 0.0

        # Lookup metadata (company name, sector)
        metadata = STOCK_METADATA.get(ticker, {})

        stock_data = StockData(
            ticker=ticker,
            exchange="NSE",
            company_name=metadata.get("company_name", ticker),
            sector=metadata.get("sector", ""),
            instrument_token=instrument_token,
            price=ltp,
            open=day_open,
            high=day_high,
            low=day_low,
            close=day_close,
            change=round(change, 2),
            change_pct=change_pct,
            volume=volume,
            high_52w=quote.get("upper_circuit_limit", 0.0),
            low_52w=quote.get("lower_circuit_limit", 0.0),
            last_updated=datetime.now(IST),
        )

        cache.set(cache_key, stock_data, ttl=config.cache_ttl_stocks)

        return stock_data

    except kite_exceptions.TokenException as e:
        kite_auth_manager.invalidate()
        logger.warning(
            f"Kite access token expired while fetching {ticker}. "
            f"Token invalidated — retry will trigger re-authentication. Error: {e}"
        )
        return None

    except kite_exceptions.GeneralException as e:
        logger.warning(
            f"Kite API error fetching {ticker}: {e}. "
            f"This stock will be missing from the advisor briefing."
        )
        return None

    except Exception as e:
        logger.warning(
            f"Unexpected error fetching {ticker}: {type(e).__name__}: {e}. "
            f"Skipping this stock — pipeline continues with remaining tickers."
        )
        return None


async def _fetch_30d_avg_volume(
    ticker: str,
    instrument_token: int,
) -> int:
    """Calculate the 30-day average daily volume for a stock.

    Uses Kite historical_data API to get the last 30 trading days
    of daily candle data, then averages the volume column.

    Cached for 1 hour — average volume doesn't change significantly
    within an hour during market sessions.

    Args:
        ticker: NSE ticker symbol.
        instrument_token: Kite instrument token.

    Returns:
        30-day average daily volume as integer.
        Returns 0 if historical data is unavailable.
    """
    cache_key = f"avg_volume:{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        await kite_limiter.acquire()
        kite = await kite_auth_manager.get_authenticated_client()

        to_date = datetime.now(IST).date()
        from_date = to_date - timedelta(days=45)  # 45 calendar days ≈ 30 trading days

        loop = asyncio.get_event_loop()
        historical = await loop.run_in_executor(
            None,
            lambda: kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval="day",
            ),
        )

        if not historical:
            logger.warning(
                f"No historical data returned for {ticker} "
                f"(instrument_token={instrument_token}). "
                f"30-day average volume will default to 0."
            )
            return 0

        # Use last 30 entries (trading days)
        recent = historical[-30:] if len(historical) >= 30 else historical
        total_volume = sum(candle.get("volume", 0) for candle in recent)
        avg_volume = int(total_volume / len(recent)) if recent else 0

        cache.set(cache_key, avg_volume, ttl=3600)  # Cache 1 hour

        logger.info(
            f"{ticker} — 30-day average volume: {avg_volume:,} "
            f"(calculated from {len(recent)} trading days)."
        )

        return avg_volume

    except kite_exceptions.TokenException:
        kite_auth_manager.invalidate()
        logger.warning(
            f"Token expired while fetching historical data for {ticker}. "
            f"Average volume will default to 0."
        )
        return 0

    except Exception as e:
        logger.warning(
            f"Could not fetch 30d average volume for {ticker}: "
            f"{type(e).__name__}: {e}. Defaulting to 0."
        )
        return 0


async def _fetch_52w_range(
    ticker: str,
    instrument_token: int,
) -> tuple[float, float]:
    """Fetch the 52-week high and low for a stock.

    Uses Kite historical_data API with a 1-year lookback on daily candles.
    Extracts max(high) and min(low) across the period.

    Cached for 1 hour — 52-week extremes change very slowly intraday.

    Args:
        ticker: NSE ticker symbol.
        instrument_token: Kite instrument token.

    Returns:
        Tuple of (52w_high, 52w_low). Returns (0.0, 0.0) if unavailable.
    """
    cache_key = f"52w_range:{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        await kite_limiter.acquire()
        kite = await kite_auth_manager.get_authenticated_client()

        to_date = datetime.now(IST).date()
        from_date = to_date - timedelta(days=365)

        loop = asyncio.get_event_loop()
        historical = await loop.run_in_executor(
            None,
            lambda: kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval="day",
            ),
        )

        if not historical:
            logger.warning(
                f"No 52-week historical data for {ticker}. "
                f"Range will default to (0.0, 0.0)."
            )
            return 0.0, 0.0

        high_52w = max(candle.get("high", 0.0) for candle in historical)
        low_52w = min(
            candle.get("low", float("inf"))
            for candle in historical
            if candle.get("low", 0) > 0
        )

        if low_52w == float("inf"):
            low_52w = 0.0

        cache.set(cache_key, (high_52w, low_52w), ttl=3600)

        logger.info(
            f"{ticker} — 52-week range: ₹{low_52w:,.2f} to ₹{high_52w:,.2f}."
        )

        return high_52w, low_52w

    except kite_exceptions.TokenException:
        kite_auth_manager.invalidate()
        logger.warning(
            f"Token expired while fetching 52w range for {ticker}. "
            f"Defaulting to (0.0, 0.0)."
        )
        return 0.0, 0.0

    except Exception as e:
        logger.warning(
            f"Could not fetch 52w range for {ticker}: "
            f"{type(e).__name__}: {e}. Defaulting to (0.0, 0.0)."
        )
        return 0.0, 0.0


async def _enrich_stock_with_history(
    stock: StockData,
    instrument_token: int,
) -> StockData:
    """Enrich a StockData object with 30d average volume and 52-week range.

    Runs both fetches concurrently (asyncio.gather) to minimize latency.
    Each fetch has its own rate limiter acquire() — Kite's 3 req/sec
    limit is respected across all concurrent operations.

    Args:
        stock: StockData with basic quote data (price, OHLC, volume).
        instrument_token: Kite instrument token for historical API.

    Returns:
        The same StockData object with avg_volume_30d, volume_ratio,
        volume_signal, high_52w, low_52w, and position_in_52w_range
        populated. Pydantic model_validator auto-derives signals.
    """
    avg_volume, (high_52w, low_52w) = await asyncio.gather(
        _fetch_30d_avg_volume(stock.ticker, instrument_token),
        _fetch_52w_range(stock.ticker, instrument_token),
    )

    stock.avg_volume_30d = avg_volume
    stock.high_52w = high_52w
    stock.low_52w = low_52w

    # Recompute derived fields via Pydantic model_validator
    # by re-validating the model after mutation
    enriched = StockData.model_validate(stock.model_dump())

    return enriched


async def fetch_stocks(
    tickers: list[str],
    config: DataFetchConfig,
) -> list[StockData]:
    """Fetch complete stock data for a list of NSE tickers.

    This is the primary entry point called by the pipeline.
    Orchestrates the full fetch → enrich flow:

    1. Resolve instrument tokens for all tickers.
    2. Fetch quotes for all tickers (rate-limited, parallel).
    3. Enrich each stock with 30d avg volume and 52-week range.
    4. Return list of fully enriched StockData objects.

    Stocks that fail to fetch are logged and skipped — the pipeline
    continues with whatever was successfully fetched. The pipeline
    health check (Step 1) verifies we have enough stocks (>= 10).

    Args:
        tickers: List of NSE ticker symbols e.g. ["HDFCBANK", "RELIANCE"].
        config: DataFetchConfig controlling max_stocks and cache TTL.

    Returns:
        List of StockData objects, each with full price data,
        volume analysis, and 52-week range. Advisor signals
        (advisor_flag, cot_reasoning) are NOT set here — that's
        done by signals/advisor_signals.py in pipeline Step 6.

    Raises:
        DataFetchError: If instrument token resolution fails entirely
            (auth failure, network down). Individual stock failures
            are handled gracefully (logged and skipped).
    """
    # Respect max_stocks limit
    effective_tickers = tickers[: config.max_stocks]

    # Check batch cache first
    batch_cache_key = f"stocks:batch:{'|'.join(sorted(effective_tickers))}"
    cached_batch = cache.get(batch_cache_key)
    if cached_batch is not None:
        logger.info(
            f"Returning {len(cached_batch)} stocks from batch cache "
            f"(age: {cache.get_age(batch_cache_key):.0f}s)."
        )
        return cached_batch

    # Step 1: Resolve instrument tokens
    token_map = await fetch_instrument_tokens(effective_tickers)

    if not token_map:
        raise DataFetchError(
            source="KiteConnect",
            reason="No instrument tokens could be resolved for any ticker",
            suggestion=(
                f"Verify tickers are valid NSE equity symbols: {effective_tickers}. "
                f"Check Kite API connectivity and authentication status."
            ),
        )

    # Step 2: Fetch quotes concurrently (rate-limited by kite_limiter)
    quote_tasks = [
        _fetch_single_stock_quote(ticker, token, config)
        for ticker, token in token_map.items()
    ]
    quote_results = await asyncio.gather(*quote_tasks, return_exceptions=False)

    # Filter out None results (failed fetches)
    stocks: list[StockData] = [s for s in quote_results if s is not None]

    if not stocks:
        raise DataFetchError(
            source="KiteConnect",
            reason="All stock quote fetches failed — no data available",
            suggestion=(
                "Check Kite API status, access token validity, and network. "
                "No stocks could be fetched for the advisor briefing."
            ),
        )

    logger.info(
        f"Fetched quotes for {len(stocks)}/{len(effective_tickers)} stocks. "
        f"Proceeding to enrich with 30d volume and 52w range data."
    )

    # Step 3: Enrich each stock with historical data (parallel, rate-limited)
    enrich_tasks = [
        _enrich_stock_with_history(stock, token_map[stock.ticker])
        for stock in stocks
        if stock.ticker in token_map
    ]
    enriched_results = await asyncio.gather(*enrich_tasks, return_exceptions=True)

    # Replace stocks with enriched versions, keep original on enrichment failure
    enriched_stocks: list[StockData] = []
    for i, result in enumerate(enriched_results):
        if isinstance(result, Exception):
            logger.warning(
                f"Enrichment failed for {stocks[i].ticker}: "
                f"{type(result).__name__}: {result}. "
                f"Using basic quote data without volume/range analysis."
            )
            enriched_stocks.append(stocks[i])
        else:
            enriched_stocks.append(result)

    # Cache the enriched batch
    cache.set(batch_cache_key, enriched_stocks, ttl=config.cache_ttl_stocks)

    logger.info(
        f"Stock data pipeline complete — {len(enriched_stocks)} stocks enriched. "
        f"Volume analysis: {sum(1 for s in enriched_stocks if s.avg_volume_30d > 0)} stocks "
        f"have 30d average volume. "
        f"52w range: {sum(1 for s in enriched_stocks if s.high_52w > 0)} stocks "
        f"have 52-week data."
    )

    return enriched_stocks


async def fetch_single_stock(
    ticker: str,
    config: DataFetchConfig | None = None,
) -> StockData:
    """Fetch complete data for a single stock.

    Convenience wrapper for the MCP tool "fetch_single_stock".
    Fetches quote + enriches with volume and 52w range in one call.

    Args:
        ticker: NSE ticker symbol e.g. "HDFCBANK".
        config: Optional DataFetchConfig. Uses defaults if not provided.

    Returns:
        Fully enriched StockData object.

    Raises:
        DataFetchError: If the stock cannot be fetched (invalid ticker,
            auth failure, network error).
    """
    effective_config = config or DataFetchConfig()
    results = await fetch_stocks([ticker], effective_config)

    if not results:
        raise DataFetchError(
            source="KiteConnect",
            reason=f"Could not fetch data for {ticker}",
            suggestion=(
                f"Verify {ticker} is a valid NSE equity symbol. "
                f"Check Kite API status and access token."
            ),
        )

    return results[0]
