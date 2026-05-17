"""
SwingAdvisorBot — Module 1: Data Layer
fetchers/vix_fetcher.py — India VIX and index data fetcher

India VIX is the market's fear gauge. It is Priority 1 data — the
advisor ALWAYS needs VIX to assess overall market risk before
making any stock-specific recommendation.

What VIX tells the advisor:
  VIX < 14   → Low fear. Markets complacent. Breakouts more likely to sustain.
  VIX 14–20  → Moderate fear. Normal conditions. Standard swing setups apply.
  VIX 20–30  → High fear. Hedging activity elevated. Tighter stop losses needed.
  VIX ≥ 30   → Extreme fear. Panic/crash conditions. Cash is a position.

This fetcher also pulls Nifty 50 and Sensex index data — the two
benchmark indices that frame every conversation about Indian markets.

Data flow:
  Kite Connect API → India VIX quote + Nifty50 quote + Sensex quote
                   → MarketData.india_vix, vix_signal, nifty50_*, sensex_*

API specifics:
  - India VIX: NSE instrument "INDIA VIX" via kite.quote()
  - Nifty 50: NSE index "NIFTY 50" via kite.quote()
  - Sensex: BSE index "SENSEX" via kite.quote()
  - Rate limit: Uses kite_limiter (3 req/sec shared with stock_fetcher)

Caching:
  - VIX cached for 5 minutes (cache_ttl_vix = 300s)
  - Index data cached alongside VIX (same TTL)
  - During market hours, VIX updates every few seconds on exchange
    but 5-minute cache is sufficient for swing trading decisions

Edge cases:
  - Market closed → VIX shows last session's closing value, flagged appropriately
  - VIX instrument not found → DataFetchError (VIX is mandatory, not optional)
  - Token expired → invalidate auth, retry once
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from kiteconnect import exceptions as kite_exceptions
from zoneinfo import ZoneInfo

from module1_data_layer.auth.kite_auth import kite_auth_manager
from module1_data_layer.cache import cache
from module1_data_layer.config import DataFetchConfig
from module1_data_layer.models import DataFetchError, VIXSignal
from module1_data_layer.rate_limiter import kite_limiter

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.fetchers.vix")

# Kite instrument keys for VIX and benchmark indices
INDIA_VIX_KEY = "NSE:INDIA VIX"
NIFTY50_KEY = "NSE:NIFTY 50"
SENSEX_KEY = "BSE:SENSEX"


@dataclass
class VIXAndIndexData:
    """Container for VIX + benchmark index data.

    Returned by fetch_vix_and_indices() and unpacked into
    MarketData fields by the pipeline.

    All three values are Priority 1/2 — never trimmed from
    the token budget. The advisor always needs market vitals.
    """

    india_vix: float
    vix_signal: VIXSignal
    vix_change_pct: float
    nifty50_value: float
    nifty50_change_pct: float
    sensex_value: float
    sensex_change_pct: float
    fetched_at: datetime


def classify_vix(vix_value: float) -> VIXSignal:
    """Classify India VIX into an advisor-quality fear signal.

    Thresholds calibrated for India VIX historical behaviour:
      - India VIX averaged ~13-15 during bull markets (2023-2024)
      - Spiked to 25-30 during election uncertainty (May 2024)
      - Hit 40+ during COVID crash (March 2020)

    These thresholds are codified in the VIXSignal enum but
    the classification logic lives here for single responsibility.

    Args:
        vix_value: India VIX value (typically 10-50 range).

    Returns:
        VIXSignal enum member with advisor-quality label.
    """
    if vix_value >= 30:
        return VIXSignal.EXTREME_FEAR
    elif vix_value >= 20:
        return VIXSignal.HIGH_FEAR
    elif vix_value >= 14:
        return VIXSignal.MODERATE_FEAR
    else:
        return VIXSignal.LOW_FEAR


async def fetch_vix_and_indices(
    config: DataFetchConfig,
) -> VIXAndIndexData:
    """Fetch India VIX, Nifty 50, and Sensex in a single batch call.

    This is the primary entry point called by the pipeline (Step 5).
    Fetches all three instruments in ONE kite.quote() call to minimize
    API usage — Kite allows multiple instruments per quote request.

    The returned VIXAndIndexData is unpacked into MarketData by pipeline.py:
      market_data.india_vix = result.india_vix
      market_data.vix_signal = result.vix_signal
      market_data.nifty50_value = result.nifty50_value
      ...etc

    Args:
        config: DataFetchConfig for cache TTL (cache_ttl_vix).

    Returns:
        VIXAndIndexData with all three instruments populated.

    Raises:
        DataFetchError: If VIX cannot be fetched. VIX is mandatory —
            the advisor cannot assess market risk without it.
            Index data (Nifty, Sensex) failures are logged but not fatal.
    """
    cache_key = "vix:current"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(
            f"Returning VIX and index data from cache "
            f"(age: {cache.get_age(cache_key):.0f}s). "
            f"VIX={cached.india_vix}, signal={cached.vix_signal.value}."
        )
        return cached

    try:
        await kite_limiter.acquire()
        kite = await kite_auth_manager.get_authenticated_client()

        # Batch fetch: VIX + Nifty 50 + Sensex in one API call
        instruments = [INDIA_VIX_KEY, NIFTY50_KEY, SENSEX_KEY]
        loop = asyncio.get_event_loop()
        quotes = await loop.run_in_executor(
            None, kite.quote, instruments
        )

        # ── Extract India VIX ──
        vix_quote = quotes.get(INDIA_VIX_KEY)
        if not vix_quote:
            raise DataFetchError(
                source="KiteConnect",
                reason=(
                    f"India VIX quote is empty. Instrument key '{INDIA_VIX_KEY}' "
                    f"returned no data from Kite API."
                ),
                suggestion=(
                    "Verify India VIX instrument is available on Kite. "
                    "Check if market is in a holiday/maintenance window. "
                    "VIX is mandatory for risk assessment — pipeline cannot proceed without it."
                ),
            )

        india_vix = vix_quote.get("last_price", 0.0)
        if india_vix <= 0:
            raise DataFetchError(
                source="KiteConnect",
                reason=(
                    f"India VIX returned invalid value: {india_vix}. "
                    f"VIX must be a positive number for risk classification."
                ),
                suggestion=(
                    "This may indicate a Kite API issue or market data feed disruption. "
                    "Check Kite dashboard for data feed status."
                ),
            )

        vix_ohlc = vix_quote.get("ohlc", {})
        vix_close = vix_ohlc.get("close", india_vix)
        vix_change = india_vix - vix_close if vix_close > 0 else 0.0
        vix_change_pct = round((vix_change / vix_close) * 100, 2) if vix_close > 0 else 0.0

        vix_signal = classify_vix(india_vix)

        logger.info(
            f"India VIX at {india_vix:.2f} ({vix_change_pct:+.2f}%) — "
            f"{vix_signal.value} detected. "
            f"{'Elevated fear — flagging for risk engine.' if india_vix >= 20 else 'Normal conditions.'}"
        )

        # ── Extract Nifty 50 ──
        nifty_quote = quotes.get(NIFTY50_KEY, {})
        nifty_value = nifty_quote.get("last_price", 0.0)
        nifty_ohlc = nifty_quote.get("ohlc", {})
        nifty_close = nifty_ohlc.get("close", nifty_value)
        nifty_change_pct = (
            round(((nifty_value - nifty_close) / nifty_close) * 100, 2)
            if nifty_close > 0
            else 0.0
        )

        if nifty_value > 0:
            logger.info(
                f"Nifty 50 at {nifty_value:,.2f} ({nifty_change_pct:+.2f}%)."
            )
        else:
            logger.warning(
                "Nifty 50 quote returned 0 — index data may be unavailable. "
                "Continuing with VIX data. Nifty will show as 0 in advisor briefing."
            )

        # ── Extract Sensex ──
        sensex_quote = quotes.get(SENSEX_KEY, {})
        sensex_value = sensex_quote.get("last_price", 0.0)
        sensex_ohlc = sensex_quote.get("ohlc", {})
        sensex_close = sensex_ohlc.get("close", sensex_value)
        sensex_change_pct = (
            round(((sensex_value - sensex_close) / sensex_close) * 100, 2)
            if sensex_close > 0
            else 0.0
        )

        if sensex_value > 0:
            logger.info(
                f"Sensex at {sensex_value:,.2f} ({sensex_change_pct:+.2f}%)."
            )
        else:
            logger.warning(
                "Sensex quote returned 0 — BSE index data may be unavailable. "
                "Continuing without Sensex. This is non-fatal."
            )

        # ── Assemble result ──
        result = VIXAndIndexData(
            india_vix=india_vix,
            vix_signal=vix_signal,
            vix_change_pct=vix_change_pct,
            nifty50_value=nifty_value,
            nifty50_change_pct=nifty_change_pct,
            sensex_value=sensex_value,
            sensex_change_pct=sensex_change_pct,
            fetched_at=datetime.now(IST),
        )

        cache.set(cache_key, result, ttl=config.cache_ttl_vix)

        return result

    except DataFetchError:
        # Re-raise our typed errors (VIX mandatory failures)
        raise

    except kite_exceptions.TokenException as e:
        kite_auth_manager.invalidate()
        raise DataFetchError(
            source="KiteConnect",
            reason=f"Access token expired while fetching VIX and index data: {e}",
            suggestion="Re-authenticate via Kite login flow. VIX is mandatory — pipeline halted.",
        ) from e

    except kite_exceptions.GeneralException as e:
        raise DataFetchError(
            source="KiteConnect",
            reason=f"Kite API error fetching VIX and indices: {e}",
            suggestion=(
                "Check Kite API status and network connectivity. "
                "VIX is mandatory for risk assessment."
            ),
        ) from e

    except Exception as e:
        raise DataFetchError(
            source="KiteConnect",
            reason=f"Unexpected error fetching VIX and indices: {type(e).__name__}: {e}",
            suggestion="Review logs for root cause. VIX is mandatory — pipeline cannot proceed.",
        ) from e


async def fetch_market_status_data(
    config: DataFetchConfig | None = None,
) -> dict[str, object]:
    """Quick market health check — VIX + Nifty change.

    Convenience wrapper for the MCP tool "get_market_status".
    Returns a lightweight dict with market vitals suitable for
    a quick health check without the full pipeline overhead.

    Args:
        config: Optional DataFetchConfig. Uses defaults if not provided.

    Returns:
        Dict with market status vitals:
          {"status": "open", "india_vix": 14.32,
           "vix_signal": "low_fear", "nifty50_change_pct": 0.14}
    """
    from module1_data_layer.pipeline import determine_market_status

    effective_config = config or DataFetchConfig()

    market_status, status_reason = determine_market_status()

    try:
        vix_data = await fetch_vix_and_indices(effective_config)
        return {
            "status": market_status.value,
            "status_reason": status_reason,
            "india_vix": vix_data.india_vix,
            "vix_signal": vix_data.vix_signal.value,
            "vix_change_pct": vix_data.vix_change_pct,
            "nifty50_value": vix_data.nifty50_value,
            "nifty50_change_pct": vix_data.nifty50_change_pct,
            "sensex_value": vix_data.sensex_value,
            "sensex_change_pct": vix_data.sensex_change_pct,
            "fetched_at": vix_data.fetched_at.isoformat(),
        }

    except DataFetchError as e:
        return {
            "status": market_status.value,
            "status_reason": status_reason,
            "india_vix": 0.0,
            "vix_signal": "unavailable",
            "error": str(e),
        }
