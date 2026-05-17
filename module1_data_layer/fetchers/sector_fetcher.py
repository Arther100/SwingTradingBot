"""
SwingAdvisorBot — Module 1: Data Layer
fetchers/sector_fetcher.py — NSE sector performance fetcher

Sector rotation is how institutional money moves through the market.
When banking is up 2% and IT is down 1%, that tells the advisor
WHERE the money is flowing — critical for swing trade direction.

What the advisor needs from sector data:
  - Which sectors are leading today (bullish rotation targets)
  - Which sectors are lagging (bearish, avoid or short)
  - Top gainer/loser in each sector (find the leader/laggard)
  - Sector signal (bullish/bearish/neutral) for quick assessment

Data flow:
  Kite Connect (sector index quotes) → SectorPerformance objects
  + Stock data from stock_fetcher → top gainer/loser per sector
  → MarketData.sectors

API specifics:
  - Sector indices fetched via kite.quote() using NSE_SECTOR_INDICES mapping
  - Example: "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA"
  - All sector indices fetched in ONE batched kite.quote() call
  - Top gainer/loser derived from already-fetched stock data (no extra API call)

Caching:
  - Sector data cached with stock TTL (cache_ttl_stocks = 180s)
  - Sector indices update with market, same freshness as stock prices

Edge cases:
  - Sector index not available → skip sector, log warning
  - No stocks in a sector → top_gainer/top_loser left empty
  - Market closed → show last session's sector performance
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from kiteconnect import exceptions as kite_exceptions
from zoneinfo import ZoneInfo

from module1_data_layer.auth.kite_auth import kite_auth_manager
from module1_data_layer.cache import cache
from module1_data_layer.config import NSE_SECTOR_INDICES, DataFetchConfig
from module1_data_layer.models import SectorPerformance, SectorSignal, StockData
from module1_data_layer.rate_limiter import kite_limiter

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.fetchers.sector")


def _classify_sector_signal(change_pct: float) -> SectorSignal:
    """Classify sector direction based on percentage change.

    Thresholds for Indian market sector indices:
      > +0.5%  → bullish  (meaningful positive move)
      < -0.5%  → bearish  (meaningful negative move)
      else     → neutral  (noise, no directional bias)

    These thresholds are intentionally wider than stock-level
    thresholds because sector indices are less volatile and
    a 0.3% move is often noise from a single large-cap.

    Args:
        change_pct: Sector index percentage change for the session.

    Returns:
        SectorSignal enum member.
    """
    if change_pct > 0.5:
        return SectorSignal.BULLISH
    elif change_pct < -0.5:
        return SectorSignal.BEARISH
    else:
        return SectorSignal.NEUTRAL


def _generate_sector_advisor_note(
    sector_name: str,
    change_pct: float,
    signal: SectorSignal,
    top_gainer: str,
    top_gainer_pct: float,
    top_loser: str,
    top_loser_pct: float,
) -> str:
    """Generate an advisor-quality note for a sector.

    The advisor uses this note to quickly assess sector conditions
    without digging into individual stock data. Must be concise
    but informative — one sentence max.

    Examples:
      "Banking sector up 1.2% — bullish rotation. HDFCBANK leading (+2.1%), AXISBANK lagging (-0.3%)."
      "IT sector flat at +0.1% — neutral. No clear direction."
      "Metal sector down 1.8% — bearish pressure. TATASTEEL weakest (-3.2%)."

    Args:
        sector_name: Sector name (e.g. "Banking").
        change_pct: Sector index change percentage.
        signal: Classified sector signal.
        top_gainer: Ticker of best performer.
        top_gainer_pct: Change % of best performer.
        top_loser: Ticker of worst performer.
        top_loser_pct: Change % of worst performer.

    Returns:
        One-sentence advisor note string.
    """
    direction = {
        SectorSignal.BULLISH: "bullish rotation",
        SectorSignal.BEARISH: "bearish pressure",
        SectorSignal.NEUTRAL: "neutral, no clear direction",
    }[signal]

    note = f"{sector_name} sector {change_pct:+.1f}% — {direction}."

    if top_gainer and top_loser and top_gainer != top_loser:
        note += (
            f" {top_gainer} leading ({top_gainer_pct:+.1f}%), "
            f"{top_loser} lagging ({top_loser_pct:+.1f}%)."
        )
    elif top_gainer:
        note += f" {top_gainer} leading ({top_gainer_pct:+.1f}%)."

    return note


def _find_sector_movers(
    sector_name: str,
    stocks: list[StockData],
) -> tuple[str, float, str, float]:
    """Find the top gainer and top loser in a sector from fetched stocks.

    Uses the already-fetched stock data (from stock_fetcher) to avoid
    additional API calls. Matches stocks by their sector field.

    Args:
        sector_name: Sector name to filter by (e.g. "Banking").
        stocks: List of StockData from the current fetch cycle.

    Returns:
        Tuple of (top_gainer_ticker, top_gainer_change_pct,
                  top_loser_ticker, top_loser_change_pct).
        Returns ("", 0.0, "", 0.0) if no stocks match the sector.
    """
    sector_stocks = [s for s in stocks if s.sector == sector_name]

    if not sector_stocks:
        return "", 0.0, "", 0.0

    top_gainer = max(sector_stocks, key=lambda s: s.change_pct)
    top_loser = min(sector_stocks, key=lambda s: s.change_pct)

    return (
        top_gainer.ticker,
        top_gainer.change_pct,
        top_loser.ticker,
        top_loser.change_pct,
    )


async def fetch_sectors(
    config: DataFetchConfig,
    stocks: list[StockData] | None = None,
) -> list[SectorPerformance]:
    """Fetch sector index performance and derive sector signals.

    This is the primary entry point called by the pipeline.
    Fetches all NSE sector indices in ONE batched kite.quote() call,
    then enriches each sector with top gainer/loser from the
    already-fetched stock data.

    Args:
        config: DataFetchConfig for cache TTL.
        stocks: Already-fetched stock data for top gainer/loser lookup.
                If None, top_gainer/top_loser will be empty.

    Returns:
        List of SectorPerformance objects, one per sector in
        NSE_SECTOR_INDICES. Sorted by change_pct descending
        (strongest sector first) for the advisor's rotation view.

    Note:
        This function does NOT raise DataFetchError on failure.
        Sector data is Priority 5 — nice to have, not mandatory.
        Failures are logged and an empty list is returned.
    """
    cache_key = "sectors:all"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(
            f"Returning {len(cached)} sector results from cache "
            f"(age: {cache.get_age(cache_key):.0f}s)."
        )
        return cached

    effective_stocks = stocks or []

    try:
        await kite_limiter.acquire()
        kite = await kite_auth_manager.get_authenticated_client()

        # Build instrument keys for all sector indices
        # NSE_SECTOR_INDICES: {"Banking": "NIFTY BANK", "IT": "NIFTY IT", ...}
        instrument_keys = [
            f"NSE:{index_name}" for index_name in NSE_SECTOR_INDICES.values()
        ]

        # Single batched API call for all sector indices
        loop = asyncio.get_event_loop()
        quotes = await loop.run_in_executor(
            None, kite.quote, instrument_keys
        )

        sectors: list[SectorPerformance] = []

        for sector_name, index_name in NSE_SECTOR_INDICES.items():
            instrument_key = f"NSE:{index_name}"
            quote = quotes.get(instrument_key)

            if not quote:
                logger.warning(
                    f"No quote data for sector index {instrument_key}. "
                    f"Skipping {sector_name} sector in this cycle."
                )
                continue

            ltp = quote.get("last_price", 0.0)
            ohlc = quote.get("ohlc", {})
            prev_close = ohlc.get("close", ltp)

            change_pct = (
                round(((ltp - prev_close) / prev_close) * 100, 2)
                if prev_close > 0
                else 0.0
            )

            signal = _classify_sector_signal(change_pct)

            # Find top gainer/loser from fetched stocks
            gainer_ticker, gainer_pct, loser_ticker, loser_pct = (
                _find_sector_movers(sector_name, effective_stocks)
            )

            advisor_note = _generate_sector_advisor_note(
                sector_name=sector_name,
                change_pct=change_pct,
                signal=signal,
                top_gainer=gainer_ticker,
                top_gainer_pct=gainer_pct,
                top_loser=loser_ticker,
                top_loser_pct=loser_pct,
            )

            sector_perf = SectorPerformance(
                sector_name=sector_name,
                change_pct=change_pct,
                top_gainer=gainer_ticker,
                top_gainer_change_pct=gainer_pct,
                top_loser=loser_ticker,
                top_loser_change_pct=loser_pct,
                sector_signal=signal,
                advisor_note=advisor_note,
            )
            sectors.append(sector_perf)

        # Sort by change_pct descending — strongest sector first
        sectors.sort(key=lambda s: s.change_pct, reverse=True)

        # Cache the result
        cache.set(cache_key, sectors, ttl=config.cache_ttl_stocks)

        # Log sector summary
        if sectors:
            strongest = sectors[0]
            weakest = sectors[-1]
            logger.info(
                f"Sector rotation snapshot — {len(sectors)} sectors fetched. "
                f"Strongest: {strongest.sector_name} ({strongest.change_pct:+.1f}%). "
                f"Weakest: {weakest.sector_name} ({weakest.change_pct:+.1f}%). "
                f"Bullish: {sum(1 for s in sectors if s.sector_signal == SectorSignal.BULLISH)}, "
                f"Bearish: {sum(1 for s in sectors if s.sector_signal == SectorSignal.BEARISH)}, "
                f"Neutral: {sum(1 for s in sectors if s.sector_signal == SectorSignal.NEUTRAL)}."
            )
        else:
            logger.warning(
                "No sector data available. All sector index quotes were empty. "
                "Advisor will proceed without sector rotation context."
            )

        return sectors

    except kite_exceptions.TokenException as e:
        kite_auth_manager.invalidate()
        logger.warning(
            f"Kite token expired while fetching sector indices: {e}. "
            f"Sector data unavailable for this cycle. "
            f"Pipeline continues — sector data is Priority 5 (non-critical)."
        )
        return []

    except kite_exceptions.GeneralException as e:
        logger.warning(
            f"Kite API error fetching sector indices: {e}. "
            f"Sector data unavailable for this cycle."
        )
        return []

    except Exception as e:
        logger.warning(
            f"Unexpected error fetching sector data: {type(e).__name__}: {e}. "
            f"Sector data unavailable — pipeline continues without it."
        )
        return []
