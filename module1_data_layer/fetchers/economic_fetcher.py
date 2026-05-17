"""
SwingAdvisorBot — Module 1: Data Layer
fetchers/economic_fetcher.py — FRED macro-economic data fetcher

Global macro events move Indian markets. When the Fed raises rates,
FII money flows out of India. When crude oil spikes, India's fiscal
deficit widens and the rupee weakens. The advisor needs this context
to explain WHY Indian stocks are moving, not just WHAT is happening.

Tracked indicators (from config.FRED_SERIES):
  FEDFUNDS   — Federal Funds Rate (US interest rates → FII flows)
  CPIAUCSL   — US CPI (inflation → Fed policy → global risk)
  DCOILWTICO — Crude Oil WTI (India imports 85% → rupee + fiscal)
  DGS10      — US 10-Year Treasury (competes with EM equities)
  DEXINUS    — USD/INR Exchange Rate (rupee strength → export/import)

Data flow:
  FRED API → raw JSON → EconomicEvent (Pydantic) → MarketData.economic_events

API specifics:
  - Base URL: https://api.stlouisfed.org/fred/series/observations
  - Auth: API key as query parameter
  - Response: JSON with observations array
  - Rate limit: 120 requests/minute (generous — we use ConcurrencySemaphore(10))
  - Each series fetched independently, all concurrent via asyncio.gather

Caching:
  - Economic data cached for 60 minutes (cache_ttl_events = 3600s)
  - Macro indicators update daily or less frequently
  - 60-minute cache ensures we rarely hit the API during a trading day

Edge cases:
  - FRED API key missing → DataFetchError with setup instructions
  - Series not found → skip, log warning (non-fatal)
  - No new observations → use last available value
  - FRED API down → return empty list (Priority 6 — trimmed first)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import httpx
from zoneinfo import ZoneInfo

from module1_data_layer.cache import cache
from module1_data_layer.config import FRED_SERIES, DataFetchConfig, get_settings
from module1_data_layer.models import (
    DataFetchError,
    EconomicEvent,
    MarketImpact,
)
from module1_data_layer.rate_limiter import fred_limiter

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.fetchers.economic")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Impact classification for each FRED series.
# Maps series_id → MarketImpact based on how significantly
# changes in this indicator affect Indian equity markets.
SERIES_IMPACT: dict[str, MarketImpact] = {
    "FEDFUNDS": MarketImpact.HIGH,      # Fed rate directly moves FII flows
    "CPIAUCSL": MarketImpact.HIGH,      # Inflation drives Fed policy
    "DCOILWTICO": MarketImpact.HIGH,    # India imports 85% crude oil
    "DGS10": MarketImpact.MEDIUM,       # Indirect via global risk appetite
    "DEXINUS": MarketImpact.MEDIUM,     # Rupee affects export/import sectors
}


def _generate_economic_advisor_note(
    event_name: str,
    series_id: str,
    value: float,
    previous_value: float,
    impact_note: str,
) -> str:
    """Generate an advisor-quality note for an economic indicator.

    The advisor uses this note to contextualize macro data for
    Indian market impact. Must be concise and actionable.

    Examples:
      "Federal Funds Rate at 5.50% (unchanged from 5.50%).
       US rate changes affect FII flows into Indian equities."
      "Crude Oil WTI at $82.30 (up from $78.50).
       Rising crude pressures India's fiscal deficit and weakens INR."

    Args:
        event_name: Human-readable indicator name.
        series_id: FRED series identifier.
        value: Latest observation value.
        previous_value: Previous observation value.
        impact_note: Pre-configured impact description from FRED_SERIES.

    Returns:
        Advisor note string.
    """
    if previous_value > 0:
        change = value - previous_value
        direction = "up" if change > 0 else "down" if change < 0 else "unchanged"

        if direction == "unchanged":
            note = f"{event_name} at {value:.2f} (unchanged from {previous_value:.2f})."
        else:
            note = f"{event_name} at {value:.2f} ({direction} from {previous_value:.2f})."
    else:
        note = f"{event_name} at {value:.2f}."

    note += f" {impact_note}"
    return note


async def _fetch_single_series(
    series_id: str,
    series_config: dict[str, str],
    api_key: str,
) -> EconomicEvent | None:
    """Fetch the latest observation for a single FRED series.

    Makes an HTTP GET to the FRED observations endpoint with
    sort_order=desc and limit=2 to get the latest and previous
    observations for change direction calculation.

    Args:
        series_id: FRED series ID e.g. "FEDFUNDS".
        series_config: Config dict with event_name and impact_note.
        api_key: FRED API key from settings.

    Returns:
        EconomicEvent with latest value and change direction.
        Returns None if the fetch fails (logged, non-fatal).
    """
    cache_key = f"economic:{series_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        async with fred_limiter:
            # Fetch last 2 observations for current + previous value
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 2,
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(FRED_BASE_URL, params=params)

            if response.status_code == 400:
                logger.warning(
                    f"FRED API returned 400 for series {series_id}. "
                    f"Series may not exist or parameters are invalid. Skipping."
                )
                return None

            if response.status_code == 401:
                raise DataFetchError(
                    source="FRED",
                    reason=f"FRED API key invalid or expired (HTTP 401) for series {series_id}",
                    suggestion="Verify FRED_API_KEY in .env. Get a key from https://fred.stlouisfed.org/docs/api/api_key.html",
                )

            if response.status_code != 200:
                logger.warning(
                    f"FRED API returned HTTP {response.status_code} for {series_id}: "
                    f"{response.text[:200]}. Skipping this indicator."
                )
                return None

            data = response.json()
            observations = data.get("observations", [])

            if not observations:
                logger.warning(
                    f"FRED returned no observations for {series_id}. "
                    f"Series may be discontinued or not yet published. Skipping."
                )
                return None

            # Parse latest observation
            latest_obs = observations[0]
            latest_value_str = latest_obs.get("value", ".")

            # FRED uses "." for missing/unreported values
            if latest_value_str == ".":
                logger.warning(
                    f"FRED latest observation for {series_id} is '.' (unreported). "
                    f"Using previous observation if available."
                )
                if len(observations) > 1:
                    latest_obs = observations[1]
                    latest_value_str = latest_obs.get("value", ".")
                if latest_value_str == ".":
                    return None

            latest_value = float(latest_value_str)

            # Parse previous observation for change direction
            previous_value = 0.0
            if len(observations) > 1:
                prev_value_str = observations[1].get("value", ".")
                if prev_value_str != ".":
                    previous_value = float(prev_value_str)

            # Parse observation date → IST
            obs_date_str = latest_obs.get("date", "")
            published_at = _parse_fred_date(obs_date_str)

            # Build advisor note
            impact_note = series_config.get("impact_note", "")
            advisor_note = _generate_economic_advisor_note(
                event_name=series_config["event_name"],
                series_id=series_id,
                value=latest_value,
                previous_value=previous_value,
                impact_note=impact_note,
            )

            event = EconomicEvent(
                event_name=series_config["event_name"],
                series_id=series_id,
                value=latest_value,
                previous_value=previous_value,
                impact_level=SERIES_IMPACT.get(series_id, MarketImpact.LOW),
                source="FRED",
                affected_markets=["NSE"],
                advisor_note=advisor_note,
                published_at=published_at,
            )

            # Cache for 60 minutes — macro data changes slowly
            cache.set(cache_key, event, ttl=3600)

            logger.info(
                f"{series_config['event_name']} ({series_id}): "
                f"{latest_value:.2f} (prev: {previous_value:.2f}) — "
                f"impact: {event.impact_level.value}, "
                f"direction: {event.change_direction.value}."
            )

            return event

    except DataFetchError:
        raise

    except httpx.TimeoutException:
        logger.warning(
            f"FRED API timeout for {series_id} after 15 seconds. "
            f"Skipping this indicator."
        )
        return None

    except httpx.HTTPError as e:
        logger.warning(
            f"FRED HTTP error for {series_id}: {type(e).__name__}: {e}. Skipping."
        )
        return None

    except (ValueError, KeyError) as e:
        logger.warning(
            f"Failed to parse FRED response for {series_id}: "
            f"{type(e).__name__}: {e}. Skipping."
        )
        return None

    except Exception as e:
        logger.warning(
            f"Unexpected error fetching FRED series {series_id}: "
            f"{type(e).__name__}: {e}. Skipping."
        )
        return None


def _parse_fred_date(date_str: str) -> datetime:
    """Parse FRED observation date string into IST datetime.

    FRED dates are in YYYY-MM-DD format (no time component).
    We set time to 00:00 IST since FRED publishes daily/monthly.

    Args:
        date_str: Date string like "2026-05-01".

    Returns:
        datetime in IST timezone.
    """
    if not date_str:
        return datetime.now(IST)

    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
        return parsed.replace(tzinfo=IST)
    except ValueError:
        logger.debug(f"Could not parse FRED date '{date_str}', using current IST time.")
        return datetime.now(IST)


async def fetch_economic_events(
    config: DataFetchConfig,
) -> list[EconomicEvent]:
    """Fetch all tracked macro-economic indicators from FRED.

    This is the primary entry point called by the pipeline (Step 5).
    Fetches all series defined in FRED_SERIES concurrently via
    asyncio.gather, respecting the ConcurrencySemaphore(10).

    Args:
        config: DataFetchConfig controlling max_economic_events.

    Returns:
        List of EconomicEvent objects, one per successfully fetched
        FRED series. May be empty if FRED is unreachable or API key
        is missing. Sorted by impact_level (high first).

    Raises:
        DataFetchError: Only if FRED_API_KEY is missing from .env.
            Individual series failures are handled gracefully.
    """
    settings = get_settings()

    if not settings.fred_api_key:
        raise DataFetchError(
            source="FRED",
            reason="FRED_API_KEY not found in .env file",
            suggestion=(
                "Add FRED_API_KEY=<your_key> to .env. "
                "Get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html"
            ),
        )

    # Check batch cache
    cache_key = "economic:batch"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(
            f"Returning {len(cached)} economic events from batch cache "
            f"(age: {cache.get_age(cache_key):.0f}s)."
        )
        return cached

    # Fetch all series concurrently
    tasks = [
        _fetch_single_series(series_id, series_config, settings.fred_api_key)
        for series_id, series_config in FRED_SERIES.items()
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    events: list[EconomicEvent] = []
    for i, result in enumerate(results):
        series_id = list(FRED_SERIES.keys())[i]
        if isinstance(result, DataFetchError):
            raise result  # API key errors propagate
        elif isinstance(result, Exception):
            logger.warning(
                f"FRED series {series_id} failed with "
                f"{type(result).__name__}: {result}. Skipping."
            )
        elif result is not None:
            events.append(result)

    # Sort by impact level (HIGH first, then MEDIUM, then LOW)
    impact_order = {MarketImpact.HIGH: 0, MarketImpact.MEDIUM: 1, MarketImpact.LOW: 2}
    events.sort(key=lambda e: impact_order.get(e.impact_level, 2))

    # Respect max_economic_events limit
    events = events[: config.max_economic_events]

    # Cache the batch result
    cache.set(cache_key, events, ttl=config.cache_ttl_events)

    logger.info(
        f"Economic data pipeline complete — {len(events)}/{len(FRED_SERIES)} "
        f"indicators fetched successfully. "
        f"High impact: {sum(1 for e in events if e.impact_level == MarketImpact.HIGH)}, "
        f"Medium: {sum(1 for e in events if e.impact_level == MarketImpact.MEDIUM)}, "
        f"Low: {sum(1 for e in events if e.impact_level == MarketImpact.LOW)}."
    )

    return events
