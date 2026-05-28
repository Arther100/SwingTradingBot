"""
SwingAdvisorBot — Module 1: Data Layer
fetchers/nse_session_manager.py — NSE India session/cookie manager

NSE India's public APIs (FII/DII, corporate announcements) require
browser-like session cookies before they return data. A direct API
call without cookies returns HTTP 401 or an empty response.

How NSE authentication works:
  1. Visit https://www.nseindia.com (homepage) → server sets cookies
  2. Use those cookies on subsequent API calls → 200 OK with JSON
  3. Cookies expire after ~30 minutes of inactivity → re-visit homepage

This manager handles the cookie lifecycle automatically:
  - Fetches fresh cookies on first use
  - Auto-refreshes every 30 minutes
  - Used by FiiDiiFetcher and EarningsFetcher as a shared singleton

Why httpx and not requests:
  - Project uses httpx throughout (consistent with other fetchers)
  - async/await native (no asyncio.run() bridging needed)
  - Timeout and retry handling consistent with other fetchers

Headers explanation:
  - User-Agent: NSE blocks non-browser agents. Must look like Chrome.
  - Referer: NSE checks this for most API endpoints.
  - Accept: NSE returns JSON only when Accept includes application/json.
  - Accept-Encoding: Required for gzip responses (NSE compresses).

Usage:
  from module1_data_layer.fetchers.nse_session_manager import nse_session

  headers, cookies = await nse_session.get_session_context()
  async with httpx.AsyncClient(headers=headers, cookies=cookies) as client:
      resp = await client.get("https://www.nseindia.com/api/fiidiiTradeReact")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import httpx
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.fetchers.nse_session")

# NSE homepage — must be visited first to obtain cookies
NSE_HOME_URL = "https://www.nseindia.com"

# Browser-like headers required by NSE
_NSE_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
    "DNT": "1",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

# Cookie refresh interval in seconds (30 minutes)
_REFRESH_INTERVAL_SEC: int = 1800

# Timeout for homepage cookie fetch
_HOME_TIMEOUT_SEC: float = 15.0


class NseSessionManager:
    """Manages NSE India session cookies for public API access.

    NSE requires cookies from a homepage visit before any data API
    will respond. This manager maintains a live cookie jar and
    refreshes it every 30 minutes automatically.

    The manager is a singleton — all fetchers share one session.
    This prevents hammering the NSE homepage with unnecessary visits.

    Thread safety:
      Designed for asyncio single-event-loop use.
      Do not share across threads.

    Lifecycle:
      Lazy initialisation — cookies fetched on first get_session_context()
      call, not at import time. Safe to import at module level.
    """

    def __init__(self) -> None:
        self._cookies: dict[str, str] = {}
        self._last_refresh: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    async def get_session_context(self) -> tuple[dict[str, str], dict[str, str]]:
        """Return (headers, cookies) suitable for an httpx request to NSE APIs.

        Refreshes cookies automatically if they are stale (>30 min old).

        Returns:
            Tuple of (headers dict, cookies dict) — pass both to
            httpx.AsyncClient(headers=..., cookies=...) or
            client.get(..., headers=..., cookies=...).

        Example::

            headers, cookies = await nse_session.get_session_context()
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, cookies=cookies)
        """
        if self._needs_refresh():
            await self._refresh_cookies()
        return _NSE_HEADERS.copy(), dict(self._cookies)

    def is_initialised(self) -> bool:
        """Return True if the session has valid cookies."""
        return bool(self._cookies) and not self._needs_refresh()

    def invalidate(self) -> None:
        """Force a cookie refresh on the next request.

        Call this when an NSE API returns HTTP 401 or empty response,
        indicating the cookies have expired ahead of schedule.
        """
        self._last_refresh = None
        self._cookies.clear()
        logger.info("[NSE] Session invalidated — will refresh on next request.")

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _needs_refresh(self) -> bool:
        """Return True if cookies are absent or older than 30 minutes."""
        if not self._cookies or self._last_refresh is None:
            return True
        elapsed = (datetime.now(IST) - self._last_refresh).total_seconds()
        return elapsed > _REFRESH_INTERVAL_SEC

    async def _refresh_cookies(self) -> None:
        """Visit NSE homepage to obtain fresh session cookies.

        Sets self._cookies and self._last_refresh on success.
        Logs a warning (never raises) on failure — callers should
        handle empty-cookie situations gracefully.
        """
        logger.info("[NSE] Refreshing session cookies from NSE homepage …")
        try:
            async with httpx.AsyncClient(
                headers=_NSE_HEADERS,
                follow_redirects=True,
                timeout=_HOME_TIMEOUT_SEC,
            ) as client:
                response = await client.get(NSE_HOME_URL)
                response.raise_for_status()

                # httpx stores cookies on the response object
                new_cookies: dict[str, str] = dict(response.cookies)

                if not new_cookies:
                    logger.warning(
                        "[NSE] Homepage returned no cookies — "
                        "NSE may have changed its auth mechanism."
                    )
                else:
                    self._cookies = new_cookies
                    self._last_refresh = datetime.now(IST)
                    logger.info(
                        f"[NSE] Session refreshed ✅  "
                        f"({len(new_cookies)} cookies set at "
                        f"{self._last_refresh.strftime('%H:%M:%S IST')})"
                    )

        except httpx.TimeoutException:
            logger.warning(
                "[NSE] Session refresh timed out after "
                f"{_HOME_TIMEOUT_SEC}s — using stale/empty cookies."
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                f"[NSE] Session refresh HTTP error {exc.response.status_code} "
                f"— {exc.request.url}"
            )
        except Exception as exc:
            logger.warning(f"[NSE] Session refresh failed: {exc}")


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

nse_session: NseSessionManager = NseSessionManager()
"""Shared NSE session manager singleton.

Import and use this instance everywhere:
    from module1_data_layer.fetchers.nse_session_manager import nse_session
"""
