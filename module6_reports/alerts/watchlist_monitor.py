"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
alerts/watchlist_monitor.py — Real-time watchlist price monitoring

Runs every 3 minutes during market hours (9:15–3:30 IST).
Checks if any morning setup stocks have entered their entry zone.
Sends instant Telegram alerts — NO Claude call (pure template).

Dedup: max one alert per ticker per alert_type per day.

Usage:
    from module6_reports.alerts.watchlist_monitor import watchlist_monitor

    alerts = await watchlist_monitor.check_and_alert()
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional

from zoneinfo import ZoneInfo

from module6_reports.config import (
    IST,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
)
from module6_reports.models import (
    AlertType,
    DeliveryStatus,
    ReportType,
    WatchlistAlert,
)

logger = logging.getLogger("swing_advisor.watchlist_monitor")


class WatchlistMonitor:
    """Monitors watchlist stocks for entry zone triggers.

    Lifecycle:
      1. Morning brief generates setups → load_setups() stores them
      2. Every 3 min during market hours → check_and_alert()
      3. Fetch live prices from M1 for setup tickers
      4. Compare price vs entry zone
      5. If in zone + not already alerted today → send alert
      6. Dedup via alert_tracker (SQLite)

    No Claude calls — watchlist alerts are template-based for speed.
    """

    def __init__(self) -> None:
        # Active setups from today's morning brief
        # {ticker: setup_dict} with entry zone, target, stop, shares
        self._active_setups: dict[str, dict] = {}

    def load_setups(self, setups: list) -> None:
        """Load today's setups from morning brief for monitoring.

        Called once after morning brief is generated.
        Accepts list of SetupSummary or TradeSetup objects.

        Args:
            setups: List of setup objects with ticker, entry_low/high,
                    target, stop_loss, shares, risk_rupees.
        """
        self._active_setups.clear()

        for setup in setups:
            ticker = getattr(setup, "ticker", None)
            if not ticker:
                continue

            self._active_setups[ticker] = {
                "entry_low": _to_decimal(
                    getattr(setup, "entry_low", None)
                    or getattr(setup, "entry_zone_low", None)
                ),
                "entry_high": _to_decimal(
                    getattr(setup, "entry_high", None)
                    or getattr(setup, "entry_zone_high", None)
                ),
                "target": _to_decimal(
                    getattr(setup, "target", None)
                    or getattr(setup, "target_price", None)
                ),
                "stop_loss": _to_decimal(
                    getattr(setup, "stop_loss", None)
                ),
                "shares": getattr(setup, "shares", 0)
                or getattr(setup, "position_size_shares", 0),
                "risk_rupees": _to_decimal(
                    getattr(setup, "risk_rupees", None)
                    or getattr(setup, "max_risk_rupees", None)
                ),
                "risk_reward": getattr(setup, "risk_reward", None)
                or getattr(setup, "risk_reward_ratio", None),
            }

        logger.info(
            f"[WatchlistMonitor] Loaded {len(self._active_setups)} setups: "
            f"{list(self._active_setups.keys())}"
        )

    def clear_setups(self) -> None:
        """Clear all active setups (end of day)."""
        self._active_setups.clear()
        logger.info("[WatchlistMonitor] Setups cleared")

    @property
    def active_tickers(self) -> list[str]:
        """Get list of tickers being monitored."""
        return list(self._active_setups.keys())

    async def check_and_alert(self) -> list[WatchlistAlert]:
        """Check live prices and send alerts for entry zone triggers.

        Called every 3 minutes during market hours.

        Returns:
            List of WatchlistAlert objects that were triggered and sent.
        """
        if not self._active_setups:
            logger.debug("[WatchlistMonitor] No active setups to monitor")
            return []

        if not _is_market_hours():
            logger.debug("[WatchlistMonitor] Outside market hours — skipping")
            return []

        # Fetch live prices for monitored tickers
        prices = await _fetch_live_prices(list(self._active_setups.keys()))
        if not prices:
            logger.warning("[WatchlistMonitor] No live prices fetched")
            return []

        triggered: list[WatchlistAlert] = []

        for ticker, setup in self._active_setups.items():
            current_price = prices.get(ticker)
            if current_price is None:
                continue

            # Check if price is in entry zone
            alert = self._check_entry_zone(ticker, current_price, setup)
            if alert:
                # Check dedup before sending
                if await _is_already_alerted(ticker, alert.alert_type):
                    logger.debug(
                        f"[WatchlistMonitor] {ticker} already alerted today — skip"
                    )
                    continue

                # Send via Telegram
                sent = await _send_alert(alert)
                if sent:
                    await _record_alert(ticker, alert.alert_type, sent)
                    alert.delivery_status = DeliveryStatus.SENT
                    triggered.append(alert)
                    logger.info(
                        f"[WatchlistMonitor] ALERT SENT: {ticker} @ "
                        f"₹{current_price} in entry zone"
                    )

        return triggered

    def _check_entry_zone(
        self,
        ticker: str,
        current_price: Decimal,
        setup: dict,
    ) -> Optional[WatchlistAlert]:
        """Check if current price is within entry zone.

        Returns WatchlistAlert if triggered, None otherwise.
        """
        entry_low = setup.get("entry_low")
        entry_high = setup.get("entry_high")

        if entry_low is None or entry_high is None:
            return None

        if entry_low <= current_price <= entry_high:
            return WatchlistAlert(
                report_type=ReportType.WATCHLIST_ALERT,
                alert_type=AlertType.ENTRY_ZONE,
                user_id="XCU700",
                ticker=ticker,
                current_price=current_price,
                entry_zone_low=entry_low,
                entry_zone_high=entry_high,
                target=setup.get("target"),
                stop_loss=setup.get("stop_loss"),
                shares=setup.get("shares", 0),
                risk_rupees=setup.get("risk_rupees"),
                risk_reward=str(setup.get("risk_reward", "")),
                delivery_status=DeliveryStatus.PENDING,
            )

        return None


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════


def _to_decimal(value) -> Optional[Decimal]:
    """Safely convert a value to Decimal."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _is_market_hours() -> bool:
    """Check if current time is within NSE market hours.

    Market hours: 9:15 AM to 3:30 PM IST, Mon–Fri.
    """
    now = datetime.now(IST)

    # Weekend check
    if now.weekday() >= 5:
        return False

    market_open = now.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE,
        second=0, microsecond=0,
    )
    market_close = now.replace(
        hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE,
        second=0, microsecond=0,
    )

    return market_open <= now <= market_close


async def _fetch_live_prices(tickers: list[str]) -> dict[str, Decimal]:
    """Fetch live prices for given tickers from M1.

    Returns dict of {ticker: Decimal(price)}.
    """
    try:
        from module1_data_layer.models import DataFetchConfig
        from module1_data_layer.pipeline import run_data_pipeline

        config = DataFetchConfig(
            tickers=tickers,
            max_stocks=len(tickers),
            max_news=0,
            max_events=0,
        )
        market_data = await run_data_pipeline(
            tickers=tickers,
            config=config,
        )

        if not market_data or not market_data.stocks:
            return {}

        return {
            s.ticker: Decimal(str(s.current_price))
            for s in market_data.stocks
        }
    except Exception as e:
        logger.error(f"[WatchlistMonitor] Price fetch failed: {e}")
        return {}


async def _is_already_alerted(ticker: str, alert_type: AlertType) -> bool:
    """Check if this ticker+alert_type was already sent today.

    Uses alert_tracker (SQLite dedup).
    """
    try:
        from module6_reports.alerts.alert_tracker import alert_tracker

        today = datetime.now(IST).strftime("%Y-%m-%d")
        return alert_tracker.has_alert(
            ticker=ticker,
            alert_type=alert_type.value,
            date=today,
        )
    except Exception as e:
        logger.warning(f"[WatchlistMonitor] Dedup check failed: {e}")
        return False


async def _record_alert(
    ticker: str,
    alert_type: AlertType,
    telegram_message_id: int,
) -> None:
    """Record that an alert was sent (for dedup)."""
    try:
        from module6_reports.alerts.alert_tracker import alert_tracker

        today = datetime.now(IST).strftime("%Y-%m-%d")
        alert_tracker.record_alert(
            ticker=ticker,
            alert_type=alert_type.value,
            date=today,
            telegram_message_id=telegram_message_id,
        )
    except Exception as e:
        logger.warning(f"[WatchlistMonitor] Alert record failed: {e}")


async def _send_alert(alert: WatchlistAlert) -> Optional[int]:
    """Format and send a watchlist alert via Telegram.

    Returns telegram message_id on success, None on failure.
    """
    try:
        from module6_reports.telegram.message_formatter import message_formatter
        from module6_reports.telegram.telegram_client import get_telegram_client

        # Format alert message (template — no Claude)
        html = message_formatter.format_watchlist_alert(alert)
        alert.telegram_text = html

        # Send
        client = get_telegram_client()
        msg_id = await client.send(html, parse_mode="HTML")
        return msg_id

    except Exception as e:
        logger.error(f"[WatchlistMonitor] Alert send failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

watchlist_monitor = WatchlistMonitor()
