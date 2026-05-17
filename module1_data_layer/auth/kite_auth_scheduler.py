"""
SwingAdvisorBot — Module 1: Data Layer
auth/kite_auth_scheduler.py — Daily token validation scheduler

Kite Connect tokens expire daily. This scheduler runs at 5:50 AM IST
(before market opens at 9:15 AM) to validate the current token and
alert the user via Telegram if re-authentication is needed.

Why 5:50 AM?
  → Market pre-open starts at 9:00 AM IST.
  → This gives the user ~3 hours to complete the browser login flow.
  → All data fetches depend on a valid token — catch it early.

Why not auto-refresh?
  → Kite Connect requires a browser-based login (OAuth2 redirect).
  → No headless refresh is possible without Zerodha credentials in code.
  → The scheduler validates + alerts. The user completes the login.

Telegram alerts:
  → Success: "✅ Kite token is valid. Ready for trading."
  → Failure: "🚨 Kite token expired. Re-authenticate before market opens."
  → Uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env.

Usage:
    from module1_data_layer.auth.kite_auth_scheduler import token_scheduler
    token_scheduler.start()  # Call once at bot startup
    # Scheduler runs in background thread, non-blocking.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from module1_data_layer.auth.kite_auth import kite_auth_manager

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.kite_auth_scheduler")


class KiteTokenScheduler:
    """Daily Kite token validation scheduler.

    Runs at 5:50 AM IST every day. Validates the current access token.
    Sends a Telegram alert on success or failure.

    Usage:
        scheduler = KiteTokenScheduler()
        scheduler.start()   # Start background scheduler
        scheduler.stop()    # Shutdown gracefully
        scheduler.run_now() # Manual trigger for testing
    """

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(
            timezone=IST,
            job_defaults={"misfire_grace_time": 3600},
        )
        self._telegram_bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
        self._telegram_chat_id: str | None = os.getenv("TELEGRAM_CHAT_ID")

    def start(self) -> None:
        """Start the background scheduler.

        Adds the daily validation job at 5:50 AM IST.
        Non-blocking — runs in a daemon thread.
        """
        self._scheduler.add_job(
            func=self._validate_and_alert,
            trigger=CronTrigger(hour=5, minute=50, timezone=IST),
            id="kite_token_validation",
            name="Daily Kite Token Validation (5:50 AM IST)",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "KiteTokenScheduler started. "
            "Token validation runs daily at 5:50 AM IST."
        )

    def stop(self) -> None:
        """Shutdown the scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("KiteTokenScheduler stopped.")

    def run_now(self) -> bool:
        """Manually trigger token validation + alert.

        Useful for testing or on-demand checks.

        Returns:
            True if token is valid, False if expired.
        """
        return self._validate_and_alert()

    def _validate_and_alert(self) -> bool:
        """Core job: validate token and send Telegram alert.

        Returns:
            True if token is valid, False if expired.
        """
        timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        logger.info(f"[TokenScheduler] Running token validation at {timestamp}")

        is_valid = kite_auth_manager.validate_token()

        if is_valid:
            msg = (
                f"✅ Kite token is valid.\n"
                f"Client: {kite_auth_manager.client_id}\n"
                f"Time: {timestamp}\n"
                f"Ready for trading."
            )
            logger.info(f"[TokenScheduler] Token valid. Client: {kite_auth_manager.client_id}")
        else:
            login_url = kite_auth_manager.get_login_url()
            msg = (
                f"🚨 Kite token EXPIRED!\n"
                f"Time: {timestamp}\n"
                f"Re-authenticate before market opens (9:15 AM).\n\n"
                f"Login URL:\n{login_url}"
            )
            logger.warning(f"[TokenScheduler] Token expired! Alert sent.")

        self._send_telegram_alert(msg)
        return is_valid

    def _send_telegram_alert(self, message: str) -> None:
        """Send a Telegram message via Bot API.

        Silently logs if Telegram credentials are missing —
        the scheduler still works without Telegram.
        """
        if not self._telegram_bot_token or not self._telegram_chat_id:
            logger.warning(
                "[TokenScheduler] Telegram credentials not configured. "
                "Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env. "
                "Skipping alert."
            )
            return

        url = f"https://api.telegram.org/bot{self._telegram_bot_token}/sendMessage"

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    url,
                    json={
                        "chat_id": self._telegram_chat_id,
                        "text": message,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                if response.status_code == 200:
                    logger.info("[TokenScheduler] Telegram alert sent successfully.")
                else:
                    logger.warning(
                        f"[TokenScheduler] Telegram API returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
        except httpx.HTTPError as e:
            logger.warning(
                f"[TokenScheduler] Failed to send Telegram alert: {e}. "
                f"Check network and bot token."
            )


# Module-level singleton
token_scheduler = KiteTokenScheduler()
