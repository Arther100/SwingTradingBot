"""
SwingAdvisorBot — Module 1: Data Layer
auth/smart_auth.py — Smart Kite authentication with Telegram integration

Automates the daily Kite re-authentication flow:
  1. Scheduler checks token at 5:50 AM IST.
  2. If expired → sends Telegram alert with login link.
  3. User clicks link → logs in → sends redirect URL to Telegram bot.
  4. Bot extracts request_token → exchanges for access_token.
  5. Saves to .env → confirms via Telegram → morning brief runs.

Total user effort: ~60 seconds instead of 5+ minutes.

SEBI regulations require daily manual browser login. We cannot skip
the browser step. But we can automate everything after it.

Usage:
    from module1_data_layer.auth.smart_auth import smart_auth

    # Scheduler calls this at 5:50 AM
    is_valid = await smart_auth.check_and_alert()

    # Telegram listener calls this when user sends redirect URL
    success = await smart_auth.process_redirect_url(url)

    # Telegram listener calls this for every incoming message
    result = await smart_auth.handle_telegram_message(text)
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from module1_data_layer.auth.kite_auth import kite_auth_manager

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.smart_auth")

# Regex to extract request_token from Kite redirect URL
REQUEST_TOKEN_PATTERN = re.compile(r"request_token=([A-Za-z0-9]+)")


class SmartKiteAuth:
    """Smart Kite authentication with Telegram-based re-auth flow.

    Wraps KiteAuthManager and adds:
      - Telegram alerts when token expires
      - Automatic extraction of request_token from redirect URLs
      - One-message re-auth: user sends URL, bot completes the rest
      - Auth state tracking (pending/complete)

    Flow:
      5:50 AM → check_and_alert() → token OK? done. expired? alert.
      User sends redirect URL → handle_telegram_message() detects it
        → process_redirect_url() → extract token → exchange → save → confirm.
    """

    def __init__(self) -> None:
        self._auth_pending: bool = False

    @property
    def auth_pending(self) -> bool:
        """Whether re-authentication is pending."""
        return self._auth_pending

    async def check_and_alert(self) -> bool:
        """Check token validity and send Telegram alert if expired.

        Called by scheduler at 5:50 AM IST daily.

        Steps:
          1. Try kite_auth_manager.validate_token()
          2. If valid → log success, return True
          3. If expired → generate login URL, send Telegram alert,
             set auth_pending = True, return False

        Returns:
            True if token is valid, False if expired (alert sent).
        """
        logger.info("[SmartAuth] Checking Kite token validity")

        is_valid = kite_auth_manager.validate_token()

        if is_valid:
            self._auth_pending = False
            logger.info("[SmartAuth] Token valid ✓ — ready for trading")
            return True

        # Token expired — send Telegram alert
        self._auth_pending = True
        logger.warning("[SmartAuth] Token expired — sending Telegram alert")

        try:
            alert_html = self.generate_telegram_alert()
            await self._send_telegram(alert_html)
            logger.info("[SmartAuth] Telegram alert sent to Vijay")
        except Exception as e:
            logger.error(f"[SmartAuth] Failed to send Telegram alert: {e}")

        return False

    async def process_redirect_url(self, url: str) -> bool:
        """Process a Kite redirect URL to complete re-authentication.

        Called when user sends the redirect URL to Telegram bot.

        Steps:
          1. Parse request_token from URL
          2. Exchange for access_token via kite_auth_manager
          3. Reload .env with updated token
          4. Verify new token works
          5. Send Telegram confirmation
          6. Set auth_pending = False

        Args:
            url: The redirect URL containing request_token parameter.

        Returns:
            True if auth succeeded, False if failed.
        """
        logger.info("[SmartAuth] Processing redirect URL for auth")

        # Step 1: Extract request_token
        match = REQUEST_TOKEN_PATTERN.search(url)
        if not match:
            logger.error("[SmartAuth] No request_token found in URL")
            await self._send_telegram(
                "❌ <b>Auth Failed</b>\n\n"
                "Could not find <code>request_token</code> in the URL.\n"
                "Make sure you copied the full redirect URL."
            )
            return False

        request_token = match.group(1)
        logger.info(
            f"[SmartAuth] Extracted request_token: {request_token[:8]}..."
        )

        # Step 2: Exchange for access_token
        try:
            access_token = kite_auth_manager.set_access_token_from_request_token(
                request_token
            )
        except Exception as e:
            logger.error(f"[SmartAuth] Token exchange failed: {e}")
            await self._send_telegram(
                "❌ <b>Auth Failed</b>\n\n"
                f"Token exchange error: {e}\n\n"
                "The request_token may have expired (valid ~2 minutes).\n"
                f"Try again: {self.get_login_url()}"
            )
            return False

        # Step 3: Force reload into running process immediately
        os.environ["KITE_ACCESS_TOKEN"] = access_token
        load_dotenv(override=True)  # reload all .env vars

        logger.info(
            f"[SmartAuth] Token saved to .env and "
            f"reloaded into process environment. "
            f"New token: {access_token[:10]}..."
        )

        # Verify it worked
        verify = os.getenv("KITE_ACCESS_TOKEN")
        if verify == access_token:
            logger.info("[SmartAuth] Token reload verified ✅")
        else:
            logger.error("[SmartAuth] Token reload FAILED ❌")

        # Step 4: Verify the new token works
        is_valid = kite_auth_manager.validate_token()
        if not is_valid:
            logger.error("[SmartAuth] New token failed validation")
            await self._send_telegram(
                "⚠️ <b>Auth Warning</b>\n\n"
                "Token was exchanged but validation failed.\n"
                "Try the login flow again."
            )
            return False

        # Step 5: Send confirmation
        self._auth_pending = False
        now = datetime.now(IST).strftime("%I:%M %p")
        await self._send_telegram(
            f"✅ <b>Auth Complete</b>\n\n"
            f"Kite session is active at {now} IST.\n"
            f"Bot is ready. Morning brief will run normally. ⚡"
        )

        logger.info("[SmartAuth] Re-authentication complete ✓")
        return True

    async def handle_telegram_message(self, text: str) -> str | None:
        """Check if an incoming Telegram message is a Kite auth URL.

        Called by TelegramListener for every incoming message.
        Detects redirect URLs containing request_token and
        automatically processes them.

        Args:
            text: The message text from Telegram.

        Returns:
            Confirmation string if auth was processed, None otherwise.
        """
        if not text:
            return None

        # Detect Kite auth redirect URLs
        is_auth_url = (
            "request_token=" in text
            or "127.0.0.1" in text and "request_token" in text
            or "kite.zerodha" in text and "request_token" in text
        )

        if not is_auth_url:
            return None

        logger.info("[SmartAuth] Auth URL detected in Telegram message")
        success = await self.process_redirect_url(text)

        if success:
            return "Auth complete — bot is ready."
        return "Auth failed — check logs."

    def get_login_url(self) -> str:
        """Get the Kite Connect login URL.

        Returns:
            Formatted Kite login URL with API key.
        """
        return kite_auth_manager.get_login_url()

    def generate_telegram_alert(self) -> str:
        """Generate the HTML-formatted Telegram alert for expired token.

        Returns:
            HTML string ready for Telegram sendMessage (under 4096 chars).
        """
        login_url = self.get_login_url()

        return (
            "🔐 <b>Kite Daily Login Required</b>\n\n"
            "Your Zerodha session has expired.\n"
            "Morning brief starts at 8:50 AM.\n\n"
            f"<b>Step 1:</b> Click to login:\n"
            f"{login_url}\n\n"
            "<b>Step 2:</b> After login, forward the "
            "redirect URL to this chat.\n"
            "It looks like:\n"
            "<code>http://127.0.0.1:8000/?action=login"
            "&amp;request_token=xxx&amp;status=success</code>\n\n"
            "I'll handle the rest automatically. ⚡"
        )

    async def _send_telegram(self, message: str) -> None:
        """Send a message via Telegram.

        Uses the existing TelegramClient from M6.
        Logs error but doesn't raise — auth flow shouldn't
        crash if Telegram is temporarily down.
        """
        try:
            from module6_reports.telegram.telegram_client import get_telegram_client

            client = get_telegram_client()
            await client.send(message, parse_mode="HTML")
        except Exception as e:
            logger.error(f"[SmartAuth] Telegram send failed: {e}")


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

smart_auth = SmartKiteAuth()
