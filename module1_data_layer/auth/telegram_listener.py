"""
SwingAdvisorBot — Module 1: Data Layer
auth/telegram_listener.py — Telegram polling listener for Kite auth URLs

Polls Telegram for new messages every 30 seconds.
When a message containing a Kite redirect URL arrives,
automatically extracts the request_token and completes auth.

Uses Telegram Bot API getUpdates (long polling).
Runs as a background asyncio task — non-blocking.

Only processes messages from Vijay's chat_id (from .env).
Ignores all other messages for security.

Usage:
    from module1_data_layer.auth.telegram_listener import telegram_listener

    # Start as background task (call once at bot startup)
    asyncio.create_task(telegram_listener.start_listening())

    # Stop gracefully
    telegram_listener.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from module1_data_layer.auth.smart_auth import smart_auth

logger = logging.getLogger("swing_advisor.telegram_listener")

TELEGRAM_API_BASE = "https://api.telegram.org/bot"
POLL_INTERVAL = 30       # seconds between polls
ERROR_BACKOFF = 60       # seconds to wait after an error
LONG_POLL_TIMEOUT = 25   # Telegram long poll timeout


class TelegramListener:
    """Polls Telegram for incoming messages and processes Kite auth URLs.

    Runs as a background asyncio task. Detects Kite redirect URLs
    in incoming messages and automatically completes re-authentication.

    Security:
      - Only processes messages from TELEGRAM_CHAT_ID (Vijay).
      - Ignores messages from all other users/groups.
      - Never logs or stores full message content beyond auth URLs.

    Usage:
        listener = TelegramListener()
        asyncio.create_task(listener.start_listening())
    """

    def __init__(self) -> None:
        self._token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._last_update_id: int = 0
        self._running: bool = False

        if not self._token or not self._chat_id:
            logger.warning(
                "[TelegramListener] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID "
                "not set in .env. Listener will not start."
            )

    @property
    def is_running(self) -> bool:
        return self._running

    async def start_listening(self) -> None:
        """Start polling Telegram for new messages.

        Runs indefinitely as a background task.
        Polls every 30 seconds using long polling.
        Automatically processes Kite auth URLs.
        """
        if not self._token or not self._chat_id:
            logger.error(
                "[TelegramListener] Cannot start — missing Telegram credentials"
            )
            return

        self._running = True
        logger.info(
            "[TelegramListener] Started polling for Kite auth URLs "
            f"(chat_id: {self._chat_id})"
        )

        while self._running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._process_update(update)
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                logger.info("[TelegramListener] Polling cancelled")
                break
            except Exception as e:
                logger.error(
                    f"[TelegramListener] Error during polling: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(ERROR_BACKOFF)

        logger.info("[TelegramListener] Stopped")

    async def _get_updates(self) -> list[dict]:
        """Fetch new updates from Telegram Bot API.

        Uses long polling with 25s timeout for efficiency.
        Only fetches updates after last_update_id to avoid duplicates.

        Returns:
            List of update dicts from Telegram.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TELEGRAM_API_BASE}{self._token}/getUpdates",
                params={
                    "offset": self._last_update_id + 1,
                    "timeout": LONG_POLL_TIMEOUT,
                },
                timeout=LONG_POLL_TIMEOUT + 5,
            )
            data = response.json()

            if not data.get("ok"):
                logger.warning(
                    f"[TelegramListener] getUpdates failed: "
                    f"{data.get('description', 'unknown error')}"
                )
                return []

            updates = data.get("result", [])
            if updates:
                self._last_update_id = updates[-1]["update_id"]

            return updates

    async def _process_update(self, update: dict) -> None:
        """Process a single Telegram update.

        Only handles messages from Vijay's chat_id.
        Checks if the message contains a Kite auth URL.

        Args:
            update: Raw update dict from Telegram getUpdates.
        """
        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = str(message.get("chat", {}).get("id", ""))

        # Security: only process messages from Vijay
        if chat_id != self._chat_id:
            return

        if not text:
            return

        # Check if this is a Kite auth URL
        result = await smart_auth.handle_telegram_message(text)
        if result:
            logger.info(
                f"[TelegramListener] Auth URL processed: {result}"
            )

    def stop(self) -> None:
        """Stop the polling loop gracefully."""
        self._running = False
        logger.info("[TelegramListener] Stop requested")


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

telegram_listener = TelegramListener()
