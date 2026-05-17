"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
telegram/telegram_client.py — Telegram Bot API wrapper

Sends formatted messages to Vijay's Telegram.
Handles message splitting for long reports (>4096 chars).
Retries on failure. Never silently fails.

Uses httpx (same pattern as M2 Claude client).
No Telegram SDK — direct Bot API calls.

Usage:
    from module6_reports.telegram.telegram_client import telegram_client

    msg_id = await telegram_client.send("Hello Vijay!")
    msg_id = await telegram_client.send("<b>Bold</b> text", parse_mode="HTML")
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from module6_reports.config import (
    TELEGRAM_API_BASE,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TELEGRAM_RETRY_ATTEMPTS,
    TELEGRAM_RETRY_DELAY,
    TELEGRAM_SEND_TIMEOUT,
)

logger = logging.getLogger("swing_advisor.telegram")


class TelegramConfigError(Exception):
    """Raised when Telegram bot token or chat ID is missing."""
    pass


class TelegramSendError(Exception):
    """Raised when Telegram API rejects the message."""
    pass


class TelegramClient:
    """Sends formatted messages to Vijay's Telegram.

    Features:
      - Auto-splits messages over 4096 chars at newlines
      - Retries on transient failures (2 attempts)
      - Raises TelegramSendError on permanent failure
      - HTML parse mode by default

    Usage:
        client = TelegramClient()
        msg_id = await client.send("Hello!")
        msg_ids = await client.send_parts(["Part 1", "Part 2"])
    """

    def __init__(self) -> None:
        self._token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        if not self._token or not self._chat_id:
            raise TelegramConfigError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be in .env. "
                "Setup: search @BotFather in Telegram, create a bot, "
                "then get your chat_id from /getUpdates."
            )

        self._base_url = f"{TELEGRAM_API_BASE}{self._token}"

    @property
    def is_configured(self) -> bool:
        """Check if Telegram credentials are set."""
        return bool(self._token and self._chat_id)

    async def send(
        self,
        message: str,
        parse_mode: str = "HTML",
    ) -> int:
        """Send a message to Vijay's Telegram.

        Auto-splits if over 4096 characters.
        Returns the last message_id sent.

        Args:
            message: Text to send (HTML formatted).
            parse_mode: Telegram parse mode (HTML or Markdown).

        Returns:
            Telegram message_id of the last sent message.

        Raises:
            TelegramSendError: If send fails after retries.
        """
        if len(message) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            return await self._send_with_retry(message, parse_mode)

        # Split long messages
        parts = self._split_message(message)
        logger.info(
            f"[Telegram] Message too long ({len(message)} chars), "
            f"split into {len(parts)} parts"
        )

        last_id = 0
        for i, part in enumerate(parts, 1):
            last_id = await self._send_with_retry(part, parse_mode)
            logger.debug(f"[Telegram] Part {i}/{len(parts)} sent: msg_id={last_id}")
        return last_id

    async def send_parts(
        self,
        parts: list[str],
        parse_mode: str = "HTML",
    ) -> list[int]:
        """Send multiple message parts sequentially.

        Returns list of message_ids.
        """
        message_ids = []
        for part in parts:
            msg_id = await self.send(part, parse_mode)
            message_ids.append(msg_id)
        return message_ids

    async def _send_with_retry(
        self,
        message: str,
        parse_mode: str,
    ) -> int:
        """Send a single message with retry logic.

        Retries on transient errors (network, timeout, 5xx).
        Raises immediately on permanent errors (4xx except 429).

        Returns:
            Telegram message_id.
        """
        last_error = None

        for attempt in range(1, TELEGRAM_RETRY_ATTEMPTS + 1):
            try:
                return await self._send_single(message, parse_mode)
            except TelegramSendError as e:
                last_error = e
                error_str = str(e)

                # Don't retry on permanent errors
                if any(
                    phrase in error_str.lower()
                    for phrase in [
                        "chat not found",
                        "bot was blocked",
                        "not enough rights",
                        "can't parse",
                    ]
                ):
                    raise

                logger.warning(
                    f"[Telegram] Send attempt {attempt}/{TELEGRAM_RETRY_ATTEMPTS} "
                    f"failed: {e}"
                )

                if attempt < TELEGRAM_RETRY_ATTEMPTS:
                    await asyncio.sleep(TELEGRAM_RETRY_DELAY)

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = TelegramSendError(f"Network error: {e}")
                logger.warning(
                    f"[Telegram] Network error attempt {attempt}/{TELEGRAM_RETRY_ATTEMPTS}: {e}"
                )

                if attempt < TELEGRAM_RETRY_ATTEMPTS:
                    await asyncio.sleep(TELEGRAM_RETRY_DELAY)

        raise last_error or TelegramSendError("Send failed after all retries")

    async def _send_single(
        self,
        message: str,
        parse_mode: str,
    ) -> int:
        """Send a single message via Telegram Bot API.

        Returns message_id on success.
        Raises TelegramSendError on failure.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                },
                timeout=TELEGRAM_SEND_TIMEOUT,
            )

            data = response.json()

            if not data.get("ok"):
                error_desc = data.get("description", "Unknown error")
                error_code = data.get("error_code", response.status_code)
                raise TelegramSendError(
                    f"Telegram API error {error_code}: {error_desc}"
                )

            msg_id = data["result"]["message_id"]
            logger.debug(f"[Telegram] Sent message_id={msg_id}")
            return msg_id

    def _split_message(self, message: str) -> list[str]:
        """Split a long message into parts at newline boundaries.

        Keeps HTML tags intact by splitting at line breaks.
        Each part stays under TELEGRAM_MAX_MESSAGE_LENGTH.

        Args:
            message: Full message text to split.

        Returns:
            List of message parts, each under 4096 chars.
        """
        max_len = TELEGRAM_MAX_MESSAGE_LENGTH
        parts: list[str] = []
        current = ""

        for line in message.split("\n"):
            # If a single line exceeds max, force-split it
            if len(line) > max_len:
                if current.strip():
                    parts.append(current.strip())
                    current = ""
                # Split the long line at max_len boundaries
                for i in range(0, len(line), max_len):
                    parts.append(line[i : i + max_len])
                continue

            # Check if adding this line would exceed max
            if len(current) + len(line) + 1 > max_len:
                if current.strip():
                    parts.append(current.strip())
                current = line + "\n"
            else:
                current += line + "\n"

        if current.strip():
            parts.append(current.strip())

        return parts


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

def get_telegram_client() -> TelegramClient:
    """Get or create the Telegram client singleton.

    Lazy initialization to avoid import-time errors
    when TELEGRAM_BOT_TOKEN is not yet set.
    """
    global _telegram_client
    if _telegram_client is None:
        _telegram_client = TelegramClient()
    return _telegram_client


_telegram_client: TelegramClient | None = None
