"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
scheduler/kite_token_job.py — Daily Kite token validation jobs

Jobs:
  5:50 AM  — run_kite_token_job(): Check token, send alert if expired.
  8:00 AM  — run_kite_auth_reminder(): Reminder if still not authed.

Uses SmartKiteAuth for Telegram-based re-auth flow.
User clicks login link → sends redirect URL → bot auto-completes auth.

Usage:
    from module6_reports.scheduler.kite_token_job import (
        run_kite_token_job,
        run_kite_auth_reminder,
    )

    await run_kite_token_job()
    await run_kite_auth_reminder()
"""

from __future__ import annotations

import logging
from datetime import datetime

from module6_reports.config import IST

logger = logging.getLogger("swing_advisor.kite_token_job")


async def run_kite_token_job() -> dict:
    """Validate Kite token and send smart Telegram alert if expired.

    Runs at 5:50 AM IST daily.
    Uses SmartKiteAuth which sends a clickable login link
    and auto-processes the redirect URL when user sends it back.

    Returns:
        dict with 'status' ('valid' or 'expired' or 'error').
    """
    logger.info("[KiteTokenJob] Starting daily token validation")

    result = {
        "status": "unknown",
        "validated_at": datetime.now(IST).isoformat(),
    }

    try:
        from module1_data_layer.auth.smart_auth import smart_auth

        is_valid = await smart_auth.check_and_alert()

        if is_valid:
            result["status"] = "valid"
            logger.info(
                "[KiteTokenJob] Token valid ✅ "
                "Morning brief will run normally."
            )
        else:
            result["status"] = "expired"
            logger.warning(
                "[KiteTokenJob] Token expired ⚠️ "
                "Telegram alert sent to Vijay. "
                "Waiting for re-authentication."
            )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"[KiteTokenJob] Validation failed: {e}")

        # Fallback: try to send error alert directly
        await _send_fallback_error_alert(str(e))

    return result


async def run_kite_auth_reminder() -> dict:
    """Send reminder if Kite token is still not authenticated.

    Runs at 8:00 AM IST weekdays.
    Gives Vijay 50 minutes before morning brief at 8:50 AM.

    Returns:
        dict with 'status' ('valid' or 'reminder_sent' or 'error').
    """
    logger.info("[KiteAuthReminder] Checking if auth is still pending")

    result = {
        "status": "unknown",
        "checked_at": datetime.now(IST).isoformat(),
    }

    try:
        from module1_data_layer.auth.smart_auth import smart_auth

        is_valid = await smart_auth.check_and_alert()

        if is_valid:
            result["status"] = "valid"
            logger.info("[KiteAuthReminder] Token valid — no reminder needed")
        else:
            result["status"] = "reminder_sent"

            # Send additional urgency reminder
            try:
                from module6_reports.telegram.telegram_client import (
                    get_telegram_client,
                )

                client = get_telegram_client()
                await client.send(
                    "⏰ <b>Reminder: Login needed in 50 mins</b>\n\n"
                    "Morning brief starts at 8:50 AM.\n"
                    f"Login: {smart_auth.get_login_url()}\n\n"
                    "After login, send the redirect URL here.",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.error(
                    f"[KiteAuthReminder] Failed to send reminder: {e}"
                )

            logger.warning(
                "[KiteAuthReminder] Token still expired — "
                "reminder sent to Vijay"
            )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"[KiteAuthReminder] Check failed: {e}")

    return result


async def _send_fallback_error_alert(error_message: str) -> None:
    """Send a basic Telegram alert when SmartAuth itself fails."""
    try:
        from module6_reports.telegram.telegram_client import get_telegram_client

        client = get_telegram_client()
        await client.send(
            "🚨 <b>Kite Auth System Error</b>\n\n"
            f"<code>{error_message[:500]}</code>\n\n"
            "Manual intervention may be needed.",
            parse_mode="HTML",
        )
        logger.info("[KiteTokenJob] Fallback error alert sent")
    except Exception as e:
        logger.error(
            f"[KiteTokenJob] Fallback alert failed: {e}. "
            f"Original: {error_message}"
        )
