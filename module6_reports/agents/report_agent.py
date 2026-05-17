"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
agents/report_agent.py — CrewAI report orchestration agent

Provides a CrewAI-compatible agent that orchestrates report generation
and delivery. Wraps the report generators and Telegram client
behind a unified agent interface.

Can be invoked by CrewAI crews or called directly.

Usage:
    from module6_reports.agents.report_agent import report_agent

    # Direct call
    result = await report_agent.generate_and_send_morning_brief()
    result = await report_agent.generate_and_send_evening_review()
    result = await report_agent.generate_and_send_weekly_summary()
    result = await report_agent.send_error_alert(source, message)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from zoneinfo import ZoneInfo

from module6_reports.config import IST
from module6_reports.models import (
    DeliveryStatus,
    ErrorAlert,
    ReportType,
)

logger = logging.getLogger("swing_advisor.report_agent")


class ReportAgent:
    """Orchestrates report generation → formatting → delivery.

    Single responsibility: take a report type, run the generator,
    format the output, send via Telegram, return status.

    Each method returns a dict with:
      - status: 'sent' | 'failed' | 'partial'
      - report_type: ReportType value
      - telegram_message_ids: list[int]
      - error: Optional[str]
      - generation_time_ms: int
    """

    async def generate_and_send_morning_brief(
        self,
        user_id: str = "XCU700",
        skip_claude: bool = False,
    ) -> dict:
        """Generate morning brief and send to Telegram.

        Returns result dict with delivery status.
        """
        logger.info("[ReportAgent] Starting morning brief")

        try:
            from module6_reports.reports.morning_brief import generate_morning_brief
            from module6_reports.telegram.message_formatter import message_formatter
            from module6_reports.telegram.telegram_client import get_telegram_client

            # Generate
            brief = await generate_morning_brief(
                user_id=user_id,
                skip_claude=skip_claude,
            )

            # Format
            html = message_formatter.format_morning_brief(brief)

            # Send
            client = get_telegram_client()
            msg_id = await client.send(html, parse_mode="HTML")

            brief.telegram_message_ids = [msg_id]
            brief.delivery_status = DeliveryStatus.SENT

            # Load setups into watchlist
            if brief.top_setups:
                from module6_reports.alerts.watchlist_monitor import watchlist_monitor
                watchlist_monitor.load_setups(brief.top_setups)

            return {
                "status": "sent",
                "report_type": ReportType.MORNING_BRIEF.value,
                "telegram_message_ids": [msg_id],
                "error": brief.error,
                "generation_time_ms": brief.generation_time_ms,
                "setups_count": len(brief.top_setups),
                "vix_gate": brief.vix_gate,
            }

        except Exception as e:
            logger.error(f"[ReportAgent] Morning brief failed: {e}", exc_info=True)
            await self.send_error_alert("report_agent/morning_brief", str(e))
            return {
                "status": "failed",
                "report_type": ReportType.MORNING_BRIEF.value,
                "telegram_message_ids": [],
                "error": str(e),
                "generation_time_ms": 0,
            }

    async def generate_and_send_evening_review(
        self,
        user_id: str = "XCU700",
        skip_claude: bool = False,
    ) -> dict:
        """Generate evening review and send to Telegram."""
        logger.info("[ReportAgent] Starting evening review")

        try:
            from module6_reports.alerts.watchlist_monitor import watchlist_monitor
            from module6_reports.reports.evening_review import generate_evening_review
            from module6_reports.telegram.message_formatter import message_formatter
            from module6_reports.telegram.telegram_client import get_telegram_client

            review = await generate_evening_review(
                user_id=user_id,
                skip_claude=skip_claude,
            )

            html = message_formatter.format_evening_review(review)
            client = get_telegram_client()
            msg_id = await client.send(html, parse_mode="HTML")

            review.telegram_message_ids = [msg_id]
            review.delivery_status = DeliveryStatus.SENT

            # Clear watchlist for the day
            watchlist_monitor.clear_setups()

            return {
                "status": "sent",
                "report_type": ReportType.EVENING_REVIEW.value,
                "telegram_message_ids": [msg_id],
                "error": review.error,
                "generation_time_ms": review.generation_time_ms,
            }

        except Exception as e:
            logger.error(f"[ReportAgent] Evening review failed: {e}", exc_info=True)
            await self.send_error_alert("report_agent/evening_review", str(e))
            return {
                "status": "failed",
                "report_type": ReportType.EVENING_REVIEW.value,
                "telegram_message_ids": [],
                "error": str(e),
                "generation_time_ms": 0,
            }

    async def generate_and_send_weekly_summary(
        self,
        user_id: str = "XCU700",
        skip_claude: bool = False,
    ) -> dict:
        """Generate weekly summary and send to Telegram."""
        logger.info("[ReportAgent] Starting weekly summary")

        try:
            from module6_reports.reports.weekly_summary import generate_weekly_summary
            from module6_reports.telegram.message_formatter import message_formatter
            from module6_reports.telegram.telegram_client import get_telegram_client

            summary = await generate_weekly_summary(
                user_id=user_id,
                skip_claude=skip_claude,
            )

            html = message_formatter.format_weekly_summary(summary)
            client = get_telegram_client()
            msg_id = await client.send(html, parse_mode="HTML")

            summary.telegram_message_ids = [msg_id]
            summary.delivery_status = DeliveryStatus.SENT

            # Cleanup old alerts
            from module6_reports.alerts.alert_tracker import alert_tracker
            alert_tracker.cleanup_old_alerts(days_to_keep=30)

            return {
                "status": "sent",
                "report_type": ReportType.WEEKLY_SUMMARY.value,
                "telegram_message_ids": [msg_id],
                "error": summary.error,
                "generation_time_ms": summary.generation_time_ms,
            }

        except Exception as e:
            logger.error(f"[ReportAgent] Weekly summary failed: {e}", exc_info=True)
            await self.send_error_alert("report_agent/weekly_summary", str(e))
            return {
                "status": "failed",
                "report_type": ReportType.WEEKLY_SUMMARY.value,
                "telegram_message_ids": [],
                "error": str(e),
                "generation_time_ms": 0,
            }

    async def send_error_alert(
        self,
        source: str,
        message: str,
        is_critical: bool = False,
    ) -> Optional[int]:
        """Send an error alert to Telegram.

        Never raises — logs failure and returns None.

        Args:
            source: Which module/step failed.
            message: Human-readable error description.
            is_critical: Whether this blocks all operations.

        Returns:
            Telegram message_id or None if send failed.
        """
        try:
            from module6_reports.telegram.message_formatter import message_formatter
            from module6_reports.telegram.telegram_client import get_telegram_client

            error = ErrorAlert(
                error_source=source,
                error_message=message,
                is_critical=is_critical,
            )

            html = message_formatter.format_error_alert(error)
            client = get_telegram_client()
            msg_id = await client.send(html, parse_mode="HTML")

            logger.info(
                f"[ReportAgent] Error alert sent: {source} — msg_id={msg_id}"
            )
            return msg_id

        except Exception as e:
            logger.critical(
                f"[ReportAgent] CANNOT SEND ERROR ALERT: {e}. "
                f"Original: {source} — {message}"
            )
            return None

    async def send_custom_message(
        self,
        message: str,
        parse_mode: str = "HTML",
    ) -> Optional[int]:
        """Send a custom message to Telegram.

        Useful for ad-hoc notifications or testing.

        Returns:
            Telegram message_id or None on failure.
        """
        try:
            from module6_reports.telegram.telegram_client import get_telegram_client

            client = get_telegram_client()
            return await client.send(message, parse_mode=parse_mode)
        except Exception as e:
            logger.error(f"[ReportAgent] Custom message failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

report_agent = ReportAgent()
