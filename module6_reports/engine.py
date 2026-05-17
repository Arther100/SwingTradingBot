"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
engine.py — M6 public API and lifecycle manager

Single entry point for all M6 operations.
Manages scheduler lifecycle (start/stop) and exposes
report generation + delivery as simple async methods.

Usage:
    from module6_reports.engine import report_engine

    # Start scheduler (all cron jobs)
    await report_engine.start()

    # Manual triggers
    result = await report_engine.send_morning_brief()
    result = await report_engine.send_evening_review()
    result = await report_engine.send_weekly_summary()
    await report_engine.send_error_alert("source", "message")
    await report_engine.send_message("Hello Vijay!")

    # Status
    status = report_engine.get_status()

    # Stop
    await report_engine.stop()
"""

from __future__ import annotations

import logging
from typing import Optional

from module6_reports.config import IST

logger = logging.getLogger("swing_advisor.m6_engine")


class ReportEngine:
    """Module 6 public API — reports, alerts, and scheduling.

    Lifecycle:
      1. Import: `from module6_reports.engine import report_engine`
      2. Start:  `await report_engine.start()` — registers all cron jobs
      3. Use:    manual triggers or let scheduler run automatically
      4. Stop:   `await report_engine.stop()` — graceful shutdown

    All methods are async. All methods catch errors internally.
    """

    def __init__(self) -> None:
        self._started = False

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return self._started

    # ═══════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════

    async def start(self) -> None:
        """Start the M6 scheduler with all cron jobs.

        Jobs:
          5:50 AM  — Kite token validation
          8:50 AM  — Morning brief (Mon–Fri)
          Every 3m — Watchlist monitor (9:15–3:30 Mon–Fri)
          4:30 PM  — Evening review (Mon–Fri)
          Sat 10AM — Weekly summary

        Idempotent — safe to call multiple times.
        """
        if self._started:
            logger.info("[M6 Engine] Already running")
            return

        from module6_reports.scheduler.report_scheduler import report_scheduler

        await report_scheduler.start()
        self._started = True
        logger.info("[M6 Engine] Started — all jobs registered")

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._started:
            return

        from module6_reports.scheduler.report_scheduler import report_scheduler

        await report_scheduler.stop()
        self._started = False
        logger.info("[M6 Engine] Stopped")

    # ═══════════════════════════════════════════════════════
    # MANUAL TRIGGERS
    # ═══════════════════════════════════════════════════════

    async def send_morning_brief(
        self,
        user_id: str = "XCU700",
        skip_claude: bool = False,
    ) -> dict:
        """Manually trigger morning brief generation and delivery.

        Returns dict with status, telegram_message_ids, error, etc.
        """
        from module6_reports.agents.report_agent import report_agent

        return await report_agent.generate_and_send_morning_brief(
            user_id=user_id,
            skip_claude=skip_claude,
        )

    async def send_evening_review(
        self,
        user_id: str = "XCU700",
        skip_claude: bool = False,
    ) -> dict:
        """Manually trigger evening review generation and delivery."""
        from module6_reports.agents.report_agent import report_agent

        return await report_agent.generate_and_send_evening_review(
            user_id=user_id,
            skip_claude=skip_claude,
        )

    async def send_weekly_summary(
        self,
        user_id: str = "XCU700",
        skip_claude: bool = False,
    ) -> dict:
        """Manually trigger weekly summary generation and delivery."""
        from module6_reports.agents.report_agent import report_agent

        return await report_agent.generate_and_send_weekly_summary(
            user_id=user_id,
            skip_claude=skip_claude,
        )

    async def send_error_alert(
        self,
        source: str,
        message: str,
        is_critical: bool = False,
    ) -> Optional[int]:
        """Send an error alert to Telegram.

        Returns telegram message_id or None.
        """
        from module6_reports.agents.report_agent import report_agent

        return await report_agent.send_error_alert(
            source=source,
            message=message,
            is_critical=is_critical,
        )

    async def send_message(
        self,
        message: str,
        parse_mode: str = "HTML",
    ) -> Optional[int]:
        """Send a custom message to Telegram.

        Returns telegram message_id or None.
        """
        from module6_reports.agents.report_agent import report_agent

        return await report_agent.send_custom_message(
            message=message,
            parse_mode=parse_mode,
        )

    # ═══════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════

    def get_status(self) -> dict:
        """Get M6 engine status.

        Returns dict with scheduler state, job info, and alert counts.
        """
        result = {
            "module": "M6 Reports & Alerts",
            "is_running": self._started,
            "scheduler": None,
            "today_alert_count": 0,
        }

        try:
            from module6_reports.scheduler.report_scheduler import report_scheduler
            result["scheduler"] = report_scheduler.get_status()
        except Exception:
            pass

        try:
            from module6_reports.alerts.alert_tracker import alert_tracker
            result["today_alert_count"] = alert_tracker.get_alert_count()
        except Exception:
            pass

        return result

    async def check_kite_token(self) -> dict:
        """Manually trigger Kite token validation.

        Returns dict with status ('valid', 'expired', 'error').
        """
        from module6_reports.scheduler.kite_token_job import run_kite_token_job

        return await run_kite_token_job()

    async def check_watchlist(self) -> list:
        """Manually trigger one watchlist check cycle.

        Returns list of WatchlistAlert objects triggered.
        """
        from module6_reports.alerts.watchlist_monitor import watchlist_monitor

        return await watchlist_monitor.check_and_alert()


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

report_engine = ReportEngine()
