"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
scheduler/report_scheduler.py — APScheduler-based job scheduler

Manages all scheduled jobs:
  5:50 AM  — Kite token validation (daily)
  8:50 AM  — Morning brief (Mon–Fri)
  9:15–3:30 — Watchlist monitoring every 3 min (Mon–Fri)
  4:30 PM  — Evening review (Mon–Fri)
  Sat 10 AM — Weekly summary

Uses APScheduler AsyncIOScheduler with IST timezone.

Usage:
    from module6_reports.scheduler.report_scheduler import report_scheduler

    await report_scheduler.start()
    await report_scheduler.stop()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from module6_reports.config import (
    EVENING_REVIEW_HOUR,
    EVENING_REVIEW_MINUTE,
    IST,
    KITE_REFRESH_HOUR,
    KITE_REFRESH_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MISFIRE_GRACE_EVENING,
    MISFIRE_GRACE_KITE,
    MISFIRE_GRACE_MORNING,
    MISFIRE_GRACE_WEEKLY,
    MORNING_BRIEF_HOUR,
    MORNING_BRIEF_MINUTE,
    MORNING_BRIEF_RETRY_MINUTES,
    WATCHLIST_INTERVAL_MINUTES,
    WEEKLY_SUMMARY_DAY,
    WEEKLY_SUMMARY_HOUR,
    WEEKLY_SUMMARY_MINUTE,
)

logger = logging.getLogger("swing_advisor.scheduler")


class ReportScheduler:
    """APScheduler wrapper for all M6 scheduled jobs.

    Jobs:
      1. kite_token     — 5:50 AM daily
      2. morning_brief  — 8:50 AM Mon–Fri
      3. watchlist      — every 3 min, 9:15–15:30 Mon–Fri
      4. evening_review — 4:30 PM Mon–Fri
      5. weekly_summary — Saturday 10:00 AM

    All times IST. Misfire grace ensures late runs still execute.
    """

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler(timezone=IST)
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def start(self) -> None:
        """Start the scheduler with all jobs registered.

        Idempotent — safe to call multiple times.
        """
        if self._is_running:
            logger.info("[Scheduler] Already running")
            return

        self._register_jobs()
        self._scheduler.start()
        self._is_running = True

        logger.info(
            "[Scheduler] Started with %d jobs",
            len(self._scheduler.get_jobs()),
        )
        for job in self._scheduler.get_jobs():
            logger.info(
                f"  {job.id}: next run {job.next_run_time}"
            )

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._is_running:
            return

        self._scheduler.shutdown(wait=False)
        self._is_running = False
        logger.info("[Scheduler] Stopped")

    def get_status(self) -> dict:
        """Get scheduler status and next run times for all jobs."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "next_run": (
                    job.next_run_time.isoformat()
                    if job.next_run_time
                    else None
                ),
            })
        return {
            "is_running": self._is_running,
            "jobs": jobs,
            "job_count": len(jobs),
        }

    def _register_jobs(self) -> None:
        """Register all scheduled jobs."""

        # 1. Kite token validation — 5:50 AM daily
        self._scheduler.add_job(
            _job_kite_token,
            CronTrigger(
                hour=KITE_REFRESH_HOUR,
                minute=KITE_REFRESH_MINUTE,
                timezone=IST,
            ),
            id="kite_token",
            name="Kite Token Validation",
            misfire_grace_time=MISFIRE_GRACE_KITE,
            replace_existing=True,
        )

        # 1b. Kite auth reminder — 8:00 AM Mon–Fri
        self._scheduler.add_job(
            _job_kite_auth_reminder,
            CronTrigger(
                hour=8,
                minute=0,
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="kite_auth_reminder",
            name="Kite Auth Reminder",
            misfire_grace_time=MISFIRE_GRACE_KITE,
            replace_existing=True,
        )

        # 2. Morning brief — 8:50 AM Mon–Fri
        self._scheduler.add_job(
            _job_morning_brief,
            CronTrigger(
                hour=MORNING_BRIEF_HOUR,
                minute=MORNING_BRIEF_MINUTE,
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="morning_brief",
            name="Morning Brief",
            misfire_grace_time=MISFIRE_GRACE_MORNING,
            replace_existing=True,
        )

        # 3. Watchlist monitoring — every 3 min, Mon–Fri 9:15–15:30
        self._scheduler.add_job(
            _job_watchlist_check,
            IntervalTrigger(
                minutes=WATCHLIST_INTERVAL_MINUTES,
                timezone=IST,
            ),
            id="watchlist_monitor",
            name="Watchlist Monitor",
            replace_existing=True,
        )

        # 4. Evening review — 4:30 PM Mon–Fri
        self._scheduler.add_job(
            _job_evening_review,
            CronTrigger(
                hour=EVENING_REVIEW_HOUR,
                minute=EVENING_REVIEW_MINUTE,
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="evening_review",
            name="Evening Review",
            misfire_grace_time=MISFIRE_GRACE_EVENING,
            replace_existing=True,
        )

        # 5. Weekly summary — Saturday 10:00 AM
        self._scheduler.add_job(
            _job_weekly_summary,
            CronTrigger(
                hour=WEEKLY_SUMMARY_HOUR,
                minute=WEEKLY_SUMMARY_MINUTE,
                day_of_week=WEEKLY_SUMMARY_DAY,
                timezone=IST,
            ),
            id="weekly_summary",
            name="Weekly Summary",
            misfire_grace_time=MISFIRE_GRACE_WEEKLY,
            replace_existing=True,
        )


# ═══════════════════════════════════════════════════════════
# JOB FUNCTIONS — Each wraps a pipeline with error alerting
# ═══════════════════════════════════════════════════════════


async def _job_kite_token() -> None:
    """Job: Validate Kite token at 5:50 AM."""
    logger.info("[Job] kite_token starting")
    try:
        from module6_reports.scheduler.kite_token_job import run_kite_token_job

        result = await run_kite_token_job()
        logger.info(f"[Job] kite_token done: {result['status']}")
    except Exception as e:
        logger.error(f"[Job] kite_token failed: {e}", exc_info=True)
        await _send_job_error("kite_token", str(e))


async def _job_kite_auth_reminder() -> None:
    """Job: Send auth reminder at 8:00 AM if still not authenticated."""
    logger.info("[Job] kite_auth_reminder starting")
    try:
        from module6_reports.scheduler.kite_token_job import run_kite_auth_reminder

        result = await run_kite_auth_reminder()
        logger.info(f"[Job] kite_auth_reminder done: {result['status']}")
    except Exception as e:
        logger.error(f"[Job] kite_auth_reminder failed: {e}", exc_info=True)


async def _job_morning_brief() -> None:
    """Job: Generate and send morning brief at 8:50 AM.

    On failure, schedules retries at +10 and +30 minutes.
    Loads setups into watchlist monitor after success.
    """
    logger.info("[Job] morning_brief starting")
    try:
        from module6_reports.reports.morning_brief import generate_morning_brief
        from module6_reports.telegram.message_formatter import message_formatter
        from module6_reports.telegram.telegram_client import get_telegram_client

        brief = await generate_morning_brief()

        if brief.error:
            logger.error(f"[Job] morning_brief had error: {brief.error}")
            await _retry_morning_brief(attempt=1)
            return

        # Format and send
        html = message_formatter.format_morning_brief(brief)
        client = get_telegram_client()
        msg_id = await client.send(html, parse_mode="HTML")

        brief.telegram_message_ids = [msg_id]
        brief.delivery_status = "sent"
        logger.info(f"[Job] morning_brief sent: msg_id={msg_id}")

        # Load setups into watchlist monitor
        if brief.top_setups:
            from module6_reports.alerts.watchlist_monitor import watchlist_monitor
            watchlist_monitor.load_setups(brief.top_setups)

    except Exception as e:
        logger.error(f"[Job] morning_brief failed: {e}", exc_info=True)
        await _send_job_error("morning_brief", str(e))
        await _retry_morning_brief(attempt=1)


async def _retry_morning_brief(attempt: int) -> None:
    """Schedule a retry for failed morning brief.

    Retry schedule from config: [10, 30] minutes after first attempt.
    """
    if attempt > len(MORNING_BRIEF_RETRY_MINUTES):
        logger.error("[Job] morning_brief — all retries exhausted")
        await _send_job_error(
            "morning_brief",
            f"All {len(MORNING_BRIEF_RETRY_MINUTES)} retries exhausted. "
            "Morning brief not delivered today.",
        )
        return

    delay_minutes = MORNING_BRIEF_RETRY_MINUTES[attempt - 1]
    logger.info(
        f"[Job] morning_brief retry {attempt} scheduled in {delay_minutes} min"
    )

    await asyncio.sleep(delay_minutes * 60)

    try:
        from module6_reports.reports.morning_brief import generate_morning_brief
        from module6_reports.telegram.message_formatter import message_formatter
        from module6_reports.telegram.telegram_client import get_telegram_client

        brief = await generate_morning_brief()

        if brief.error:
            logger.error(
                f"[Job] morning_brief retry {attempt} failed: {brief.error}"
            )
            await _retry_morning_brief(attempt + 1)
            return

        html = message_formatter.format_morning_brief(brief)
        client = get_telegram_client()
        msg_id = await client.send(html, parse_mode="HTML")
        logger.info(
            f"[Job] morning_brief retry {attempt} succeeded: msg_id={msg_id}"
        )

        if brief.top_setups:
            from module6_reports.alerts.watchlist_monitor import watchlist_monitor
            watchlist_monitor.load_setups(brief.top_setups)

    except Exception as e:
        logger.error(
            f"[Job] morning_brief retry {attempt} crashed: {e}",
            exc_info=True,
        )
        await _retry_morning_brief(attempt + 1)


async def _job_watchlist_check() -> None:
    """Job: Check watchlist prices every 3 minutes.

    Only runs during market hours (checked inside monitor).
    """
    try:
        from module6_reports.alerts.watchlist_monitor import watchlist_monitor

        alerts = await watchlist_monitor.check_and_alert()
        if alerts:
            logger.info(
                f"[Job] watchlist: {len(alerts)} alerts triggered"
            )
    except Exception as e:
        logger.error(f"[Job] watchlist failed: {e}")


async def _job_evening_review() -> None:
    """Job: Generate and send evening review at 4:30 PM."""
    logger.info("[Job] evening_review starting")
    try:
        from module6_reports.alerts.watchlist_monitor import watchlist_monitor
        from module6_reports.reports.evening_review import generate_evening_review
        from module6_reports.telegram.message_formatter import message_formatter
        from module6_reports.telegram.telegram_client import get_telegram_client

        review = await generate_evening_review()

        if review.error:
            logger.error(f"[Job] evening_review had error: {review.error}")
            await _send_job_error("evening_review", review.error)

        # Format and send (even with partial error — send what we have)
        html = message_formatter.format_evening_review(review)
        client = get_telegram_client()
        msg_id = await client.send(html, parse_mode="HTML")

        review.telegram_message_ids = [msg_id]
        review.delivery_status = "sent"
        logger.info(f"[Job] evening_review sent: msg_id={msg_id}")

        # Clear watchlist setups for the day
        watchlist_monitor.clear_setups()

    except Exception as e:
        logger.error(f"[Job] evening_review failed: {e}", exc_info=True)
        await _send_job_error("evening_review", str(e))


async def _job_weekly_summary() -> None:
    """Job: Generate and send weekly summary on Saturday 10 AM."""
    logger.info("[Job] weekly_summary starting")
    try:
        from module6_reports.alerts.alert_tracker import alert_tracker
        from module6_reports.reports.weekly_summary import generate_weekly_summary
        from module6_reports.telegram.message_formatter import message_formatter
        from module6_reports.telegram.telegram_client import get_telegram_client

        summary = await generate_weekly_summary()

        if summary.error:
            logger.error(f"[Job] weekly_summary had error: {summary.error}")
            await _send_job_error("weekly_summary", summary.error)

        html = message_formatter.format_weekly_summary(summary)
        client = get_telegram_client()
        msg_id = await client.send(html, parse_mode="HTML")

        summary.telegram_message_ids = [msg_id]
        summary.delivery_status = "sent"
        logger.info(f"[Job] weekly_summary sent: msg_id={msg_id}")

        # Cleanup old alert records
        alert_tracker.cleanup_old_alerts(days_to_keep=30)

    except Exception as e:
        logger.error(f"[Job] weekly_summary failed: {e}", exc_info=True)
        await _send_job_error("weekly_summary", str(e))


# ═══════════════════════════════════════════════════════════
# ERROR ALERTING
# ═══════════════════════════════════════════════════════════


async def _send_job_error(job_name: str, error_message: str) -> None:
    """Send error alert to Telegram when a job fails.

    Never silently fail — always try to notify Vijay.
    """
    try:
        from module6_reports.models import ErrorAlert
        from module6_reports.telegram.message_formatter import message_formatter
        from module6_reports.telegram.telegram_client import get_telegram_client

        error = ErrorAlert(
            error_source=f"scheduler/{job_name}",
            error_message=error_message,
            is_critical=job_name in ("morning_brief", "kite_token"),
        )

        html = message_formatter.format_error_alert(error)
        client = get_telegram_client()
        await client.send(html, parse_mode="HTML")

    except Exception as e:
        # Last resort — log it
        logger.critical(
            f"[Scheduler] CANNOT SEND ERROR ALERT for {job_name}: {e}. "
            f"Original error: {error_message}"
        )


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

report_scheduler = ReportScheduler()
