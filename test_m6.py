"""
SwingAdvisorBot — test_m6.py
Module 6 test suite — Reports, Alerts, and Telegram delivery.

Tests (no live API calls unless noted):
  1. Models — MorningBrief, EveningReview, WeeklySummary, WatchlistAlert creation
  2. Config — All config constants loaded correctly
  3. Message Formatter — HTML output for all report types
  4. Alert Tracker — SQLite dedup (create, check, record, cleanup)
  5. Watchlist Monitor — Load setups, entry zone detection
  6. Telegram Client — Send real message to Vijay (LIVE)
  7. Engine — Status check, scheduler init
  8. Lesson Rotation — Day-of-year based lesson selection

Usage:
    python test_m6.py

Prerequisites:
    - .env with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
    - python seed_vijay.py (for M5 profile)
"""

import asyncio
import os
import sys
import tempfile
import traceback
from datetime import datetime
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv(override=True)


# ─────────────────────────────────────────────────────────────
# Results tracker
# ─────────────────────────────────────────────────────────────

results = {}


def mark(step: str, passed: bool, error: str = ""):
    results[step] = (passed, error)


# ─────────────────────────────────────────────────────────────
# Test 1 — Models
# ─────────────────────────────────────────────────────────────

def test_models():
    print("=" * 58)
    print("  TEST 1 — M6 Models")
    print("=" * 58)

    try:
        from module6_reports.models import (
            AlertRecord,
            AlertType,
            DeliveryStatus,
            ErrorAlert,
            EveningReview,
            LessonOfDay,
            MorningBrief,
            PositionSummary,
            ReportType,
            SetupSummary,
            WatchlistAlert,
            WeeklySummary,
        )

        # MorningBrief
        brief = MorningBrief(
            user_id="XCU700",
            india_vix=18.33,
            vix_signal="moderate_fear",
            nifty_value=22500.0,
            nifty_change_pct=0.45,
            market_mood="cautiously_bullish",
            vix_gate="open",
            total_capital=Decimal("50000"),
        )
        assert brief.report_type == ReportType.MORNING_BRIEF
        assert brief.delivery_status == DeliveryStatus.PENDING

        # SetupSummary
        setup = SetupSummary(
            ticker="HDFCBANK",
            entry_low=Decimal("1650"),
            entry_high=Decimal("1670"),
            target=Decimal("1800"),
            stop_loss=Decimal("1600"),
            confidence=7.5,
            shares=6,
            risk_reward="1:2.3",
        )
        assert setup.ticker == "HDFCBANK"

        # EveningReview
        review = EveningReview(user_id="XCU700")
        assert review.report_type == ReportType.EVENING_REVIEW

        # WeeklySummary
        summary = WeeklySummary(user_id="XCU700")
        assert summary.report_type == ReportType.WEEKLY_SUMMARY

        # WatchlistAlert
        alert = WatchlistAlert(
            ticker="RELIANCE",
            current_price=Decimal("2800"),
            entry_zone_low=Decimal("2780"),
            entry_zone_high=Decimal("2820"),
        )
        assert alert.alert_type == AlertType.ENTRY_ZONE

        # ErrorAlert
        err = ErrorAlert(
            error_source="test",
            error_message="Test error",
        )
        assert not err.is_critical

        # AlertRecord
        record = AlertRecord(
            alert_id="test-001",
            ticker="HDFCBANK",
            alert_type=AlertType.ENTRY_ZONE,
            date="2026-05-15",
        )
        assert record.date == "2026-05-15"

        print("  ✅ All 7 models created and validated")
        mark("Models", True)

    except Exception as e:
        print(f"  ❌ {e}")
        traceback.print_exc()
        mark("Models", False, str(e))


# ─────────────────────────────────────────────────────────────
# Test 2 — Config
# ─────────────────────────────────────────────────────────────

def test_config():
    print("\n" + "=" * 58)
    print("  TEST 2 — M6 Config")
    print("=" * 58)

    try:
        from module6_reports.config import (
            CLAUDE_MODEL,
            DEFAULT_TICKERS,
            EVENING_REVIEW_SYSTEM_PROMPT,
            EVENING_REVIEW_TOKEN_BUDGET,
            LESSON_CONCEPTS,
            LESSON_SUMMARIES,
            MORNING_BRIEF_HOUR,
            MORNING_BRIEF_MINUTE,
            MORNING_BRIEF_RETRY_MINUTES,
            MORNING_BRIEF_SYSTEM_PROMPT,
            MORNING_BRIEF_TOKEN_BUDGET,
            TELEGRAM_MAX_MESSAGE_LENGTH,
            WATCHLIST_INTERVAL_MINUTES,
            WEEKLY_SUMMARY_SYSTEM_PROMPT,
            WEEKLY_SUMMARY_TOKEN_BUDGET,
        )

        assert MORNING_BRIEF_HOUR == 8
        assert MORNING_BRIEF_MINUTE == 50
        assert TELEGRAM_MAX_MESSAGE_LENGTH == 4096
        assert WATCHLIST_INTERVAL_MINUTES == 3
        assert len(DEFAULT_TICKERS) == 10
        assert len(LESSON_CONCEPTS) == 10
        assert len(LESSON_SUMMARIES) == 10
        assert MORNING_BRIEF_TOKEN_BUDGET["grand_total"] == 2630
        assert EVENING_REVIEW_TOKEN_BUDGET["grand_total"] == 1500
        assert WEEKLY_SUMMARY_TOKEN_BUDGET["grand_total"] == 1800
        assert MORNING_BRIEF_RETRY_MINUTES == [10, 30]
        assert "claude" in CLAUDE_MODEL
        assert "Vijay" in MORNING_BRIEF_SYSTEM_PROMPT
        assert "Vijay" in EVENING_REVIEW_SYSTEM_PROMPT
        assert "Vijay" in WEEKLY_SUMMARY_SYSTEM_PROMPT

        print(f"  ✅ Schedule: morning {MORNING_BRIEF_HOUR}:{MORNING_BRIEF_MINUTE:02d}")
        print(f"  ✅ Tickers: {DEFAULT_TICKERS[:3]}... ({len(DEFAULT_TICKERS)} total)")
        print(f"  ✅ Token budgets: morning={MORNING_BRIEF_TOKEN_BUDGET['grand_total']}, "
              f"evening={EVENING_REVIEW_TOKEN_BUDGET['grand_total']}, "
              f"weekly={WEEKLY_SUMMARY_TOKEN_BUDGET['grand_total']}")
        print(f"  ✅ Lessons: {len(LESSON_CONCEPTS)} concepts with summaries")
        print(f"  ✅ Claude model: {CLAUDE_MODEL}")
        mark("Config", True)

    except Exception as e:
        print(f"  ❌ {e}")
        traceback.print_exc()
        mark("Config", False, str(e))


# ─────────────────────────────────────────────────────────────
# Test 3 — Message Formatter
# ─────────────────────────────────────────────────────────────

def test_message_formatter():
    print("\n" + "=" * 58)
    print("  TEST 3 — Message Formatter")
    print("=" * 58)

    try:
        from module6_reports.models import (
            ErrorAlert,
            EveningReview,
            LessonOfDay,
            MorningBrief,
            PositionSummary,
            SetupSummary,
            WatchlistAlert,
            WeeklySummary,
        )
        from module6_reports.telegram.message_formatter import message_formatter

        # Morning brief with setups
        brief = MorningBrief(
            user_id="XCU700",
            india_vix=18.33,
            vix_signal="moderate_fear",
            nifty_value=22500.0,
            nifty_change_pct=0.45,
            market_mood="cautiously_bullish",
            mood_confidence=0.72,
            vix_gate="open",
            total_capital=Decimal("50000"),
            available_capital=Decimal("40000"),
            top_setups=[
                SetupSummary(
                    ticker="HDFCBANK",
                    entry_low=Decimal("1650"),
                    entry_high=Decimal("1670"),
                    target=Decimal("1800"),
                    stop_loss=Decimal("1600"),
                    confidence=7.5,
                    shares=6,
                    risk_rupees=Decimal("420"),
                    reward_rupees=Decimal("780"),
                    risk_reward="1:1.86",
                    position_rupees=Decimal("10020"),
                ),
            ],
            open_positions=[
                PositionSummary(
                    ticker="RELIANCE",
                    entry_price=Decimal("2750"),
                    current_price=Decimal("2800"),
                    shares=3,
                    pnl_rupees=Decimal("150"),
                ),
            ],
            lesson_of_day=LessonOfDay(
                concept="stop_loss",
                summary="A stop loss is your exit price for a losing trade.",
            ),
        )

        html = message_formatter.format_morning_brief(brief)
        assert "Good morning Vijay" in html
        assert "HDFCBANK" in html
        assert "RELIANCE" in html
        assert "stop_loss" not in html  # Should be "Stop Loss" (formatted)
        assert len(html) < 4096
        print(f"  ✅ Morning brief: {len(html)} chars")

        # Evening review
        review = EveningReview(
            user_id="XCU700",
            nifty_close=22550.0,
            nifty_change_pct=0.22,
            india_vix=Decimal("17.80"),
            top_gainers=["HDFCBANK +2.1% (₹1680)"],
            top_losers=["AXISBANK -1.3% (₹1050)"],
        )
        html_ev = message_formatter.format_evening_review(review)
        assert "Evening Review" in html_ev
        assert "0.22%" in html_ev
        print(f"  ✅ Evening review: {len(html_ev)} chars")

        # Weekly summary
        summary = WeeklySummary(
            user_id="XCU700",
            week_start="2026-05-11",
            week_end="2026-05-15",
            trades_opened=3,
            trades_closed=2,
            winning_trades=1,
            losing_trades=1,
            week_pnl=Decimal("450"),
            win_rate=50.0,
            lessons_taught=["stop_loss", "position_sizing"],
        )
        html_ws = message_formatter.format_weekly_summary(summary)
        assert "Weekly Review" in html_ws
        assert "50%" in html_ws
        print(f"  ✅ Weekly summary: {len(html_ws)} chars")

        # Watchlist alert
        alert = WatchlistAlert(
            ticker="HDFCBANK",
            current_price=Decimal("1660"),
            entry_zone_low=Decimal("1650"),
            entry_zone_high=Decimal("1670"),
            target=Decimal("1800"),
            stop_loss=Decimal("1600"),
            shares=6,
            risk_rupees=Decimal("420"),
        )
        html_wa = message_formatter.format_watchlist_alert(alert)
        assert "ENTRY ALERT" in html_wa
        assert "HDFCBANK" in html_wa
        assert "1,660" in html_wa or "1660" in html_wa
        print(f"  ✅ Watchlist alert: {len(html_wa)} chars")

        # Error alert
        error = ErrorAlert(
            error_source="kite_token_job",
            error_message="Token expired",
            is_critical=True,
        )
        html_err = message_formatter.format_error_alert(error)
        assert "SwingAdvisorBot Alert" in html_err
        assert "kite_token_job" in html_err
        assert "🔴" in html_err  # Critical
        print(f"  ✅ Error alert: {len(html_err)} chars")

        mark("Message Formatter", True)

    except Exception as e:
        print(f"  ❌ {e}")
        traceback.print_exc()
        mark("Message Formatter", False, str(e))


# ─────────────────────────────────────────────────────────────
# Test 4 — Alert Tracker (SQLite dedup)
# ─────────────────────────────────────────────────────────────

def test_alert_tracker():
    print("\n" + "=" * 58)
    print("  TEST 4 — Alert Tracker (SQLite Dedup)")
    print("=" * 58)

    try:
        from module6_reports.alerts.alert_tracker import AlertTracker

        # Use temp DB for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            test_db = f.name

        try:
            tracker = AlertTracker(db_path=test_db)

            # Initially no alerts
            assert not tracker.has_alert("HDFCBANK", "entry_zone", "2026-05-15")
            print("  ✅ No alerts initially")

            # Record an alert
            tracker.record_alert(
                ticker="HDFCBANK",
                alert_type="entry_zone",
                date="2026-05-15",
                telegram_message_id=42,
            )

            # Now it should exist
            assert tracker.has_alert("HDFCBANK", "entry_zone", "2026-05-15")
            print("  ✅ Alert recorded and found")

            # Different ticker — not found
            assert not tracker.has_alert("RELIANCE", "entry_zone", "2026-05-15")
            print("  ✅ Different ticker not found (dedup works)")

            # Same ticker, different date — not found
            assert not tracker.has_alert("HDFCBANK", "entry_zone", "2026-05-16")
            print("  ✅ Different date not found (daily dedup)")

            # Duplicate insert — should not error (INSERT OR IGNORE)
            tracker.record_alert(
                ticker="HDFCBANK",
                alert_type="entry_zone",
                date="2026-05-15",
                telegram_message_id=99,
            )
            assert tracker.get_alert_count("2026-05-15") == 1
            print("  ✅ Duplicate insert handled (INSERT OR IGNORE)")

            # Cleanup old alerts
            tracker.record_alert("TCS", "entry_zone", "2025-01-01", 1)
            deleted = tracker.cleanup_old_alerts(days_to_keep=30)
            assert deleted >= 1
            print(f"  ✅ Cleanup: {deleted} old alert(s) deleted")

            tracker.close()

        finally:
            os.unlink(test_db)

        mark("Alert Tracker", True)

    except Exception as e:
        print(f"  ❌ {e}")
        traceback.print_exc()
        mark("Alert Tracker", False, str(e))


# ─────────────────────────────────────────────────────────────
# Test 5 — Watchlist Monitor
# ─────────────────────────────────────────────────────────────

def test_watchlist_monitor():
    print("\n" + "=" * 58)
    print("  TEST 5 — Watchlist Monitor")
    print("=" * 58)

    try:
        from module6_reports.alerts.watchlist_monitor import WatchlistMonitor
        from module6_reports.models import SetupSummary

        monitor = WatchlistMonitor()

        # Load setups
        setups = [
            SetupSummary(
                ticker="HDFCBANK",
                entry_low=Decimal("1650"),
                entry_high=Decimal("1670"),
                target=Decimal("1800"),
                stop_loss=Decimal("1600"),
                confidence=7.5,
                shares=6,
            ),
            SetupSummary(
                ticker="RELIANCE",
                entry_low=Decimal("2780"),
                entry_high=Decimal("2820"),
                target=Decimal("3000"),
                stop_loss=Decimal("2700"),
                confidence=6.8,
                shares=3,
            ),
        ]

        monitor.load_setups(setups)
        assert len(monitor.active_tickers) == 2
        assert "HDFCBANK" in monitor.active_tickers
        assert "RELIANCE" in monitor.active_tickers
        print(f"  ✅ Loaded {len(monitor.active_tickers)} setups")

        # Test entry zone detection — in zone
        alert = monitor._check_entry_zone(
            "HDFCBANK", Decimal("1660"), monitor._active_setups["HDFCBANK"]
        )
        assert alert is not None
        assert alert.ticker == "HDFCBANK"
        assert alert.current_price == Decimal("1660")
        print("  ✅ Entry zone detected: HDFCBANK @ ₹1,660")

        # Test entry zone — below zone
        alert_below = monitor._check_entry_zone(
            "HDFCBANK", Decimal("1640"), monitor._active_setups["HDFCBANK"]
        )
        assert alert_below is None
        print("  ✅ No alert below zone: HDFCBANK @ ₹1,640")

        # Test entry zone — above zone
        alert_above = monitor._check_entry_zone(
            "HDFCBANK", Decimal("1680"), monitor._active_setups["HDFCBANK"]
        )
        assert alert_above is None
        print("  ✅ No alert above zone: HDFCBANK @ ₹1,680")

        # Test boundary — exactly at entry_low
        alert_low = monitor._check_entry_zone(
            "RELIANCE", Decimal("2780"), monitor._active_setups["RELIANCE"]
        )
        assert alert_low is not None
        print("  ✅ Alert at zone boundary: RELIANCE @ ₹2,780")

        # Clear setups
        monitor.clear_setups()
        assert len(monitor.active_tickers) == 0
        print("  ✅ Setups cleared")

        mark("Watchlist Monitor", True)

    except Exception as e:
        print(f"  ❌ {e}")
        traceback.print_exc()
        mark("Watchlist Monitor", False, str(e))


# ─────────────────────────────────────────────────────────────
# Test 6 — Telegram Client (LIVE)
# ─────────────────────────────────────────────────────────────

async def test_telegram_send():
    print("\n" + "=" * 58)
    print("  TEST 6 — Telegram Send (LIVE)")
    print("=" * 58)

    try:
        from module6_reports.telegram.telegram_client import get_telegram_client

        client = get_telegram_client()

        msg = (
            "🧪 <b>M6 Test Suite</b>\n\n"
            "Module 6 (Reports & Alerts) test message.\n"
            "If you see this, Telegram delivery is working.\n\n"
            f"<i>Sent at {datetime.now().strftime('%H:%M:%S IST')}</i>"
        )

        msg_id = await client.send(msg, parse_mode="HTML")
        assert msg_id > 0
        print(f"  ✅ Message sent: msg_id={msg_id}")

        mark("Telegram Send", True)

    except Exception as e:
        print(f"  ❌ {e}")
        traceback.print_exc()
        mark("Telegram Send", False, str(e))


# ─────────────────────────────────────────────────────────────
# Test 7 — Engine Status
# ─────────────────────────────────────────────────────────────

def test_engine_status():
    print("\n" + "=" * 58)
    print("  TEST 7 — Engine Status")
    print("=" * 58)

    try:
        from module6_reports.engine import report_engine

        status = report_engine.get_status()
        assert status["module"] == "M6 Reports & Alerts"
        assert status["is_running"] is False  # Not started yet
        print(f"  ✅ Module: {status['module']}")
        print(f"  ✅ Running: {status['is_running']}")
        print(f"  ✅ Today alerts: {status['today_alert_count']}")

        mark("Engine Status", True)

    except Exception as e:
        print(f"  ❌ {e}")
        traceback.print_exc()
        mark("Engine Status", False, str(e))


# ─────────────────────────────────────────────────────────────
# Test 8 — Lesson Rotation
# ─────────────────────────────────────────────────────────────

def test_lesson_rotation():
    print("\n" + "=" * 58)
    print("  TEST 8 — Lesson Rotation")
    print("=" * 58)

    try:
        from module6_reports.config import LESSON_CONCEPTS, LESSON_SUMMARIES

        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")

        day_of_year = datetime.now(IST).timetuple().tm_yday
        index = day_of_year % len(LESSON_CONCEPTS)
        concept = LESSON_CONCEPTS[index]
        summary = LESSON_SUMMARIES.get(concept, "")

        assert concept in LESSON_CONCEPTS
        assert len(summary) > 20
        concept_display = concept.replace("_", " ").title()
        print(f"  ✅ Today's lesson (day {day_of_year}): {concept_display}")
        print(f"  ✅ Summary: {summary[:60]}...")

        # Verify all concepts have summaries
        for c in LESSON_CONCEPTS:
            assert c in LESSON_SUMMARIES, f"Missing summary for {c}"
        print(f"  ✅ All {len(LESSON_CONCEPTS)} concepts have summaries")

        mark("Lesson Rotation", True)

    except Exception as e:
        print(f"  ❌ {e}")
        traceback.print_exc()
        mark("Lesson Rotation", False, str(e))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main():
    print("\n" + "╔" + "═" * 56 + "╗")
    print("║" + "  SwingAdvisorBot — Module 6 Test Suite".center(56) + "║")
    print("║" + "  Reports, Alerts & Telegram Delivery".center(56) + "║")
    print("╚" + "═" * 56 + "╝\n")

    # Sync tests
    test_models()
    test_config()
    test_message_formatter()
    test_alert_tracker()
    test_watchlist_monitor()

    # Async tests
    await test_telegram_send()

    # More sync tests
    test_engine_status()
    test_lesson_rotation()

    # ── Summary ──
    print("\n" + "=" * 58)
    print("  RESULTS SUMMARY")
    print("=" * 58)

    passed = 0
    failed = 0
    for step, (ok, err) in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {step}")
        if not ok and err:
            print(f"     Error: {err[:80]}")
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n  {passed}/{total} PASSED", end="")
    if failed > 0:
        print(f"  |  {failed} FAILED")
    else:
        print("  ✅ ALL TESTS PASSED")

    print()
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
