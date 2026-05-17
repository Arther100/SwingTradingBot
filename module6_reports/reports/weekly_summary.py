"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
reports/weekly_summary.py — Weekly summary generator

Runs Saturday 10 AM IST. Aggregates the week's trades,
P&L, lessons taught, and generates outlook for next week.

One Claude call — 1800 tokens max (1000 input + 800 output).

Usage:
    from module6_reports.reports.weekly_summary import generate_weekly_summary

    summary = await generate_weekly_summary(user_id="XCU700")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from zoneinfo import ZoneInfo

from module6_reports.config import (
    CLAUDE_MODEL,
    IST,
    LESSON_CONCEPTS,
    LESSON_SUMMARIES,
    WEEKLY_SUMMARY_SYSTEM_PROMPT,
    WEEKLY_SUMMARY_TOKEN_BUDGET,
)
from module6_reports.models import (
    DeliveryStatus,
    PositionSummary,
    ReportType,
    WeeklySummary,
)

logger = logging.getLogger("swing_advisor.weekly_summary")


async def generate_weekly_summary(
    user_id: str = "XCU700",
    skip_claude: bool = False,
) -> WeeklySummary:
    """Generate the weekly performance summary.

    Pipeline:
      1. Calculate week date range (Mon–Fri of the past week)
      2. M5 — Load all trades for the week
      3. Aggregate trade stats (opened, closed, won, lost, P&L)
      4. Load open positions carried forward
      5. Determine lessons taught this week
      6. Claude — Generate telegram_text (one call, 1800 tokens)

    Args:
        user_id: User identifier (default: XCU700 for Vijay).
        skip_claude: If True, skip Claude call.

    Returns:
        WeeklySummary with all fields populated.
    """
    start_ms = time.time()

    summary = WeeklySummary(
        report_type=ReportType.WEEKLY_SUMMARY,
        user_id=user_id,
        delivery_status=DeliveryStatus.PENDING,
    )

    try:
        # ── Step 1: Week date range ──
        monday, friday = _get_week_range()
        summary.week_start = monday.strftime("%Y-%m-%d")
        summary.week_end = friday.strftime("%Y-%m-%d")
        logger.info(
            f"[WeeklySummary] Week: {summary.week_start} to {summary.week_end}"
        )

        # ── Step 2: Load trades for the week ──
        all_trades = _load_week_trades(user_id)

        # ── Step 3: Aggregate trade stats ──
        week_trades = _filter_week_trades(all_trades, monday, friday)
        stats = _aggregate_trade_stats(week_trades)
        summary.trades_opened = stats["trades_opened"]
        summary.trades_closed = stats["trades_closed"]
        summary.winning_trades = stats["winning_trades"]
        summary.losing_trades = stats["losing_trades"]
        summary.week_pnl = stats["week_pnl"]
        if stats["trades_closed"] > 0:
            summary.win_rate = (
                stats["winning_trades"] / stats["trades_closed"] * 100
            )
        logger.info(
            f"[WeeklySummary] Stats: {stats['trades_opened']} opened, "
            f"{stats['trades_closed']} closed, P&L=₹{stats['week_pnl']}"
        )

        # ── Step 4: Open positions carried forward ──
        summary.open_positions = _get_open_positions(all_trades)

        # ── Step 5: Lessons taught this week ──
        summary.lessons_taught = _get_week_lessons(monday, friday)

        # ── Step 6: Claude telegram_text ──
        if not skip_claude:
            telegram_text = await _generate_telegram_text(summary)
            if telegram_text:
                summary.telegram_text = telegram_text

    except Exception as e:
        logger.error(f"[WeeklySummary] Pipeline error: {e}", exc_info=True)
        summary.error = str(e)

    elapsed_ms = int((time.time() - start_ms) * 1000)
    summary.generation_time_ms = elapsed_ms
    logger.info(f"[WeeklySummary] Generated in {elapsed_ms}ms")

    return summary


# ═══════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════


def _get_week_range() -> tuple[datetime, datetime]:
    """Get Monday and Friday of the past trading week.

    If today is Saturday, returns this past Mon–Fri.
    If today is Sunday, returns this past Mon–Fri.
    If today is a weekday, returns previous Mon–Fri.
    """
    now = datetime.now(IST)
    # Days since Monday (Mon=0, Sat=5, Sun=6)
    weekday = now.weekday()

    if weekday == 5:  # Saturday — this week
        monday = now - timedelta(days=5)
    elif weekday == 6:  # Sunday — this week
        monday = now - timedelta(days=6)
    else:  # Weekday — previous week
        monday = now - timedelta(days=weekday + 7)

    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    friday = monday + timedelta(days=4, hours=23, minutes=59, seconds=59)

    return monday, friday


def _load_week_trades(user_id: str) -> list:
    """Load all trades from M5 memory engine.

    Returns list of TradeRecord objects.
    """
    try:
        from module5_memory.engine import memory_engine

        trades = memory_engine.get_trade_history(user_id=user_id, limit=200)
        return trades
    except Exception as e:
        logger.warning(f"[WeeklySummary] M5 trade load failed: {e}")
        return []


def _filter_week_trades(
    trades: list,
    monday: datetime,
    friday: datetime,
) -> list:
    """Filter trades that were opened or closed during the week.

    A trade counts for this week if:
      - entry_date is within Mon–Fri, OR
      - exit_date is within Mon–Fri
    """
    week_trades = []
    for trade in trades:
        entry_in_week = (
            hasattr(trade, "entry_date")
            and trade.entry_date
            and monday <= trade.entry_date <= friday
        )
        exit_in_week = (
            hasattr(trade, "exit_date")
            and trade.exit_date
            and monday <= trade.exit_date <= friday
        )
        if entry_in_week or exit_in_week:
            week_trades.append(trade)
    return week_trades


def _aggregate_trade_stats(week_trades: list) -> dict:
    """Aggregate trade statistics for the week.

    Returns dict with trades_opened, trades_closed,
    winning_trades, losing_trades, week_pnl.
    """
    trades_opened = 0
    trades_closed = 0
    winning_trades = 0
    losing_trades = 0
    week_pnl = Decimal("0")

    now = datetime.now(IST)

    for trade in week_trades:
        # Count opened this week
        if hasattr(trade, "entry_date") and trade.entry_date:
            trades_opened += 1

        # Count closed this week
        if trade.status in ("closed", "stopped_out"):
            trades_closed += 1
            if trade.pnl_rupees is not None:
                week_pnl += trade.pnl_rupees
                if trade.pnl_rupees > 0:
                    winning_trades += 1
                elif trade.pnl_rupees < 0:
                    losing_trades += 1

    return {
        "trades_opened": trades_opened,
        "trades_closed": trades_closed,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "week_pnl": week_pnl,
    }


def _get_open_positions(trades: list) -> list[PositionSummary]:
    """Get currently open positions for carry-forward display."""
    positions = []
    for trade in trades:
        if trade.status == "open":
            positions.append(
                PositionSummary(
                    ticker=trade.ticker,
                    entry_price=trade.entry_price,
                    shares=trade.shares,
                    stop_loss=getattr(trade, "stop_loss", None),
                    target=getattr(trade, "target_price", None),
                )
            )
    return positions


def _get_week_lessons(
    monday: datetime,
    friday: datetime,
) -> list[str]:
    """Get the lesson concepts taught during the week.

    Uses the same day-of-year rotation as morning brief.
    Returns list of concept names for Mon–Fri.
    """
    lessons = []
    current = monday
    while current <= friday:
        if current.weekday() < 5:  # Mon–Fri only
            day_of_year = current.timetuple().tm_yday
            index = day_of_year % len(LESSON_CONCEPTS)
            concept = LESSON_CONCEPTS[index]
            if concept not in lessons:
                lessons.append(concept)
        current += timedelta(days=1)
    return lessons


# ═══════════════════════════════════════════════════════════
# CLAUDE TELEGRAM TEXT GENERATION
# ═══════════════════════════════════════════════════════════


async def _generate_telegram_text(summary: WeeklySummary) -> Optional[str]:
    """Call Claude to generate the telegram_text for weekly summary.

    One call, 1800 tokens max (1000 input + 800 output).
    """
    try:
        import httpx

        api_key = _get_anthropic_key()
        if not api_key:
            logger.warning("[WeeklySummary] No ANTHROPIC_API_KEY — skipping Claude")
            return None

        user_prompt = _build_claude_prompt(summary)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": WEEKLY_SUMMARY_TOKEN_BUDGET["output_budget"],
                    "system": WEEKLY_SUMMARY_SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=30.0,
            )

            data = response.json()

            if response.status_code != 200:
                error = data.get("error", {}).get("message", "Unknown error")
                logger.error(f"[WeeklySummary] Claude API error: {error}")
                return None

            content_blocks = data.get("content", [])
            text_parts = [
                block["text"]
                for block in content_blocks
                if block.get("type") == "text"
            ]
            telegram_text = "\n".join(text_parts)

            usage = data.get("usage", {})
            summary.tokens_used = (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            )
            logger.info(
                f"[WeeklySummary] Claude: {usage.get('input_tokens', 0)} in + "
                f"{usage.get('output_tokens', 0)} out = {summary.tokens_used} tokens"
            )

            return telegram_text

    except Exception as e:
        logger.error(f"[WeeklySummary] Claude call failed: {e}")
        return None


def _build_claude_prompt(summary: WeeklySummary) -> str:
    """Build Claude prompt for weekly summary.

    Structured sections — total ~1000 input tokens.
    """
    sections: list[str] = []

    # Week range
    sections.append("== WEEK ==")
    sections.append(f"{summary.week_start} to {summary.week_end}")

    # Performance
    sections.append("\n== PERFORMANCE ==")
    sections.append(f"Trades opened: {summary.trades_opened}")
    sections.append(f"Trades closed: {summary.trades_closed}")
    if summary.trades_closed > 0:
        sections.append(f"Won: {summary.winning_trades}")
        sections.append(f"Lost: {summary.losing_trades}")
        if summary.win_rate is not None:
            sections.append(f"Win rate: {summary.win_rate:.0f}%")
    if summary.week_pnl is not None:
        sign = "+" if summary.week_pnl >= 0 else ""
        sections.append(f"Week P&L: {sign}₹{summary.week_pnl}")

    # Open positions
    sections.append("\n== POSITIONS CARRIED FORWARD ==")
    if summary.open_positions:
        for p in summary.open_positions:
            sections.append(
                f"  {p.ticker}: {p.shares} shares @ ₹{p.entry_price}"
            )
    else:
        sections.append("  None — fully in cash")

    # Lessons
    if summary.lessons_taught:
        sections.append("\n== LESSONS THIS WEEK ==")
        for concept in summary.lessons_taught:
            display = concept.replace("_", " ").title()
            lesson_text = LESSON_SUMMARIES.get(concept, "")
            sections.append(f"  {display}: {lesson_text[:100]}")

    # Instruction
    sections.append("\n== INSTRUCTION ==")
    sections.append(
        "Write Vijay's weekly review. "
        "Celebrate wins, be constructive about losses. "
        "Highlight the best and worst trade if any closed. "
        "List lessons covered. Give honest outlook for next week. "
        "Use HTML formatting (<b>bold</b>). Keep under 800 tokens."
    )

    return "\n".join(sections)


def _get_anthropic_key() -> str:
    """Get Anthropic API key from environment."""
    import os
    return os.getenv("ANTHROPIC_API_KEY", "")
