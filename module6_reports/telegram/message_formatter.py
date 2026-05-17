"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
telegram/message_formatter.py — HTML message formatting for Telegram

Converts M6 model objects (MorningBrief, EveningReview, etc.)
into Telegram-ready HTML strings.

All formatters produce messages under 4096 chars.
If a morning brief exceeds the limit, it returns a list of parts.

Telegram HTML supports: <b>, <i>, <code>, <pre>, <a href="">.
No <br> — use \n for line breaks.

Usage:
    from module6_reports.telegram.message_formatter import message_formatter

    html = message_formatter.format_morning_brief(brief)
    html = message_formatter.format_watchlist_alert(alert)
    html = message_formatter.format_error_alert(error)
"""

from __future__ import annotations

import logging
from decimal import Decimal

from module6_reports.models import (
    EveningReview,
    ErrorAlert,
    MorningBrief,
    PositionSummary,
    SetupSummary,
    WatchlistAlert,
    WeeklySummary,
)

logger = logging.getLogger("swing_advisor.msg_formatter")


class MessageFormatter:
    """Formats M6 models into Telegram HTML messages.

    Each format method returns a string ready for TelegramClient.send().
    All messages are mobile-readable with clear sections.
    """

    # ═══════════════════════════════════════════════════════
    # MORNING BRIEF
    # ═══════════════════════════════════════════════════════

    def format_morning_brief(self, brief: MorningBrief) -> str:
        """Format a MorningBrief into Telegram HTML.

        If Claude generated telegram_text, use that directly.
        Otherwise, build from structured fields.

        Args:
            brief: MorningBrief model with all fields populated.

        Returns:
            HTML-formatted string for Telegram.
        """
        # If Claude already generated the message, use it
        if brief.telegram_text:
            return brief.telegram_text

        parts: list[str] = []

        # Header
        parts.append("🌅 <b>Good morning Vijay!</b>\n")
        parts.append("Markets open in 25 minutes. Here's your brief.\n")

        # Market mood section
        parts.append(self._format_market_section(brief))

        # Portfolio section
        parts.append(self._format_portfolio_section(brief))

        # Setups section
        parts.append(self._format_setups_section(brief))

        # Lesson of the day
        if brief.lesson_of_day:
            parts.append(self._format_lesson_section(brief))

        # Key events
        if brief.key_events:
            events_text = "\n".join(f"⚠️ {e}" for e in brief.key_events)
            parts.append(f"\n{events_text}\n")

        # Advisor note
        if brief.advisor_note:
            parts.append(f"\n{brief.advisor_note}\n")

        parts.append("\nGood luck today. Trade with discipline. 🙏")

        return "\n".join(parts)

    def _format_market_section(self, brief: MorningBrief) -> str:
        """Format market mood section of morning brief."""
        lines: list[str] = []

        mood_display = (brief.market_mood or "unknown").replace("_", " ").title()
        lines.append(f"\n<b>Market Mood:</b> {mood_display}")

        # Nifty + VIX line
        nifty_part = ""
        if brief.nifty_change_pct is not None:
            sign = "+" if brief.nifty_change_pct >= 0 else ""
            nifty_part = f"Nifty {sign}{brief.nifty_change_pct:.1f}%"

        vix_part = ""
        if brief.india_vix is not None:
            vix_signal_display = (brief.vix_signal or "").replace("_", " ")
            vix_part = f"VIX {brief.india_vix} — {vix_signal_display}"

        if nifty_part and vix_part:
            lines.append(f"{nifty_part} | {vix_part}")
        elif nifty_part:
            lines.append(nifty_part)
        elif vix_part:
            lines.append(vix_part)

        # VIX gate
        if brief.vix_gate:
            gate_emoji = "✅" if brief.vix_gate == "open" else "🚫"
            lines.append(f"VIX Gate: {gate_emoji} {brief.vix_gate}")

        return "\n".join(lines)

    def _format_portfolio_section(self, brief: MorningBrief) -> str:
        """Format portfolio section of morning brief."""
        lines: list[str] = ["\n<b>Your Portfolio:</b>"]

        if not brief.open_positions:
            capital_str = self._fmt_rupees(brief.total_capital)
            lines.append(f"No open positions. Full buying power {capital_str}.")
        else:
            for pos in brief.open_positions:
                pnl_str = ""
                if pos.pnl_rupees is not None:
                    sign = "+" if pos.pnl_rupees >= 0 else ""
                    emoji = "📈" if pos.pnl_rupees >= 0 else "📉"
                    pnl_str = f" {emoji} {sign}{self._fmt_rupees(pos.pnl_rupees)}"
                lines.append(
                    f"  {pos.ticker} — {pos.shares} shares @ "
                    f"{self._fmt_rupees(pos.entry_price)}{pnl_str}"
                )

            if brief.available_capital is not None:
                lines.append(
                    f"Available capital: {self._fmt_rupees(brief.available_capital)}"
                )

        return "\n".join(lines)

    def _format_setups_section(self, brief: MorningBrief) -> str:
        """Format trade setups section of morning brief."""
        if not brief.top_setups:
            reason = brief.no_setup_reason or "No setups met quality threshold"
            reason_display = reason.replace("_", " ").capitalize()
            return f"\n<b>Today's Setups:</b>\nNone today. {reason_display}."

        count = len(brief.top_setups)
        lines: list[str] = [f"\n<b>Today's Setups ({count} found):</b>"]

        number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

        for i, setup in enumerate(brief.top_setups):
            emoji = number_emojis[i] if i < len(number_emojis) else f"{i + 1}."
            lines.append(
                f"\n{emoji} <b>{setup.ticker}</b> — "
                f"Entry {self._fmt_rupees(setup.entry_low)}-"
                f"{self._fmt_rupees(setup.entry_high)}"
            )
            lines.append(
                f"   🎯 Target {self._fmt_rupees(setup.target)} | "
                f"🛑 Stop {self._fmt_rupees(setup.stop_loss)}"
            )

            risk_str = self._fmt_rupees(setup.risk_rupees) if setup.risk_rupees else "—"
            reward_str = self._fmt_rupees(setup.reward_rupees) if setup.reward_rupees else "—"
            lines.append(
                f"   📊 Risk {risk_str} | Reward {reward_str} | "
                f"R/R {setup.risk_reward}"
            )
            lines.append(f"   💡 Confidence {setup.confidence:.1f}/10")

            if setup.shares > 0:
                pos_str = self._fmt_rupees(setup.position_rupees) if setup.position_rupees else ""
                shares_info = f"   📦 {setup.shares} shares"
                if pos_str:
                    shares_info += f" ({pos_str})"
                lines.append(shares_info)

        return "\n".join(lines)

    def _format_lesson_section(self, brief: MorningBrief) -> str:
        """Format lesson of the day section."""
        lesson = brief.lesson_of_day
        if not lesson:
            return ""

        concept_display = lesson.concept.replace("_", " ").title()
        return (
            f"\n<b>Today's Lesson:</b> {concept_display}\n"
            f"{lesson.summary}"
        )

    # ═══════════════════════════════════════════════════════
    # EVENING REVIEW
    # ═══════════════════════════════════════════════════════

    def format_evening_review(self, review: EveningReview) -> str:
        """Format an EveningReview into Telegram HTML.

        Args:
            review: EveningReview model with all fields populated.

        Returns:
            HTML-formatted string for Telegram.
        """
        if review.telegram_text:
            return review.telegram_text

        date_str = review.generated_at.strftime("%d %b %Y")
        parts: list[str] = [f"📊 <b>Evening Review — {date_str}</b>\n"]

        # Market summary
        parts.append("<b>Market Summary:</b>")
        if review.nifty_change_pct is not None:
            sign = "+" if review.nifty_change_pct >= 0 else ""
            parts.append(f"Nifty closed {sign}{review.nifty_change_pct:.2f}%")
        if review.india_vix is not None:
            parts.append(f"VIX {review.india_vix}")
        if review.market_mood:
            mood_display = review.market_mood.replace("_", " ").title()
            parts.append(f"Mood: {mood_display}")

        # Top movers
        if review.top_gainers or review.top_losers:
            parts.append("\n<b>What moved today:</b>")
            for g in review.top_gainers:
                parts.append(f"  📈 {g}")
            for l in review.top_losers:
                parts.append(f"  📉 {l}")

        # Positions
        if review.open_positions:
            parts.append("\n<b>Your positions:</b>")
            for pos in review.open_positions:
                pnl_str = ""
                if pos.pnl_rupees is not None:
                    sign = "+" if pos.pnl_rupees >= 0 else ""
                    pnl_str = f" ({sign}{self._fmt_rupees(pos.pnl_rupees)})"
                parts.append(
                    f"  {pos.ticker} — {pos.shares} shares{pnl_str}"
                )
            if review.day_pnl is not None:
                sign = "+" if review.day_pnl >= 0 else ""
                parts.append(f"Day P&L: {sign}{self._fmt_rupees(review.day_pnl)}")
        else:
            parts.append("\n<b>Your positions:</b>\nNo open positions.")

        # Tomorrow outlook
        if review.tomorrow_outlook:
            parts.append(f"\n<b>Tomorrow's outlook:</b>\n{review.tomorrow_outlook}")

        # Lesson recap
        if review.lesson_recap:
            parts.append(f"\n<b>Lesson recap:</b>\n{review.lesson_recap}")

        # Advisor note
        if review.advisor_note:
            parts.append(f"\n{review.advisor_note}")

        parts.append("\nRest well. Markets open again at 9:15 AM. 🌙")

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════
    # WEEKLY SUMMARY
    # ═══════════════════════════════════════════════════════

    def format_weekly_summary(self, summary: WeeklySummary) -> str:
        """Format a WeeklySummary into Telegram HTML."""
        if summary.telegram_text:
            return summary.telegram_text

        parts: list[str] = []

        week_range = ""
        if summary.week_start and summary.week_end:
            week_range = f" ({summary.week_start} to {summary.week_end})"
        parts.append(f"📅 <b>Weekly Review{week_range}</b>\n")

        # Performance
        parts.append("<b>Performance:</b>")
        parts.append(f"  Trades opened: {summary.trades_opened}")
        parts.append(f"  Trades closed: {summary.trades_closed}")
        if summary.trades_closed > 0:
            parts.append(
                f"  Won: {summary.winning_trades} | "
                f"Lost: {summary.losing_trades}"
            )
            if summary.win_rate is not None:
                parts.append(f"  Win rate: {summary.win_rate:.0f}%")
        if summary.week_pnl is not None:
            sign = "+" if summary.week_pnl >= 0 else ""
            emoji = "📈" if summary.week_pnl >= 0 else "📉"
            parts.append(
                f"  {emoji} Week P&L: {sign}{self._fmt_rupees(summary.week_pnl)}"
            )

        # Positions carried forward
        if summary.open_positions:
            parts.append("\n<b>Positions carried forward:</b>")
            for pos in summary.open_positions:
                parts.append(f"  {pos.ticker} — {pos.shares} shares")

        # Lessons
        if summary.lessons_taught:
            parts.append("\n<b>Lessons this week:</b>")
            for lesson in summary.lessons_taught:
                display = lesson.replace("_", " ").title()
                parts.append(f"  📚 {display}")

        # Review + outlook
        if summary.week_review:
            parts.append(f"\n<b>Week review:</b>\n{summary.week_review}")
        if summary.next_week_outlook:
            parts.append(f"\n<b>Next week:</b>\n{summary.next_week_outlook}")
        if summary.advisor_note:
            parts.append(f"\n{summary.advisor_note}")

        parts.append("\nEnjoy your weekend. See you Monday! 🙏")

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════
    # WATCHLIST ALERT
    # ═══════════════════════════════════════════════════════

    def format_watchlist_alert(self, alert: WatchlistAlert) -> str:
        """Format a WatchlistAlert into Telegram HTML.

        No Claude call — pure template formatting.
        Designed for instant delivery during market hours.

        Args:
            alert: WatchlistAlert with setup details.

        Returns:
            HTML-formatted alert message.
        """
        if alert.telegram_text:
            return alert.telegram_text

        parts: list[str] = [
            f"🚨 <b>ENTRY ALERT — {alert.ticker}</b>\n",
            f"Price {self._fmt_rupees(alert.current_price)} has entered your entry zone",
        ]

        if alert.entry_zone_low is not None and alert.entry_zone_high is not None:
            parts.append(
                f"{self._fmt_rupees(alert.entry_zone_low)} - "
                f"{self._fmt_rupees(alert.entry_zone_high)}."
            )

        parts.append("\n<b>Setup reminder:</b>")

        if alert.target is not None:
            parts.append(f"🎯 Target: {self._fmt_rupees(alert.target)}")
        if alert.stop_loss is not None:
            parts.append(f"🛑 Stop loss: {self._fmt_rupees(alert.stop_loss)}")
        if alert.shares > 0:
            position_value = alert.current_price * alert.shares
            parts.append(
                f"📊 Position: {alert.shares} shares "
                f"({self._fmt_rupees(position_value)})"
            )
        if alert.risk_rupees is not None:
            risk_pct = ""
            if alert.risk_rupees > 0:
                # Estimate as % of ₹50,000 default capital
                pct = (alert.risk_rupees / Decimal("50000")) * 100
                risk_pct = f" ({pct:.2f}% of capital)"
            parts.append(f"⚡ Risk: {self._fmt_rupees(alert.risk_rupees)}{risk_pct}")

        parts.append(
            "\nThis is the setup from this morning's brief.\n"
            "Enter only if you're ready to set the stop "
            "loss immediately after buying."
        )
        parts.append("\nAct now or wait for next opportunity. ⏰")

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════
    # ERROR ALERT
    # ═══════════════════════════════════════════════════════

    def format_error_alert(self, error: ErrorAlert) -> str:
        """Format an ErrorAlert into Telegram HTML.

        Always sent — never silently fail.
        """
        if error.telegram_text:
            return error.telegram_text

        emoji = "🚨" if error.is_critical else "⚠️"
        parts: list[str] = [
            f"{emoji} <b>SwingAdvisorBot Alert</b>\n",
            f"<b>Source:</b> {error.error_source}",
            f"<b>Error:</b> {error.error_message}",
            f"<b>Time:</b> {error.attempted_at.strftime('%H:%M IST')}",
        ]

        if error.retry_scheduled:
            parts.append(
                f"<b>Retry:</b> {error.retry_scheduled.strftime('%H:%M IST')}"
            )

        if error.is_critical:
            parts.append(
                "\n🔴 This is a critical error. "
                "Manual intervention may be required."
            )

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _fmt_rupees(amount: Decimal | float | int | None) -> str:
        """Format a number as Indian rupee string.

        Examples:
            50000    → '₹50,000'
            769.55   → '₹769.55'
            1247.00  → '₹1,247'
            None     → '₹—'
        """
        if amount is None:
            return "₹—"

        value = Decimal(str(amount))

        # If it's a whole number, skip decimals
        if value == value.to_integral_value():
            return f"₹{int(value):,}"

        return f"₹{value:,.2f}"


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

message_formatter = MessageFormatter()
