"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
models.py — Data models for reports, alerts, and scheduling

Models:
  MorningBrief      — Full morning market brief (8:50 AM)
  EveningReview     — End-of-day market review (4:30 PM)
  WeeklySummary     — Weekend performance summary (Sat 10 AM)
  WatchlistAlert    — Entry zone price alert (real-time)
  ErrorAlert        — System error notification
  ReportMetadata    — Common fields across all report types
  SetupSummary      — Compact setup for Telegram display
  LessonOfDay       — Daily trading concept
  PositionSummary   — Open position snapshot
  AlertRecord       — Dedup tracking for sent alerts

All timestamps in IST. All amounts in Decimal.
"""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from module1_data_layer.models import EarningsEvent, EarningsRisk, FiiDiiData

IST = ZoneInfo("Asia/Kolkata")


# ═══════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════


class ReportType(str, enum.Enum):
    """Types of reports M6 generates."""
    MORNING_BRIEF = "morning_brief"
    EVENING_REVIEW = "evening_review"
    WEEKLY_SUMMARY = "weekly_summary"
    WATCHLIST_ALERT = "watchlist_alert"
    ERROR_ALERT = "error_alert"


class AlertType(str, enum.Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    URGENT = "urgent"
    ENTRY_ZONE = "entry_zone"
    STOP_HIT = "stop_hit"
    TARGET_HIT = "target_hit"


class DeliveryStatus(str, enum.Enum):
    """Telegram delivery status."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SPLIT = "split"


# ═══════════════════════════════════════════════════════════
# COMPACT SUB-MODELS (for Telegram display)
# ═══════════════════════════════════════════════════════════


class SetupSummary(BaseModel):
    """Compact trade setup for Telegram message.

    Extracted from M4 TradeSetup — only fields needed
    for the morning brief Telegram card.
    """
    ticker: str = Field(
        ...,
        description="NSE ticker symbol",
    )
    sector: str = Field(
        default="Other",
        description="Business sector",
    )
    entry_low: Decimal = Field(
        ...,
        description="Entry zone lower bound",
    )
    entry_high: Decimal = Field(
        ...,
        description="Entry zone upper bound",
    )
    target: Decimal = Field(
        ...,
        description="Target price",
    )
    stop_loss: Decimal = Field(
        ...,
        description="Stop loss price",
    )
    confidence: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Confidence score 1-10",
    )
    risk_rupees: Optional[Decimal] = Field(
        default=None,
        description="Risk per trade in rupees",
    )
    reward_rupees: Optional[Decimal] = Field(
        default=None,
        description="Reward per trade in rupees",
    )
    risk_reward: str = Field(
        default="",
        description="R/R ratio string e.g. '1:3.1'",
    )
    shares: int = Field(
        default=0,
        ge=0,
        description="Recommended shares to buy",
    )
    position_rupees: Optional[Decimal] = Field(
        default=None,
        description="Total position value in rupees",
    )
    risk_pct: Optional[Decimal] = Field(
        default=None,
        description="Risk as % of capital",
    )
    setup_reasoning: Optional[str] = Field(
        default=None,
        description="Why this setup — 1-2 sentences",
    )
    entry_trigger: Optional[str] = Field(
        default=None,
        description="When to enter",
    )
    exit_strategy: Optional[str] = Field(
        default=None,
        description="How to exit",
    )
    earnings_risk: Optional[EarningsRisk] = Field(
        default=None,
        description="Upcoming earnings risk for this stock",
    )


class LessonOfDay(BaseModel):
    """Daily trading concept for education.

    Selected from M5 learning history — picks
    a concept not recently taught.
    """
    concept: str = Field(
        ...,
        description="Concept identifier e.g. 'trailing_stop_loss'",
    )
    summary: str = Field(
        ...,
        description="2-3 sentence explanation for Telegram",
    )
    difficulty: str = Field(
        default="beginner",
        description="beginner or intermediate",
    )


class PositionSummary(BaseModel):
    """Open position snapshot for reports.

    Pulled from M5 trade records.
    """
    ticker: str = Field(
        ...,
        description="NSE ticker symbol",
    )
    entry_price: Decimal = Field(
        ...,
        description="Entry price",
    )
    current_price: Optional[Decimal] = Field(
        default=None,
        description="Current market price",
    )
    shares: int = Field(
        ...,
        description="Number of shares held",
    )
    pnl_rupees: Optional[Decimal] = Field(
        default=None,
        description="Unrealised P&L in rupees",
    )
    pnl_pct: Optional[Decimal] = Field(
        default=None,
        description="Unrealised P&L percentage",
    )
    stop_loss: Optional[Decimal] = Field(
        default=None,
        description="Current stop loss",
    )
    target: Optional[Decimal] = Field(
        default=None,
        description="Current target",
    )
    days_held: int = Field(
        default=0,
        ge=0,
        description="Number of days position has been held",
    )


# ═══════════════════════════════════════════════════════════
# REPORT METADATA (shared across all report types)
# ═══════════════════════════════════════════════════════════


class ReportMetadata(BaseModel):
    """Common metadata for all M6 reports."""
    report_type: ReportType = Field(
        ...,
        description="Type of report",
    )
    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this report was generated (IST)",
    )
    delivery_status: DeliveryStatus = Field(
        default=DeliveryStatus.PENDING,
        description="Telegram delivery status",
    )
    telegram_message_ids: list[int] = Field(
        default_factory=list,
        description="Telegram message IDs (multiple if split)",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Total Claude tokens used for this report",
    )
    generation_time_ms: int = Field(
        default=0,
        ge=0,
        description="Time to generate report in milliseconds",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if generation failed",
    )


# ═══════════════════════════════════════════════════════════
# MORNING BRIEF — Primary daily output (8:50 AM)
# ═══════════════════════════════════════════════════════════


class MorningBrief(ReportMetadata):
    """Complete morning market brief sent at 8:50 AM.

    This is the flagship output of SwingAdvisorBot.
    Combines M1 data + M2 analysis + M3 risk +
    M4 setups + M5 memory into one Telegram message.

    Example:
        {
          "report_type": "morning_brief",
          "market_mood": "cautious_bullish",
          "india_vix": "18.33",
          "top_setups": [...],
          "lesson_of_day": {...},
          "telegram_sent": true
        }
    """
    report_type: ReportType = Field(
        default=ReportType.MORNING_BRIEF,
    )

    # Market data (from M1)
    market_status: str = Field(
        default="pre_market",
        description="Market status at generation time",
    )
    india_vix: Optional[Decimal] = Field(
        default=None,
        description="India VIX value",
    )
    vix_signal: Optional[str] = Field(
        default=None,
        description="VIX classification: low_fear, moderate_fear, etc.",
    )
    nifty_value: Optional[float] = Field(
        default=None,
        description="Nifty 50 index value",
    )
    nifty_change_pct: Optional[float] = Field(
        default=None,
        description="Nifty session change %",
    )
    sensex_value: Optional[float] = Field(
        default=None,
        description="Sensex value",
    )
    sensex_change_pct: Optional[float] = Field(
        default=None,
        description="Sensex session change %",
    )

    # Analysis (from M2)
    market_mood: Optional[str] = Field(
        default=None,
        description="Market mood: bullish, cautious_bullish, neutral, etc.",
    )
    mood_confidence: Optional[float] = Field(
        default=None,
        description="Mood confidence 0.0-1.0",
    )
    situation_summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence market situation from M2",
    )

    # Risk (from M3)
    vix_gate: Optional[str] = Field(
        default=None,
        description="VIX gate status: open or closed",
    )
    vix_limit: Optional[Decimal] = Field(
        default=None,
        description="VIX gate threshold",
    )

    # Setups (from M4)
    top_setups: list[SetupSummary] = Field(
        default_factory=list,
        description="Top trade setups for today (max 5)",
    )
    no_setup_reason: Optional[str] = Field(
        default=None,
        description="Why no setups today (if empty)",
    )

    # Portfolio (from M5)
    open_positions: list[PositionSummary] = Field(
        default_factory=list,
        description="Current open positions",
    )
    total_capital: Optional[Decimal] = Field(
        default=None,
        description="Total trading capital",
    )
    available_capital: Optional[Decimal] = Field(
        default=None,
        description="Available buying power",
    )

    # Education
    lesson_of_day: Optional[LessonOfDay] = Field(
        default=None,
        description="Today's trading lesson",
    )

    # Events & advisor note
    key_events: list[str] = Field(
        default_factory=list,
        description="Important events today (RBI, earnings, etc.)",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description="Personal advisor note to Vijay",
    )

    # FII/DII institutional flow (from M1)
    fii_dii: Optional[FiiDiiData] = Field(
        default=None,
        description="Today's FII/DII institutional flow data",
    )

    # Upcoming earnings calendar (from M1)
    earnings_calendar: list[EarningsEvent] = Field(
        default_factory=list,
        description="Earnings events in next 10 days for tracked tickers",
    )

    # Telegram message text (pre-formatted HTML)
    telegram_text: Optional[str] = Field(
        default=None,
        description="Pre-formatted Telegram HTML message",
    )


# ═══════════════════════════════════════════════════════════
# EVENING REVIEW — End-of-day summary (4:30 PM)
# ═══════════════════════════════════════════════════════════


class EveningReview(ReportMetadata):
    """End-of-day market review sent at 4:30 PM.

    Summarises what happened today, position updates,
    and tomorrow's outlook.
    """
    report_type: ReportType = Field(
        default=ReportType.EVENING_REVIEW,
    )

    # Market close data
    nifty_close: Optional[float] = Field(
        default=None,
        description="Nifty 50 closing value",
    )
    nifty_change_pct: Optional[float] = Field(
        default=None,
        description="Nifty session change %",
    )
    sensex_close: Optional[float] = Field(
        default=None,
        description="Sensex closing value",
    )
    sensex_change_pct: Optional[float] = Field(
        default=None,
        description="Sensex session change %",
    )
    india_vix: Optional[Decimal] = Field(
        default=None,
        description="Closing VIX",
    )
    vix_signal: Optional[str] = Field(
        default=None,
        description="VIX classification at close",
    )
    market_mood: Optional[str] = Field(
        default=None,
        description="End-of-day mood assessment",
    )

    # Top movers
    top_gainers: list[str] = Field(
        default_factory=list,
        description="Top gaining stocks from watchlist",
    )
    top_losers: list[str] = Field(
        default_factory=list,
        description="Top losing stocks from watchlist",
    )

    # Portfolio update
    open_positions: list[PositionSummary] = Field(
        default_factory=list,
        description="End-of-day position snapshots",
    )
    day_pnl: Optional[Decimal] = Field(
        default=None,
        description="Total P&L for the day",
    )
    total_pnl: Optional[Decimal] = Field(
        default=None,
        description="Total P&L across all open positions",
    )

    # Outlook
    tomorrow_outlook: Optional[str] = Field(
        default=None,
        description="Brief outlook for tomorrow",
    )
    lesson_recap: Optional[str] = Field(
        default=None,
        description="Recap of today's lesson",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description="Evening advisor note to Vijay",
    )

    # Telegram
    telegram_text: Optional[str] = Field(
        default=None,
        description="Pre-formatted Telegram HTML message",
    )


# ═══════════════════════════════════════════════════════════
# WEEKLY SUMMARY — Weekend review (Saturday 10 AM)
# ═══════════════════════════════════════════════════════════


class WeeklySummary(ReportMetadata):
    """Weekend performance summary sent Saturday 10 AM.

    Reviews the full week — trades, P&L, lessons learned,
    and the week ahead.
    """
    report_type: ReportType = Field(
        default=ReportType.WEEKLY_SUMMARY,
    )

    # Week range
    week_start: Optional[str] = Field(
        default=None,
        description="Week start date (Monday) as YYYY-MM-DD",
    )
    week_end: Optional[str] = Field(
        default=None,
        description="Week end date (Friday) as YYYY-MM-DD",
    )

    # Performance
    trades_opened: int = Field(
        default=0,
        ge=0,
        description="Trades opened this week",
    )
    trades_closed: int = Field(
        default=0,
        ge=0,
        description="Trades closed this week",
    )
    winning_trades: int = Field(
        default=0,
        ge=0,
        description="Profitable trades closed",
    )
    losing_trades: int = Field(
        default=0,
        ge=0,
        description="Loss-making trades closed",
    )
    week_pnl: Optional[Decimal] = Field(
        default=None,
        description="Net P&L for the week",
    )
    win_rate: Optional[float] = Field(
        default=None,
        description="Win rate this week (%)",
    )

    # Open positions carried forward
    open_positions: list[PositionSummary] = Field(
        default_factory=list,
        description="Positions carried to next week",
    )

    # Lessons covered
    lessons_taught: list[str] = Field(
        default_factory=list,
        description="Concepts taught this week",
    )

    # Advisor notes
    week_review: Optional[str] = Field(
        default=None,
        description="Advisor's review of the week",
    )
    next_week_outlook: Optional[str] = Field(
        default=None,
        description="Outlook for next week",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description="Personal note to Vijay",
    )

    # Telegram
    telegram_text: Optional[str] = Field(
        default=None,
        description="Pre-formatted Telegram HTML message",
    )


# ═══════════════════════════════════════════════════════════
# WATCHLIST ALERT — Real-time price alert
# ═══════════════════════════════════════════════════════════


class WatchlistAlert(ReportMetadata):
    """Real-time alert when watchlist stock enters an entry zone.

    Sent instantly during market hours (9:15 AM - 3:30 PM).
    No Claude call — pure price comparison + template message.
    """
    report_type: ReportType = Field(
        default=ReportType.WATCHLIST_ALERT,
    )

    alert_type: AlertType = Field(
        default=AlertType.ENTRY_ZONE,
        description="Type of alert triggered",
    )
    ticker: str = Field(
        ...,
        description="NSE ticker that triggered alert",
    )
    current_price: Decimal = Field(
        ...,
        description="Price at time of alert",
    )
    triggered_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When price entered the zone",
    )

    # Entry zone from morning setup
    entry_zone_low: Optional[Decimal] = Field(
        default=None,
        description="Entry zone lower bound",
    )
    entry_zone_high: Optional[Decimal] = Field(
        default=None,
        description="Entry zone upper bound",
    )

    # Setup reminder
    target: Optional[Decimal] = Field(
        default=None,
        description="Target price from morning setup",
    )
    stop_loss: Optional[Decimal] = Field(
        default=None,
        description="Stop loss from morning setup",
    )
    shares: int = Field(
        default=0,
        ge=0,
        description="Recommended shares",
    )
    risk_rupees: Optional[Decimal] = Field(
        default=None,
        description="Risk amount from morning setup",
    )
    risk_reward: Optional[str] = Field(
        default=None,
        description="R/R ratio string",
    )

    # Telegram
    telegram_text: Optional[str] = Field(
        default=None,
        description="Pre-formatted Telegram HTML alert",
    )


# ═══════════════════════════════════════════════════════════
# ERROR ALERT — System error notification
# ═══════════════════════════════════════════════════════════


class ErrorAlert(ReportMetadata):
    """System error alert sent to Vijay via Telegram.

    Never silently fail — always notify the user.
    """
    report_type: ReportType = Field(
        default=ReportType.ERROR_ALERT,
    )

    error_source: str = Field(
        ...,
        description="Which module/step failed",
    )
    error_message: str = Field(
        ...,
        description="Human-readable error description",
    )
    attempted_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When the failed operation was attempted",
    )
    retry_scheduled: Optional[datetime] = Field(
        default=None,
        description="When retry is scheduled (if any)",
    )
    is_critical: bool = Field(
        default=False,
        description="Whether this blocks all further operations",
    )

    # Telegram
    telegram_text: Optional[str] = Field(
        default=None,
        description="Pre-formatted error alert message",
    )


# ═══════════════════════════════════════════════════════════
# ALERT RECORD — Dedup tracking (SQLite)
# ═══════════════════════════════════════════════════════════


class AlertRecord(BaseModel):
    """Tracks sent alerts to prevent duplicates.

    Stored in SQLite. Checked before every alert send.
    Key: (ticker, alert_type, date) — unique per day.
    """
    alert_id: str = Field(
        ...,
        description="Unique alert identifier",
    )
    ticker: str = Field(
        ...,
        description="NSE ticker symbol",
    )
    alert_type: AlertType = Field(
        ...,
        description="Type of alert sent",
    )
    sent_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When alert was sent",
    )
    date: str = Field(
        ...,
        description="Date string YYYY-MM-DD for dedup",
    )
    telegram_message_id: Optional[int] = Field(
        default=None,
        description="Telegram message ID",
    )
