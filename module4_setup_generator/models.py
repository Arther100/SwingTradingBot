"""
SwingAdvisorBot — Module 4: Trade Setup Generator
models.py — Pydantic models for trade setups

This is where intelligence meets action. Every model here
represents a complete, actionable trade recommendation
that a retail trader can follow.

Models:
  TradeSetup     → Complete setup: entry, target, stop, reasoning
  SkippedSetup   → Stock that was rejected by M3 or filters
  SetupPackage   → Bundle of 3-5 setups + market context
  SetupFilter    → Input parameters for setup generation

Enums:
  SetupType      → swing_long (only long trades for now)
  SetupFreshness → live, pre_market_preview, next_day_watchlist

Data flow:
  MarketData (M1) + MarketAnalysis (M2) + UserContext (M2)
    → TradeSetupAgent (10-step CoT)
    → M3 risk check per stock
    → Claude reasoning for approved stocks
    → SetupPackage (to M6 Reports / M8 Frontend)

All prices are Decimal. IST timestamps.
exclude_none=True on all serialization.
"""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


# ═══════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════


class SetupType(str, enum.Enum):
    """Type of trade setup.

    Currently only long trades. Short trades may be
    added in a future module for F&O segment.
    """

    SWING_LONG = "swing_long"


class SetupFreshness(str, enum.Enum):
    """When this setup was generated relative to market hours.

    live                → Generated during market hours (9:15-15:30 IST)
    pre_market_preview  → Before market open, using previous close data
    next_day_watchlist  → After market close, for next trading day
    """

    LIVE = "live"
    PRE_MARKET_PREVIEW = "pre_market_preview"
    NEXT_DAY_WATCHLIST = "next_day_watchlist"


# ═══════════════════════════════════════════════════════════
# TRADE SETUP — The main output
# ═══════════════════════════════════════════════════════════


class TradeSetup(BaseModel):
    """Complete, actionable swing trade setup.

    Every field is populated with real data. No guessing.
    A beginner should know exactly what to do from this.

    Example (APPROVED):
        {
          "ticker": "HDFCBANK",
          "company_name": "HDFC Bank Limited",
          "sector": "Banking",
          "setup_type": "swing_long",
          "entry_zone_low": "768.00",
          "entry_zone_high": "775.00",
          "target_price": "850.00",
          "stop_loss": "738.00",
          "current_price": "769.55",
          "hold_days_min": 5,
          "hold_days_max": 8,
          "confidence_score": 7.2,
          "risk_reward_ratio": "1:3.1",
          "position_size_shares": 13,
          "position_size_rupees": "10003.15",
          "max_risk_rupees": "403.00",
          "risk_pct_of_capital": "0.81",
          "risk_verdict": "APPROVED"
        }
    """

    # ── Stock Identity ──
    ticker: str = Field(
        ...,
        description="NSE ticker symbol, e.g. HDFCBANK",
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Full company name for display",
    )
    sector: str = Field(
        default="Other",
        description="Business sector, e.g. Banking, IT, Energy",
    )
    setup_type: SetupType = Field(
        default=SetupType.SWING_LONG,
        description="Type of trade setup",
    )
    freshness: SetupFreshness = Field(
        default=SetupFreshness.LIVE,
        description="When this setup was generated relative to market hours",
    )

    # ── Price Levels (all Decimal) ──
    entry_zone_low: Decimal = Field(
        ...,
        description="Lower end of entry zone (INR)",
    )
    entry_zone_high: Decimal = Field(
        ...,
        description="Upper end of entry zone (INR)",
    )
    target_price: Decimal = Field(
        ...,
        description="Target price for profit booking (INR)",
    )
    stop_loss: Decimal = Field(
        ...,
        description="Stop loss price — exit immediately below this (INR)",
    )
    current_price: Decimal = Field(
        ...,
        description="Current market price at time of setup (INR)",
    )

    # ── Holding Period ──
    hold_days_min: int = Field(
        default=3,
        ge=1,
        description="Minimum expected holding days",
    )
    hold_days_max: int = Field(
        default=10,
        ge=1,
        description="Maximum expected holding days",
    )

    # ── Scoring ──
    confidence_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Setup quality score 1-10. Average=6.5-7.5, excellent=8+",
    )

    # ── Risk Metrics (from M3) ──
    risk_reward_ratio: str = Field(
        ...,
        description="Risk/reward as string, e.g. '1:3.69'",
    )
    position_size_shares: int = Field(
        default=0,
        ge=0,
        description="Number of shares to buy (from M3)",
    )
    position_size_rupees: Optional[Decimal] = Field(
        default=None,
        description="Total position value in INR",
    )
    max_risk_rupees: Optional[Decimal] = Field(
        default=None,
        description="Maximum risk in INR (shares × risk_per_share)",
    )
    risk_pct_of_capital: Optional[Decimal] = Field(
        default=None,
        description="Risk as percentage of total capital",
    )
    risk_verdict: str = Field(
        default="APPROVED",
        description="M3 verdict: APPROVED, REDUCE_SIZE",
    )

    # ── Claude-Generated Reasoning ──
    setup_reasoning: Optional[str] = Field(
        default=None,
        description="Why this stock, why now (2-3 sentences, data-grounded)",
    )
    entry_trigger: Optional[str] = Field(
        default=None,
        description="Specific condition to enter the trade",
    )
    exit_strategy: Optional[str] = Field(
        default=None,
        description="When and how to take profits",
    )
    risk_warning: Optional[str] = Field(
        default=None,
        description="What invalidates this setup — specific price level",
    )
    macro_context: Optional[str] = Field(
        default=None,
        description="How current macro environment affects this stock",
    )
    lesson: Optional[str] = Field(
        default=None,
        description="One trading concept this setup demonstrates",
    )

    # ── Internal Reasoning ──
    cot_reasoning: Optional[str] = Field(
        default=None,
        description="Full 10-step Chain of Thought reasoning trail",
    )
    advisor_flag: Optional[str] = Field(
        default=None,
        description="M1 advisor flag that qualified this stock",
    )
    volume_signal: Optional[str] = Field(
        default=None,
        description="M1 volume signal for this stock",
    )

    # ── Metadata ──
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this setup was generated (IST)",
    )

    def to_display_card(self) -> str:
        """Format setup as a readable display card.

        Used by M6 Reports and M8 Frontend.
        """
        lines = [
            f"{'─' * 45}",
            f"  {self.ticker} — {self.setup_type.value.upper()}",
            f"  Sector: {self.sector} | Confidence: {self.confidence_score}/10",
            f"{'─' * 45}",
            f"  Entry zone:  ₹{self.entry_zone_low} - ₹{self.entry_zone_high}",
            f"  Target:      ₹{self.target_price}",
            f"  Stop loss:   ₹{self.stop_loss}",
            f"  R/R:         {self.risk_reward_ratio}",
            f"  Position:    {self.position_size_shares} shares"
            + (f" (₹{self.position_size_rupees})" if self.position_size_rupees else ""),
            f"  Risk:        ₹{self.max_risk_rupees}"
            + (f" ({self.risk_pct_of_capital}%)" if self.risk_pct_of_capital else ""),
            f"  Hold:        {self.hold_days_min}-{self.hold_days_max} days",
        ]

        if self.setup_reasoning:
            lines.append(f"\n  Why: {self.setup_reasoning}")
        if self.entry_trigger:
            lines.append(f"  When: {self.entry_trigger}")
        if self.exit_strategy:
            lines.append(f"  Exit: {self.exit_strategy}")
        if self.risk_warning:
            lines.append(f"  ⚠️  {self.risk_warning}")
        if self.lesson:
            lines.append(f"\n  📚 {self.lesson}")

        lines.append(f"{'─' * 45}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# SKIPPED SETUP — Stock that didn't qualify
# ═══════════════════════════════════════════════════════════


class SkippedSetup(BaseModel):
    """A stock that was evaluated but skipped.

    Provides transparency — the user sees what was
    considered and why it didn't make the cut.

    Example:
        {
          "ticker": "TCS",
          "skip_reason": "M3 rejected — VIX gate failed",
          "risk_verdict": "REJECTED",
          "advisor_note": "TCS setup skipped."
        }
    """

    ticker: str = Field(
        ...,
        description="NSE ticker that was skipped",
    )
    skip_reason: str = Field(
        ...,
        description="Why this stock was skipped",
    )
    risk_verdict: Optional[str] = Field(
        default=None,
        description="M3 verdict if risk check was reached",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description="Brief explanation for the user",
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score if calculated before skip",
    )


# ═══════════════════════════════════════════════════════════
# SETUP PACKAGE — Bundle of setups for delivery
# ═══════════════════════════════════════════════════════════


class SetupPackage(BaseModel):
    """Complete daily setup package — the main M4 output.

    Contains 0-5 trade setups, skipped stocks, and
    overall market context. This is what M6 Reports
    and M8 Frontend consume.

    When no setups qualify:
        {
          "setups": [],
          "reason": "no_qualifying_setups",
          "advisor_note": "No setups today Vijay...",
          "resume_condition": "Re-evaluate when VIX < 20",
          "market_mood": "bearish"
        }
    """

    # ── Setups ──
    setups: list[TradeSetup] = Field(
        default_factory=list,
        description="List of qualifying trade setups (0-5)",
    )
    skipped_setups: list[SkippedSetup] = Field(
        default_factory=list,
        description="Stocks evaluated but not qualifying",
    )

    # ── Market Context ──
    market_mood: Optional[str] = Field(
        default=None,
        description="Overall market mood from M2",
    )
    india_vix: Optional[float] = Field(
        default=None,
        description="India VIX at time of generation",
    )

    # ── Package Metadata ──
    setup_count: int = Field(
        default=0,
        ge=0,
        description="Number of qualifying setups",
    )
    candidates_evaluated: int = Field(
        default=0,
        ge=0,
        description="Total stocks evaluated",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Why package is empty (if no setups)",
    )
    resume_condition: Optional[str] = Field(
        default=None,
        description="When to re-evaluate (if no setups)",
    )

    # ── Advisor Output ──
    advisor_note: Optional[str] = Field(
        default=None,
        description="Morning advisor note summarising setups",
    )
    cot_reasoning: Optional[str] = Field(
        default=None,
        description="Full CoT reasoning for setup selection",
    )

    # ── Token Usage ──
    total_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Total Claude input tokens used",
    )
    total_output_tokens: int = Field(
        default=0,
        ge=0,
        description="Total Claude output tokens used",
    )

    # ── Metadata ──
    freshness: SetupFreshness = Field(
        default=SetupFreshness.LIVE,
        description="Freshness of this package",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this package was generated (IST)",
    )
    display_name: str = Field(
        default="Trader",
        description="User's name for personalised notes",
    )


# ═══════════════════════════════════════════════════════════
# SETUP FILTER — Input parameters
# ═══════════════════════════════════════════════════════════


class SetupFilter(BaseModel):
    """Parameters for controlling setup generation.

    Passed to generate_setups() to customise output.
    Defaults are sensible for most users.
    """

    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID",
    )
    display_name: str = Field(
        default="Vijay",
        description="User's name for personalised output",
    )
    max_setups: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of setups to generate",
    )
    min_confidence: float = Field(
        default=6.0,
        ge=1.0,
        le=10.0,
        description="Minimum confidence score to include a setup",
    )
    capital: float = Field(
        default=50000.0,
        ge=10000,
        description="Trading capital in INR",
    )
    risk_tolerance: str = Field(
        default="moderate",
        description="Risk tolerance: conservative, moderate, aggressive",
    )
    tickers: Optional[list[str]] = Field(
        default=None,
        description="Specific tickers to evaluate (None = use screener)",
    )
    skip_claude: bool = Field(
        default=False,
        description="Skip Claude reasoning (for testing without API credits)",
    )
