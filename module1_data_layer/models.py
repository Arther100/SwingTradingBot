"""
SwingAdvisorBot — Module 1: Data Layer
models.py — All Pydantic v2 data models

Every model in this file represents data that flows through the advisor pipeline.
Each model is typed, validated, and carries advisor-quality signals.
No field exists without purpose. No model returns without context.

Data flow:
  Kite Connect / NewsAPI / FRED  →  these models  →  Module 2 (Claude AI)

Token budget awareness:
  The MarketData master object is serialized and sent to Claude API.
  Claude claude-opus-4-5 costs money per token — every token earns its place.
  Hard limit: 2500 tokens per MarketData payload.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ── Re-exports from FII/DII and Earnings fetchers ────────────
# These models are defined in the fetcher files (their natural home)
# and imported here so all downstream modules can use a single
# import path: `from module1_data_layer.models import FiiDiiData`
from module1_data_layer.fetchers.fii_dii_fetcher import (
    FiiDiiData,
    FiiDiiSignal,
)
from module1_data_layer.fetchers.earnings_fetcher import (
    EarningsEvent,
    EarningsRisk,
    EarningsRiskLevel,
    classify_earnings_risk,
)

__all__ = [
    "FiiDiiData",
    "FiiDiiSignal",
    "EarningsEvent",
    "EarningsRisk",
    "EarningsRiskLevel",
    "classify_earnings_risk",
]


# ─────────────────────────────────────────────────────────────
# Enums — Advisor-quality labels, never generic
# Every label must be meaningful to a 20+ year finance advisor.
# "high" / "low" / "flag1" are banned. Use domain language.
# ─────────────────────────────────────────────────────────────


class MarketStatus(str, enum.Enum):
    """NSE market operating state. Pre-market is 9:00-9:15 IST."""

    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"


class VIXSignal(str, enum.Enum):
    """India VIX fear classification.

    Thresholds calibrated for India VIX historical ranges:
      < 14  → low_fear (complacency / strong bull)
      14–20 → moderate_fear (normal conditions)
      20–30 → high_fear (anxiety / hedging activity)
      ≥ 30  → extreme_fear (panic / crash conditions)
    """

    LOW_FEAR = "low_fear"
    MODERATE_FEAR = "moderate_fear"
    HIGH_FEAR = "high_fear"
    EXTREME_FEAR = "extreme_fear"


class VolumeSignal(str, enum.Enum):
    """Volume anomaly classification relative to 30-day average.

    Thresholds:
      ≥ 3.0x → unusual_spike (block deal / news driven)
      ≥ 1.3x → above_average (institutional interest)
      0.7–1.3x → normal (no signal)
      < 0.7x → below_average (low participation)
    """

    UNUSUAL_SPIKE = "unusual_spike"
    ABOVE_AVERAGE = "above_average"
    NORMAL = "normal"
    BELOW_AVERAGE = "below_average"


class RangePosition(str, enum.Enum):
    """Position of current price within the 52-week high/low range.

    Percentile bands:
      ≥ 80% → near_high (potential resistance / breakout zone)
      60–80% → upper (strength, but watch for distribution)
      40–60% → middle (no clear directional bias)
      20–40% → lower (weakness, watch for accumulation)
      < 20%  → near_low (capitulation or deep value)
    """

    NEAR_HIGH = "near_high"
    UPPER = "upper"
    MIDDLE = "middle"
    LOWER = "lower"
    NEAR_LOW = "near_low"


class AdvisorFlag(str, enum.Enum):
    """Primary advisor signal for a stock.

    Each flag carries a specific meaning and suggested action:
      breakout_watch       → Price near resistance with volume. Watch for breakout.
      accumulation_zone    → Institutional buying detected. Watchlist candidate.
      unusual_activity     → Volume spike without clear price direction. Investigate.
      selling_pressure     → Distribution pattern. Caution advised.
      consolidation        → Sideways movement. Wait for direction.
      momentum_building    → Progressive higher lows with rising volume. Trend forming.
      distribution_zone    → Smart money exiting. Reduce exposure.
      neutral              → No actionable signal at this time.
    """

    BREAKOUT_WATCH = "breakout_watch"
    ACCUMULATION_ZONE = "accumulation_zone"
    UNUSUAL_ACTIVITY = "unusual_activity"
    SELLING_PRESSURE = "selling_pressure"
    CONSOLIDATION = "consolidation"
    MOMENTUM_BUILDING = "momentum_building"
    DISTRIBUTION_ZONE = "distribution_zone"
    NEUTRAL = "neutral"


class NewsSentiment(str, enum.Enum):
    """Assessed sentiment of a news headline."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class MarketImpact(str, enum.Enum):
    """Expected magnitude of market impact from a news item or economic event."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DataFreshness(str, enum.Enum):
    """How fresh the data in a MarketData snapshot is.

    real_time    → Fetched within last 3 minutes during market hours.
    delayed      → Fetched within last 15 minutes during market hours.
    end_of_day   → Market is closed. Showing final session prices.
    stale        → Data older than 15 minutes during market hours. Refresh needed.
    """

    REAL_TIME = "real_time"
    DELAYED = "delayed"
    END_OF_DAY = "end_of_day"
    STALE = "stale"


class PipelineStatus(str, enum.Enum):
    """Overall pipeline health after self-reflection check.

    healthy  → All 7 health check steps passed. Advisor can trust this data.
    degraded → Some steps failed but core data is available. Advisor warned.
    failed   → Critical failures. Data should not be used for advice.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class ChangeDirection(str, enum.Enum):
    """Direction of change for economic indicators."""

    UP = "up"
    DOWN = "down"
    UNCHANGED = "unchanged"


class SectorSignal(str, enum.Enum):
    """Overall sector direction signal for sector rotation analysis."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ─────────────────────────────────────────────────────────────
# Custom Exceptions — Informative, never cryptic
# Every exception tells you what happened, why, and what to do.
# ─────────────────────────────────────────────────────────────


class DataFetchError(Exception):
    """Raised when a data source fails and cannot be recovered.

    Always includes the source name, reason, and a suggested next step.
    A silent failure is a bug — this exception ensures noise.

    Example:
        raise DataFetchError(
            source="KiteConnect",
            reason="Access token expired mid-session",
            suggestion="Re-authenticate via KiteAuthManager.get_valid_token()"
        )
    """

    def __init__(
        self,
        source: str,
        reason: str,
        suggestion: str = "Check logs and retry.",
    ):
        self.source = source
        self.reason = reason
        self.suggestion = suggestion
        super().__init__(
            f"[DataFetchError] Source: {source} — {reason}. {suggestion}"
        )


class TokenBudgetError(Exception):
    """Raised when MarketData cannot be trimmed within the token budget.

    This means the data payload is too large for Claude API consumption
    even after all 5 trimming steps have been applied.
    Manual review of the payload is required.
    """

    def __init__(self, estimated_tokens: int, budget: int):
        self.estimated_tokens = estimated_tokens
        self.budget = budget
        super().__init__(
            f"[TokenBudgetError] Estimated {estimated_tokens} tokens exceeds "
            f"budget of {budget}. All trimming steps exhausted. "
            f"Review data payload manually."
        )


class PipelineHealthError(Exception):
    """Raised when the pipeline self-reflection health check fails.

    Includes the specific step number that failed and the reason.
    A senior advisor would not accept data from a broken pipeline.

    Health check steps:
      Step 1: Count stocks fetched (need >= 10)
      Step 2: Verify all advisor_flags set
      Step 3: Verify all timestamps IST
      Step 4: Verify is_real_data is True
      Step 5: Estimate token count (< 2500)
      Step 6: Verify market_status matches IST time
      Step 7: Generate pipeline_health_report
    """

    def __init__(self, step: int, reason: str):
        self.step = step
        self.reason = reason
        super().__init__(
            f"[PipelineHealthError] Health check failed at Step {step}: {reason}"
        )


class KiteAuthError(Exception):
    """Raised when Kite Connect authentication fails.

    Covers: token expired, invalid credentials, re-auth flow failure,
    or any state where we cannot make authenticated API calls.
    """

    def __init__(
        self,
        reason: str,
        suggestion: str = "Re-authenticate via Kite login flow.",
    ):
        self.reason = reason
        self.suggestion = suggestion
        super().__init__(f"[KiteAuthError] {reason}. {suggestion}")


# ─────────────────────────────────────────────────────────────
# Core Data Models — Advisor-quality, signal-rich
# Every model carries enough context for a 20+ year advisor
# to make an informed recommendation without asking for more.
# ─────────────────────────────────────────────────────────────


class StockData(BaseModel):
    """Complete stock data with advisor-quality signals.

    Every StockData instance must carry enough context for a senior
    finance advisor to make an informed recommendation. Bare price
    data without signals is rejected by the pipeline health check.

    Example of GOOD output:
      {"ticker": "HDFCBANK", "price": 1623.45, "volume_ratio": 1.37,
       "volume_signal": "above_average", "advisor_flag": "accumulation_zone",
       "cot_reasoning": "Price 9.5% below 52w high, volume 37% above average..."}

    Example of REJECTED output:
      {"ticker": "HDFCBANK", "price": 1623.45, "volume": 8500000}
      → No signals. No context. No advisor value. Rejected.
    """

    # ── Identity ──
    ticker: str = Field(
        ..., description="NSE ticker symbol, e.g. HDFCBANK, RELIANCE, TCS"
    )
    exchange: str = Field(
        default="NSE",
        description="Exchange code — always NSE for this system",
    )
    company_name: str = Field(
        default="",
        description="Full registered company name, e.g. HDFC Bank Limited",
    )
    sector: str = Field(
        default="",
        description="Business sector classification, e.g. Banking, IT, Pharma",
    )
    instrument_token: int = Field(
        default=0,
        description="Kite Connect instrument token for API calls",
    )

    # ── Price Data ──
    price: float = Field(
        ..., description="Last traded price (LTP) in INR"
    )
    open: float = Field(
        default=0.0, description="Day's opening price in INR"
    )
    high: float = Field(
        default=0.0, description="Day's highest traded price in INR"
    )
    low: float = Field(
        default=0.0, description="Day's lowest traded price in INR"
    )
    close: float = Field(
        default=0.0,
        description="Previous day's closing price or current session close in INR",
    )
    change: float = Field(
        default=0.0,
        description="Absolute price change from previous close in INR",
    )
    change_pct: float = Field(
        default=0.0,
        description="Percentage change from previous close",
    )

    # ── Volume Analysis ──
    volume: int = Field(
        default=0, description="Today's total traded volume (shares)"
    )
    avg_volume_30d: int = Field(
        default=0, description="30-day average daily traded volume"
    )
    volume_ratio: float = Field(
        default=0.0,
        description="Today's volume / 30-day average — ratio above 1.0 means higher than usual",
    )
    volume_signal: VolumeSignal = Field(
        default=VolumeSignal.NORMAL,
        description="Advisor-quality volume classification based on volume_ratio thresholds",
    )

    # ── 52-Week Range Analysis ──
    high_52w: float = Field(
        default=0.0,
        serialization_alias="52w_high",
        description="52-week high price in INR",
    )
    low_52w: float = Field(
        default=0.0,
        serialization_alias="52w_low",
        description="52-week low price in INR",
    )
    position_in_52w_range: RangePosition = Field(
        default=RangePosition.MIDDLE,
        description="Where current price sits in the 52-week high/low range",
    )

    # ── Advisor Signals (populated by signals/advisor_signals.py) ──
    advisor_flag: Optional[AdvisorFlag] = Field(
        default=None,
        description="Primary advisor signal flag — set by CoT signal calculator",
    )
    cot_reasoning: Optional[str] = Field(
        default=None,
        description="Chain of Thought reasoning explaining why this advisor_flag was assigned",
    )

    # ── Metadata ──
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="Timestamp of last data update — always IST (Asia/Kolkata)",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def compute_derived_fields(self) -> StockData:
        """Cross-field validation and derived field computation.

        Ensures volume_ratio, volume_signal, and position_in_52w_range
        are internally consistent with their source fields. A senior
        advisor catches inconsistencies instantly — so must our validation.

        Derivation rules:
          volume_ratio  = volume / avg_volume_30d  (if both > 0 and ratio not set)
          volume_signal = classified from volume_ratio thresholds
          position_in_52w_range = classified from price position in 52w band
        """
        if self.volume > 0 and self.avg_volume_30d > 0 and self.volume_ratio == 0.0:
            self.volume_ratio = round(self.volume / self.avg_volume_30d, 2)

        if self.volume_ratio > 0:
            if self.volume_ratio >= 3.0:
                self.volume_signal = VolumeSignal.UNUSUAL_SPIKE
            elif self.volume_ratio >= 1.3:
                self.volume_signal = VolumeSignal.ABOVE_AVERAGE
            elif self.volume_ratio >= 0.7:
                self.volume_signal = VolumeSignal.NORMAL
            else:
                self.volume_signal = VolumeSignal.BELOW_AVERAGE

        if self.high_52w > 0 and self.low_52w > 0 and self.high_52w > self.low_52w:
            range_total = self.high_52w - self.low_52w
            position_pct = (self.price - self.low_52w) / range_total
            if position_pct >= 0.80:
                self.position_in_52w_range = RangePosition.NEAR_HIGH
            elif position_pct >= 0.60:
                self.position_in_52w_range = RangePosition.UPPER
            elif position_pct >= 0.40:
                self.position_in_52w_range = RangePosition.MIDDLE
            elif position_pct >= 0.20:
                self.position_in_52w_range = RangePosition.LOWER
            else:
                self.position_in_52w_range = RangePosition.NEAR_LOW

        return self


class NewsItem(BaseModel):
    """A single news item scored for market relevance and advisor consumption.

    Every news item carries advisor-quality analysis: sentiment,
    sector impact, relevance score, and a plain English advisor note.
    Raw headlines without scoring are useless to the advisor brain.

    Only items with relevance_score >= 0.70 reach the advisor.

    Example of GOOD output:
      {"headline": "RBI holds repo rate at 6.5%", "sentiment": "neutral",
       "market_impact": "high", "affected_sectors": ["Banking", "Finance"],
       "relevance_score": 0.94,
       "advisor_note": "Rate hold positive for banking sector. Watch HDFCBANK."}
    """

    headline: str = Field(
        ..., description="News headline text — concise, factual"
    )
    source: str = Field(
        default="",
        description="News source publication, e.g. Economic Times, Moneycontrol, LiveMint",
    )
    url: str = Field(
        default="", description="URL to the full article for reference"
    )

    sentiment: NewsSentiment = Field(
        default=NewsSentiment.NEUTRAL,
        description="Assessed sentiment: positive, negative, neutral, or mixed",
    )
    market_impact: MarketImpact = Field(
        default=MarketImpact.LOW,
        description="Expected magnitude of market impact from this news",
    )
    affected_sectors: list[str] = Field(
        default_factory=list,
        description="Sectors likely affected, e.g. ['Banking', 'Finance', 'RealEstate']",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score 0.0–1.0. Only items >= 0.70 reach the advisor.",
    )

    # ── CoT Analysis (populated by signals/news_scorer.py) ──
    cot_reasoning: Optional[str] = Field(
        default=None,
        description="Chain of Thought reasoning explaining the relevance_score assignment",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description="Plain English note for the advisor summarizing market impact",
    )

    published_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="Publication timestamp — converted to IST from source timezone",
    )


class SectorPerformance(BaseModel):
    """Sector-level performance summary for the advisor.

    Gives the advisor a macro view of which sectors are moving
    and in which direction — critical for sector rotation analysis
    and identifying where institutional money is flowing.
    """

    sector_name: str = Field(
        ...,
        description="Sector name matching NSE sector classification, e.g. Banking, IT, Pharma",
    )
    change_pct: float = Field(
        default=0.0,
        description="Sector index percentage change for the current session",
    )
    top_gainer: str = Field(
        default="",
        description="NSE ticker of the best-performing stock in this sector today",
    )
    top_gainer_change_pct: float = Field(
        default=0.0,
        description="Percentage change of the sector's top gainer",
    )
    top_loser: str = Field(
        default="",
        description="NSE ticker of the worst-performing stock in this sector today",
    )
    top_loser_change_pct: float = Field(
        default=0.0,
        description="Percentage change of the sector's top loser",
    )
    sector_signal: SectorSignal = Field(
        default=SectorSignal.NEUTRAL,
        description="Overall sector direction signal for rotation analysis",
    )
    advisor_note: str = Field(
        default="",
        description="Brief advisor note on sector movement and what to watch",
    )


class EconomicEvent(BaseModel):
    """Macro-economic data point from FRED or similar source.

    Tracks US/global economic indicators that influence Indian markets.
    Fed rate decisions, US GDP, crude oil prices, US CPI — all flow here.
    The advisor uses these to understand the global context around
    domestic price action.
    """

    event_name: str = Field(
        ...,
        description="Indicator name, e.g. Federal Funds Rate, US CPI, Crude Oil WTI",
    )
    series_id: str = Field(
        default="",
        description="FRED series identifier, e.g. FEDFUNDS, CPIAUCSL, DCOILWTICO",
    )
    value: float = Field(
        default=0.0, description="Latest published value of the indicator"
    )
    previous_value: float = Field(
        default=0.0, description="Previous period value for comparison"
    )
    change_direction: ChangeDirection = Field(
        default=ChangeDirection.UNCHANGED,
        description="Direction of change from previous value",
    )
    impact_level: MarketImpact = Field(
        default=MarketImpact.LOW,
        description="Expected impact level on Indian markets",
    )
    source: str = Field(
        default="FRED",
        description="Data source identifier, e.g. FRED, RBI, WorldBank",
    )
    affected_markets: list[str] = Field(
        default_factory=lambda: ["NSE"],
        description="Markets affected by this indicator",
    )
    advisor_note: str = Field(
        default="",
        description="Plain English note on how this indicator affects Indian markets right now",
    )
    published_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="Data release timestamp converted to IST",
    )

    @model_validator(mode="after")
    def compute_change_direction(self) -> EconomicEvent:
        """Derive change_direction from value vs previous_value.

        If previous_value is available (non-zero), automatically
        classify the direction of change so the advisor sees the trend.
        """
        if self.previous_value != 0.0:
            if self.value > self.previous_value:
                self.change_direction = ChangeDirection.UP
            elif self.value < self.previous_value:
                self.change_direction = ChangeDirection.DOWN
            else:
                self.change_direction = ChangeDirection.UNCHANGED
        return self


class PipelineHealthReport(BaseModel):
    """Self-reflection report from the data pipeline.

    Before any MarketData is handed to the advisor, the pipeline
    runs a 7-step health check. This report documents the results.
    A failed health check means the advisor cannot trust the data.

    Health check steps:
      Step 1: Count successfully fetched stocks (need >= 10)
      Step 2: Verify all advisor_flags are set (not None)
      Step 3: Verify all timestamps are IST
      Step 4: Verify is_real_data is True
      Step 5: Estimate token count (must be < 2500)
      Step 6: Verify market_status matches current IST time
      Step 7: Generate this pipeline_health_report
    """

    status: PipelineStatus = Field(
        default=PipelineStatus.FAILED,
        description="Overall pipeline health: healthy, degraded, or failed",
    )
    stocks_fetched: int = Field(
        default=0,
        description="Number of stocks successfully fetched with price data",
    )
    news_fetched: int = Field(
        default=0,
        description="Number of news items fetched and scored for relevance",
    )
    vix_available: bool = Field(
        default=False,
        description="Whether India VIX data was successfully fetched",
    )
    all_signals_set: bool = Field(
        default=False,
        description="Whether every stock has a non-None advisor_flag",
    )
    all_timestamps_ist: bool = Field(
        default=False,
        description="Whether all timestamps across the payload are in IST",
    )
    all_real_data: bool = Field(
        default=False,
        description="Whether is_real_data flag confirmed True",
    )
    token_estimate: int = Field(
        default=0,
        description="Estimated token count of the serialized MarketData payload",
    )
    token_within_budget: bool = Field(
        default=False,
        description="Whether estimated tokens fit within the 2500 token budget",
    )
    market_status_correct: bool = Field(
        default=False,
        description="Whether market_status accurately reflects current IST time and NSE schedule",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of issues found during health check — empty list means fully healthy",
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this health check was performed (IST)",
    )


class MarketData(BaseModel):
    """Master data object — the complete market briefing for the advisor.

    This is THE single object that Module 2 (AI Analysis Engine) receives.
    It contains everything a senior finance advisor needs to give
    intelligent swing trading advice for NSE stocks.

    Every field earned its place through signal value. No dead weight.

    Token budget: 2500 tokens max when serialized to JSON.

    Priority order (what gets included first, what gets trimmed last):
      Priority 1: India VIX + market_status        (never trimmed)
      Priority 2: Nifty50 + Sensex values           (never trimmed)
      Priority 3: Top 5 stocks by signal strength    (trimmed last)
      Priority 4: Top 3 news by relevance_score      (trimmed after stocks)
      Priority 5: Sector performance summary         (trimmed before news)
      Priority 6: Economic events (high impact only) (trimmed first)
      Priority 7: Remaining stocks                   (trimmed first)

    Edge cases handled:
      Market closed → Shows end-of-day prices with appropriate freshness flag.
      API failure   → Raises DataFetchError, never returns stale/fake data.
      Over budget   → Runs 5-step trim sequence, raises TokenBudgetError if exhausted.
    """

    # ── Market State (Priority 1 — never trimmed) ──
    market_status: MarketStatus = Field(
        ...,
        description="Current NSE market state: open (9:15–15:30 IST), closed, or pre_market (9:00–9:15)",
    )
    market_status_reason: str = Field(
        default="",
        description="Human-readable explanation of market status, e.g. 'NSE closes at 15:30 IST'",
    )
    data_freshness: DataFreshness = Field(
        default=DataFreshness.REAL_TIME,
        description="How fresh this snapshot is — real_time during market hours, end_of_day after close",
    )

    # ── Index Data (Priority 2 — never trimmed) ──
    nifty50_value: float = Field(
        default=0.0, description="Nifty 50 index current value"
    )
    nifty50_change_pct: float = Field(
        default=0.0, description="Nifty 50 percentage change for the session"
    )
    sensex_value: float = Field(
        default=0.0, description="BSE Sensex current value"
    )
    sensex_change_pct: float = Field(
        default=0.0, description="BSE Sensex percentage change for the session"
    )

    # ── VIX (Priority 1 — never trimmed) ──
    india_vix: float = Field(
        default=0.0,
        description="India VIX value — the market's fear gauge. Critical for risk assessment.",
    )
    vix_signal: VIXSignal = Field(
        default=VIXSignal.MODERATE_FEAR,
        description="Advisor-quality VIX classification derived from india_vix value",
    )

    # ── Data Collections ──
    stocks: list[StockData] = Field(
        default_factory=list,
        description="Stock data with advisor signals. Top 5 are Priority 3, rest are Priority 7.",
    )
    news: list[NewsItem] = Field(
        default_factory=list,
        description="Scored news items filtered by relevance >= 0.70. Top 3 are Priority 4.",
    )
    sectors: list[SectorPerformance] = Field(
        default_factory=list,
        description="Sector performance summaries for rotation analysis. Priority 5.",
    )
    economic_events: list[EconomicEvent] = Field(
        default_factory=list,
        description="Macro-economic events from FRED. Only high-impact items. Priority 6.",
    )

    # ── Institutional Flow + Earnings Calendar (Priority 5 — trimmed with sectors) ──
    fii_dii: Optional[FiiDiiData] = Field(
        default=None,
        description=(
            "FII/DII institutional flow for today. "
            "None if NSE was unreachable at fetch time. "
            "Combined signal drives M4 confidence adjustment (+0.5 strong_bullish, "
            "-1.5 strong_bearish)."
        ),
    )
    earnings_events: list[EarningsEvent] = Field(
        default_factory=list,
        description=(
            "Upcoming earnings results in the next 10 days for watchlist stocks. "
            "M4 screener uses this to block (HIGH) or warn (MEDIUM/LOW) setups. "
            "M6 morning brief shows this as the 'Earnings this week' section."
        ),
    )

    # ── Advisor Context ──
    advisor_morning_signal: str = Field(
        default="",
        description=(
            "2-3 sentence plain English market summary that the AI advisor (Module 2) "
            "uses as opening context when speaking to the user. Must be informative "
            "enough to start a conversation about today's market."
        ),
    )

    # ── Data Integrity ──
    is_real_data: bool = Field(
        default=True,
        description=(
            "Must ALWAYS be True. SwingAdvisorBot never operates on mock/fake data. "
            "If real data cannot be fetched, raise DataFetchError instead of setting this to False."
        ),
    )

    # ── Pipeline Metadata ──
    pipeline_status: PipelineStatus = Field(
        default=PipelineStatus.HEALTHY,
        description="Pipeline health status after the 7-step self-reflection check",
    )
    pipeline_health_report: Optional[PipelineHealthReport] = Field(
        default=None,
        description="Detailed health check results — None until health check runs",
    )

    # ── Timestamp ──
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this MarketData snapshot was assembled (IST)",
    )

    @model_validator(mode="after")
    def validate_data_integrity(self) -> MarketData:
        """Cross-field validation — the advisor demands consistency.

        Validates:
          1. is_real_data must be True (non-negotiable)
          2. VIX signal must match the VIX value thresholds
        """
        if not self.is_real_data:
            raise ValueError(
                "is_real_data must be True. SwingAdvisorBot never operates on fake data. "
                "If real data cannot be fetched, raise DataFetchError instead."
            )

        if self.india_vix > 0:
            if self.india_vix >= 30:
                self.vix_signal = VIXSignal.EXTREME_FEAR
            elif self.india_vix >= 20:
                self.vix_signal = VIXSignal.HIGH_FEAR
            elif self.india_vix >= 14:
                self.vix_signal = VIXSignal.MODERATE_FEAR
            else:
                self.vix_signal = VIXSignal.LOW_FEAR

        return self

    def estimate_tokens(self) -> int:
        """Estimate the token count of this MarketData when serialized to JSON.

        Formula: (character_count / 4) * 1.2
          - 1 token ≈ 4 characters (standard LLM tokenizer heuristic)
          - 1.2x multiplier accounts for JSON structure overhead
            (brackets, keys, colons, quotes, commas)

        This estimate drives the trim_to_budget() decision.
        Accuracy is within ±15% of actual tiktoken count — sufficient
        for budget enforcement without adding a tokenizer dependency.
        """
        json_str = self.model_dump_json(by_alias=True)
        char_count = len(json_str)
        estimated_tokens = int((char_count / 4) * 1.2)
        return estimated_tokens

    def trim_to_budget(self, max_tokens: int = 2500) -> None:
        """Trim MarketData to fit within the token budget for Claude API.

        Follows a strict priority-based trimming sequence. Each step
        checks if the budget is met before proceeding to the next.
        VIX, market_status, Nifty50, and Sensex are NEVER trimmed —
        the advisor always needs the market's vital signs.

        Trimming sequence:
          Step 1: Remove cot_reasoning from stocks ranked 6th and below.
                  These are lower-priority stocks — their signals stay,
                  but the verbose reasoning is dropped to save tokens.
          Step 2: Keep only top 10 stocks. Stocks are assumed pre-sorted
                  by signal strength (advisor_signals.py handles ranking).
          Step 3: Keep only top 3 news items sorted by relevance_score.
          Step 4: Remove all economic events.
          Step 5: Re-estimate. If still over budget → raise TokenBudgetError.
                  At this point, manual review is needed.

        Raises:
            TokenBudgetError: When all trimming steps are exhausted and
                the payload still exceeds max_tokens.
        """
        if self.estimate_tokens() <= max_tokens:
            return

        # Step 1: Strip CoT reasoning from lower-priority stocks (positions 6+)
        if len(self.stocks) > 5:
            for stock in self.stocks[5:]:
                stock.cot_reasoning = None

        if self.estimate_tokens() <= max_tokens:
            return

        # Step 2: Trim to top 10 stocks (already sorted by signal strength)
        if len(self.stocks) > 10:
            self.stocks = self.stocks[:10]

        if self.estimate_tokens() <= max_tokens:
            return

        # Step 3: Keep only top 3 news by relevance_score
        if len(self.news) > 3:
            self.news = sorted(
                self.news, key=lambda n: n.relevance_score, reverse=True
            )[:3]

        if self.estimate_tokens() <= max_tokens:
            return

        # Step 4: Remove economic events entirely
        self.economic_events = []

        if self.estimate_tokens() <= max_tokens:
            return

        # Step 5: Remove earnings_events (M4/M6 re-fetch if needed)
        self.earnings_events = []

        if self.estimate_tokens() <= max_tokens:
            return

        # Step 6: Drop FII/DII advisor_note (keep signal fields only)
        if self.fii_dii is not None:
            self.fii_dii = self.fii_dii.model_copy(
                update={"advisor_note": "", "market_impact": ""}
            )

        if self.estimate_tokens() <= max_tokens:
            return

        # Step 7: All trimming exhausted — budget still exceeded
        raise TokenBudgetError(
            estimated_tokens=self.estimate_tokens(), budget=max_tokens
        )
