"""
SwingAdvisorBot — Module 2: AI Analysis Engine
models.py — All Pydantic v2 data models for AI analysis

This module transforms raw market data (Module 1) into intelligent,
personalised advisor analysis. Every model here carries meaning —
not just data, but context, reasoning, and actionable advice.

Data flow:
  Module 1 MarketData → Claude API → these models → Module 3/4/6/8

Model hierarchy:
  UserContext          → User memory context (from Module 5, stubbed here)
  SentimentReport      → News sentiment analysis from SentimentAnalysisAgent
  SectorAnalysis       → Per-sector mood + signals for rotation analysis
  MarketAnalysis       → Master output — the advisor's full analysis
  AnalysisResult       → Wrapper with metadata (tokens, timing, cache status)
  AnalysisQualityReport → Self-reflection quality check results

Every model enforces the advisor personality standard:
  → No bare data without context
  → No advice without reasoning
  → No action without risk warning
  → No generic output — always personalised
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


# ─────────────────────────────────────────────────────────────
# Enums — Advisor-quality labels for market mood and analysis
# ─────────────────────────────────────────────────────────────


class MarketMood(str, enum.Enum):
    """Overall market mood classification.

    The advisor's high-level read on the market. This single
    label tells Module 3 (Risk) and Module 4 (Trade Setups)
    how aggressive or conservative to be.

    Mood definitions:
      bullish          → Strong upward bias. Breakout setups appropriate.
      cautious_bullish → Positive but with caveats. Selective positioning.
      neutral          → No clear direction. Wait for clarity.
      cautious_bearish → Negative bias building. Reduce exposure.
      bearish          → Strong downward pressure. Capital protection mode.
      extreme_fear     → VIX ≥ 30 or crash conditions. Cash is king.
    """

    BULLISH = "bullish"
    CAUTIOUS_BULLISH = "cautious_bullish"
    NEUTRAL = "neutral"
    CAUTIOUS_BEARISH = "cautious_bearish"
    BEARISH = "bearish"
    EXTREME_FEAR = "extreme_fear"


class AnalysisDepth(str, enum.Enum):
    """Depth of analysis requested.

    full  → Complete MarketAnalysis with all fields populated.
            Used for morning brief and detailed user queries.
            Costs ~2400 tokens per Claude call.
    quick → Mood + top signals only. No lesson, shorter reasoning.
            Used for real-time checks and M4 pre-validation.
            Costs ~1200 tokens per Claude call.
    """

    FULL = "full"
    QUICK = "quick"


class SentimentDirection(str, enum.Enum):
    """Overall news sentiment direction.

    Derived from weighted average of all scored news items.
    The advisor uses this to confirm or challenge the
    price-based market mood assessment.

    positive → Majority of high-relevance news is bullish.
    negative → Majority of high-relevance news is bearish.
    mixed    → Conflicting signals in the news flow.
    neutral  → No strong sentiment either way.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class QualityVerdict(str, enum.Enum):
    """Self-reflection quality check result.

    passed  → All quality thresholds met. Analysis is advisor-grade.
    warning → Some thresholds marginal. Analysis usable but flagged.
    failed  → Critical quality gaps. Analysis must be regenerated.
    """

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────
# Custom Exceptions — Module 2 specific
# Every exception tells you what failed, why, and what to do.
# ─────────────────────────────────────────────────────────────


class AnalysisQualityError(Exception):
    """Raised when Claude's response fails the quality gate.

    The self-reflection check found missing or shallow fields.
    This error triggers a retry with a stronger quality prompt.

    Example:
        raise AnalysisQualityError(
            missing_fields=["risk", "lesson"],
            shallow_fields=["reasoning"],
            retry_count=1
        )
    """

    def __init__(
        self,
        missing_fields: list[str] | None = None,
        shallow_fields: list[str] | None = None,
        retry_count: int = 0,
    ):
        self.missing_fields = missing_fields or []
        self.shallow_fields = shallow_fields or []
        self.retry_count = retry_count
        detail = []
        if self.missing_fields:
            detail.append(f"Missing: {', '.join(self.missing_fields)}")
        if self.shallow_fields:
            detail.append(f"Shallow: {', '.join(self.shallow_fields)}")
        super().__init__(
            f"[AnalysisQualityError] Quality gate failed (retry {retry_count}). "
            f"{'. '.join(detail)}. "
            f"A senior advisor would not accept this output."
        )


class AnalysisParseError(Exception):
    """Raised when Claude's response is not valid JSON.

    The response could not be parsed into a structured
    MarketAnalysis. This triggers a retry with stricter
    JSON formatting instructions.

    Example:
        raise AnalysisParseError(
            raw_response="Here is my analysis: ...",
            parse_error="Expecting '{' at position 0"
        )
    """

    def __init__(self, raw_response: str, parse_error: str):
        self.raw_response = raw_response[:500]  # Cap for logging safety
        self.parse_error = parse_error
        super().__init__(
            f"[AnalysisParseError] Claude response is not valid JSON. "
            f"Parse error: {parse_error}. "
            f"Response starts with: '{raw_response[:100]}...'"
        )


class InsufficientDataError(Exception):
    """Raised when MarketData is too incomplete for quality analysis.

    The pipeline health check passed but the data is still
    insufficient for the advisor to give meaningful analysis.
    Never generate advice from incomplete data — this is the
    most dangerous failure mode.

    Example:
        raise InsufficientDataError(
            stocks_available=6,
            stocks_required=10,
            reason="Only 6/15 stocks fetched — pipeline degraded"
        )
    """

    def __init__(
        self,
        stocks_available: int,
        stocks_required: int = 10,
        reason: str = "",
    ):
        self.stocks_available = stocks_available
        self.stocks_required = stocks_required
        self.reason = reason
        super().__init__(
            f"[InsufficientDataError] Only {stocks_available}/{stocks_required} stocks available. "
            f"{reason}. "
            f"Cannot generate quality analysis from incomplete data. "
            f"Returning cached analysis with staleness warning."
        )


class FinalAnalysisError(Exception):
    """Raised after all retries are exhausted.

    Claude API was called, retried with quality and JSON
    prompts, but still could not produce acceptable output.
    This is a hard failure — the system must surface this
    to the user rather than returning bad advice.

    Example:
        raise FinalAnalysisError(
            attempts=3,
            last_error="Quality gate failed on retry 2"
        )
    """

    def __init__(self, attempts: int, last_error: str):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"[FinalAnalysisError] Analysis failed after {attempts} attempts. "
            f"Last error: {last_error}. "
            f"Claude API could not produce advisor-quality output. "
            f"Manual review required."
        )


# ─────────────────────────────────────────────────────────────
# Core Data Models — Advisor-quality, Claude-driven
# ─────────────────────────────────────────────────────────────


class UserContext(BaseModel):
    """User memory context passed from Module 5 (Memory & Personalization).

    This model carries everything the advisor needs to personalise
    its analysis. Without user context, the advisor gives generic
    advice — which is lazy advice. Every response must reference
    the user's specific situation.

    In Module 2, this is stubbed with sensible defaults.
    Module 5 will populate it from SQLite + ChromaDB.

    The advisor must know:
      → What positions the user holds (to warn about risk)
      → How much capital is available (to size recommendations)
      → What the user's risk tolerance is (to calibrate advice)
      → What the user has learned recently (to teach progressively)
    """

    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID — primary user identifier across all modules.",
    )
    display_name: str = Field(
        default="Vijay",
        description="User's display name for personalised greetings and advice.",
    )

    # ── Capital & Risk Profile ──
    total_capital: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Total trading capital in INR available for swing trades. "
            "Used by Module 3 (Risk Engine) for position sizing."
        ),
    )
    risk_tolerance: str = Field(
        default="moderate",
        description=(
            "User's self-declared risk tolerance: conservative, moderate, aggressive. "
            "Conservative → max 2% risk per trade. Moderate → max 3%. Aggressive → max 5%."
        ),
    )
    max_risk_per_trade_pct: float = Field(
        default=3.0,
        ge=0.5,
        le=10.0,
        description="Maximum percentage of capital to risk on a single trade.",
    )
    max_open_positions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent open swing trade positions.",
    )

    # ── Current Positions ──
    open_positions: list[dict] = Field(
        default_factory=list,
        description=(
            "List of currently open swing trade positions. "
            "Each dict: {ticker, entry_price, quantity, entry_date, stop_loss, target}. "
            "The advisor must know these to warn about concentration risk "
            "and to give position-specific advice."
        ),
    )
    closed_positions_count: int = Field(
        default=0,
        ge=0,
        description="Total number of closed trades. Indicates experience level.",
    )
    win_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Win rate percentage across all closed trades.",
    )

    # ── Learning Progress ──
    lessons_completed: list[str] = Field(
        default_factory=list,
        description=(
            "List of lesson topic IDs the user has completed. "
            "Used by Module 7 (Education) to avoid repeating concepts "
            "and to teach progressively."
        ),
    )
    current_learning_topic: str = Field(
        default="",
        description=(
            "The concept the user is currently learning about. "
            "The advisor weaves this into today's market context."
        ),
    )

    # ── Preferences ──
    preferred_sectors: list[str] = Field(
        default_factory=list,
        description=(
            "Sectors the user is interested in or has experience with. "
            "Advisor prioritises these in analysis and trade setups."
        ),
    )
    watchlist: list[str] = Field(
        default_factory=list,
        description=(
            "User's personal watchlist tickers (beyond the default 15). "
            "These get priority in analysis."
        ),
    )

    # ── Session Info ──
    last_interaction: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last user interaction. Used for greeting context.",
    )
    interaction_count: int = Field(
        default=0,
        ge=0,
        description="Total interactions this user has had with the bot.",
    )


class SentimentReport(BaseModel):
    """News sentiment analysis from SentimentAnalysisAgent.

    Transforms raw scored news items (Module 1 news_scorer output)
    into an actionable sentiment view that the advisor uses to
    confirm or challenge the price-based market mood.

    Example:
      Prices say bullish (Nifty +0.4%) but news says "FII selling
      ₹3200 crore" → sentiment is negative → advisor mood becomes
      cautious_bullish instead of bullish. The news provides the
      "why should I be cautious" context.

    This model is produced by SentimentAnalysisAgent (File 09)
    and consumed by MarketAnalysisAgent (File 08) for synthesis.
    """

    # ── Overall Sentiment ──
    overall_sentiment: SentimentDirection = Field(
        default=SentimentDirection.NEUTRAL,
        description=(
            "Weighted aggregate sentiment across all scored news items. "
            "Weighted by relevance_score × recency. "
            "Not a simple average — high-relevance recent news dominates."
        ),
    )
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Numeric sentiment score: -1.0 (extremely bearish) to +1.0 (extremely bullish). "
            "0.0 is neutral. Provides granularity beyond the enum label."
        ),
    )
    sentiment_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the sentiment assessment. "
            "Low if few news items scored or scores are tightly mixed. "
            "High if many items agree directionally."
        ),
    )

    # ── Sector Sentiment ──
    sector_sentiments: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-sector sentiment scores. Key: sector name, Value: -1.0 to +1.0. "
            "Example: {'Banking': 0.6, 'IT': -0.2, 'Energy': 0.1}. "
            "Used by the advisor for sector rotation context."
        ),
    )

    # ── Risk Events ──
    top_risk_events: list[str] = Field(
        default_factory=list,
        description=(
            "Top 3 risk events identified from news today. "
            "Example: ['RBI Governor speech at 11 AM', "
            "'FII selling ₹3200 crore', 'US jobs data tonight']. "
            "The advisor warns the user about these."
        ),
    )
    risk_level: str = Field(
        default="normal",
        description=(
            "Overall risk level from news: low, normal, elevated, high. "
            "Elevated or high → advisor recommends reducing position sizes."
        ),
    )

    # ── Advisor Context ──
    news_summary: str = Field(
        default="",
        description=(
            "2-3 sentence plain English summary of today's news landscape. "
            "The advisor uses this as context when explaining market moves."
        ),
    )
    cot_reasoning: str = Field(
        default="",
        description=(
            "Chain of Thought reasoning for the sentiment assessment. "
            "Documents how the agent weighted and combined news signals."
        ),
    )

    # ── Metadata ──
    news_items_analysed: int = Field(
        default=0,
        ge=0,
        description="Number of news items that were scored and analysed.",
    )
    analysed_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this sentiment analysis was performed (IST).",
    )


class SectorAnalysis(BaseModel):
    """Per-sector analysis with mood, signals, and advisor context.

    Goes deeper than Module 1's SectorPerformance — this model
    adds the advisor's interpretation of what the sector movement
    means and what to do about it.

    Module 1 tells you: "Banking is up 1.2%"
    Module 2 tells you: "Banking is up 1.2% because RBI held rates,
      which stabilises net interest margins. Institutional buying
      in HDFCBANK confirms sector strength. Consider banking stocks
      for swing setups if VIX stays below 16."
    """

    sector_name: str = Field(
        ...,
        description="Sector name matching NSE sector classification.",
    )
    sector_mood: MarketMood = Field(
        default=MarketMood.NEUTRAL,
        description=(
            "Advisor's mood assessment for this specific sector. "
            "Can differ from overall market mood — sector rotation "
            "often means some sectors are bullish while market is neutral."
        ),
    )
    change_pct: float = Field(
        default=0.0,
        description="Sector index percentage change for the session.",
    )
    news_sentiment: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "News sentiment score specific to this sector. "
            "From SentimentReport.sector_sentiments."
        ),
    )

    # ── Advisor Interpretation ──
    situation: str = Field(
        default="",
        description=(
            "What is happening in this sector today. "
            "2-3 sentences of factual context."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "Why this sector is moving this way. "
            "Connects price action to news, macro events, or rotation."
        ),
    )
    advisor_action: str = Field(
        default="",
        description=(
            "What the advisor recommends for this sector. "
            "Specific to the user's positions and risk profile."
        ),
    )

    # ── Key Stocks ──
    top_opportunity: str = Field(
        default="",
        description=(
            "Best stock opportunity in this sector right now. "
            "Ticker + brief reason. Example: 'HDFCBANK — accumulation zone with volume.'"
        ),
    )
    top_risk: str = Field(
        default="",
        description=(
            "Highest risk stock in this sector right now. "
            "Ticker + brief warning. Example: 'AXISBANK — near stop loss, watch closely.'"
        ),
    )


class MarketAnalysis(BaseModel):
    """Master output model — the advisor's complete market analysis.

    This is THE single object that represents what the senior finance
    advisor thinks about the market right now. It is produced by
    MarketAnalysisAgent after synthesising:
      → MarketData from Module 1 (prices, volume, VIX)
      → SentimentReport from SentimentAnalysisAgent (news context)
      → UserContext from Module 5 (personalisation)

    Every field follows the advisor's response structure:
      1. SITUATION — what is happening
      2. REASONING — why it is happening
      3. IMPACT — what it means for the user
      4. ACTION — what to consider doing
      5. RISK — what could go wrong
      6. LESSON — one concept to learn

    This model is consumed by:
      → Module 3 (Risk Engine) — reads market_mood, risk_events
      → Module 4 (Trade Setups) — reads situation, sector_analyses
      → Module 6 (Reports) — reads everything for morning brief
      → Module 8 (Frontend) — displays to user

    Quality gate: Every field is validated for minimum content.
    If Claude returns shallow output, the quality checker rejects it.
    """

    # ── Market Mood Assessment ──
    market_mood: MarketMood = Field(
        ...,
        description=(
            "Overall market mood — the advisor's single-word read on the market. "
            "This one label tells all downstream modules how aggressive to be."
        ),
    )
    mood_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the mood assessment (0.0–1.0). "
            "Below 0.5 → mood is uncertain, advise caution. "
            "Above 0.8 → strong conviction, setups can be more aggressive."
        ),
    )

    # ── Advisor Response Structure (6 required fields) ──
    situation: str = Field(
        ...,
        min_length=1,
        description=(
            "What is happening in the market right now. "
            "2-4 sentences covering Nifty direction, VIX level, "
            "sector leadership, and any significant events. "
            "Must be >= 100 characters for quality gate."
        ),
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description=(
            "Why the market is behaving this way. "
            "Connects price action to macro events, news, "
            "institutional flows, and technical levels. "
            "Must be >= 100 characters for quality gate."
        ),
    )
    user_impact: str = Field(
        default="",
        description=(
            "What this means for the user specifically. "
            "References the user's open positions, capital, "
            "and risk tolerance. Personalised — never generic. "
            "Empty only if no user context is available."
        ),
    )
    action: str = Field(
        ...,
        min_length=1,
        description=(
            "What to consider doing. Must include at least one "
            "specific price level or concrete step. "
            "'Do nothing' is a valid action — but must explain why. "
            "Must be >= 50 characters for quality gate."
        ),
    )
    risk: str = Field(
        ...,
        min_length=1,
        description=(
            "What could go wrong and how to protect against it. "
            "Must include specific risk scenarios. "
            "'The market could go down' is not acceptable. "
            "Must be >= 50 characters for quality gate."
        ),
    )
    lesson: str = Field(
        default="",
        description=(
            "One trading concept to learn from today's market. "
            "Tied to actual market events so the lesson is relevant. "
            "Must be >= 80 characters for quality gate (if provided). "
            "Empty only for quick analysis depth."
        ),
    )

    # ── Chain of Thought ──
    cot_reasoning: str = Field(
        default="",
        description=(
            "Full Chain of Thought reasoning trail. "
            "Documents the 5-step thinking process: "
            "assess → identify → personalise → formulate → self-check. "
            "Stored for audit and debugging."
        ),
    )

    # ── Sentiment & Sector Context ──
    sentiment_report: Optional[SentimentReport] = Field(
        default=None,
        description=(
            "News sentiment analysis from SentimentAnalysisAgent. "
            "None if sentiment analysis was skipped (quick depth)."
        ),
    )
    sector_analyses: list[SectorAnalysis] = Field(
        default_factory=list,
        description=(
            "Per-sector analysis with mood and advisor context. "
            "Sorted by absolute change_pct descending (most active first). "
            "Used by Module 4 to focus trade setups on active sectors."
        ),
    )

    # ── Key Signals for Downstream Modules ──
    top_opportunities: list[str] = Field(
        default_factory=list,
        description=(
            "Top 3–5 ticker symbols showing the strongest bullish signals. "
            "Module 4 uses these as primary candidates for trade setups. "
            "Example: ['HDFCBANK', 'RELIANCE', 'TCS']."
        ),
    )
    top_risks: list[str] = Field(
        default_factory=list,
        description=(
            "Tickers with highest risk signals (selling pressure, near stop loss). "
            "Module 3 uses these to flag positions for review. "
            "Example: ['AXISBANK', 'TATAMOTORS']."
        ),
    )
    risk_events: list[str] = Field(
        default_factory=list,
        description=(
            "Key risk events for today from sentiment analysis. "
            "Example: ['RBI Governor speech at 11 AM', 'US CPI data tonight']. "
            "Module 6 includes these in the morning brief."
        ),
    )

    # ── Metadata ──
    analysis_depth: AnalysisDepth = Field(
        default=AnalysisDepth.FULL,
        description="Whether this is a full or quick analysis.",
    )
    analysed_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this analysis was generated (IST).",
    )

    @model_validator(mode="after")
    def validate_advisor_quality(self) -> MarketAnalysis:
        """Basic structural validation — the quality checker does deep checks.

        Ensures the core advisor structure is present. This validator
        catches obvious structural issues at model creation time.
        The full quality gate (quality_checker.py) runs after this
        with configurable thresholds and retry logic.
        """
        if self.mood_confidence < 0 or self.mood_confidence > 1:
            raise ValueError(
                f"mood_confidence must be 0.0–1.0, got {self.mood_confidence}."
            )

        if self.market_mood == MarketMood.EXTREME_FEAR and self.mood_confidence < 0.7:
            # Extreme fear is a strong call — need strong confidence
            self.mood_confidence = max(self.mood_confidence, 0.7)

        return self


class AnalysisQualityReport(BaseModel):
    """Self-reflection quality check results.

    After every Claude API response, the system checks its own output
    quality. This report documents what passed, what failed, and
    whether the analysis is fit for the user.

    Quality thresholds (from Section 5 CoT Pattern):
      → situation: >= 100 characters
      → reasoning: >= 100 characters
      → action: >= 50 characters, must contain price level or concrete step
      → risk: >= 50 characters
      → lesson: >= 80 characters (if analysis_depth == full)
      → cot_reasoning: present and non-empty
      → No field contains "N/A" or "Not applicable"

    Self-reflection questions (from Master CoT Pattern):
      Q1: "Would a 20-year senior advisor be satisfied with this?"
      Q2: "Does this output have data + context + signal + advice?"
      Q3: "Is this personalised to the user's situation?"
      Q4: "Is this honest about uncertainty?"
    """

    verdict: QualityVerdict = Field(
        ...,
        description="Overall quality check result: passed, warning, or failed.",
    )

    # ── Field-Level Checks ──
    situation_ok: bool = Field(
        default=False,
        description="True if situation field meets the 100-char minimum.",
    )
    reasoning_ok: bool = Field(
        default=False,
        description="True if reasoning field meets the 100-char minimum.",
    )
    action_ok: bool = Field(
        default=False,
        description="True if action field meets the 50-char minimum.",
    )
    risk_ok: bool = Field(
        default=False,
        description="True if risk field meets the 50-char minimum.",
    )
    lesson_ok: bool = Field(
        default=False,
        description=(
            "True if lesson field meets the 80-char minimum. "
            "Automatically True for quick analysis depth."
        ),
    )
    cot_present: bool = Field(
        default=False,
        description="True if cot_reasoning is present and non-empty.",
    )
    no_na_fields: bool = Field(
        default=False,
        description="True if no field contains 'N/A' or 'Not applicable'.",
    )
    personalisation_present: bool = Field(
        default=False,
        description=(
            "True if user_impact references the user's specific situation. "
            "Checked by looking for user name or position references."
        ),
    )

    # ── Self-Reflection Answers ──
    advisor_satisfied: bool = Field(
        default=False,
        description="Q1: Would a 20-year senior advisor be satisfied?",
    )
    has_full_structure: bool = Field(
        default=False,
        description="Q2: Does output have data + context + signal + advice?",
    )
    is_personalised: bool = Field(
        default=False,
        description="Q3: Is this personalised to the user's situation?",
    )
    is_honest: bool = Field(
        default=True,
        description="Q4: Is this honest about uncertainty?",
    )

    # ── Detailed Issues ──
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields that are missing or empty when they should not be.",
    )
    shallow_fields: list[str] = Field(
        default_factory=list,
        description="Fields that exist but are below the quality threshold.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Detailed issue descriptions for debugging and retry logic.",
    )

    # ── Metadata ──
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this quality check was performed (IST).",
    )


class AnalysisResult(BaseModel):
    """Wrapper around MarketAnalysis with operational metadata.

    This is what the MCP tools return — the analysis plus
    metadata about how it was produced (tokens used, timing,
    cache status, quality report).

    Downstream modules receive this wrapper and can check:
      → Was this cached or freshly generated?
      → How many tokens did it cost?
      → Did the quality check pass?
      → How long did the analysis take?
    """

    analysis: MarketAnalysis = Field(
        ...,
        description="The actual MarketAnalysis produced by the advisor.",
    )
    quality_report: Optional[AnalysisQualityReport] = Field(
        default=None,
        description=(
            "Quality check results. None if quality check was skipped "
            "(should never happen in production)."
        ),
    )

    # ── Token Accounting ──
    input_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of input tokens sent to Claude API.",
    )
    output_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of output tokens received from Claude API.",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (input + output).",
    )

    # ── Timing ──
    api_latency_ms: int = Field(
        default=0,
        ge=0,
        description="Claude API call latency in milliseconds.",
    )
    total_latency_ms: int = Field(
        default=0,
        ge=0,
        description=(
            "Total analysis latency including data fetch, "
            "sentiment analysis, and quality check."
        ),
    )

    # ── Cache Status ──
    from_cache: bool = Field(
        default=False,
        description="True if this analysis was served from cache (no Claude API call).",
    )
    cache_key: str = Field(
        default="",
        description="Cache key used for this analysis. Empty if not cached.",
    )

    # ── Retry Info ──
    retry_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of Claude API retries needed. "
            "0 = first attempt succeeded. "
            ">= 2 = quality or parse issues required retries."
        ),
    )
    model_used: str = Field(
        default="claude-opus-4-5",
        description="Claude model used for this analysis.",
    )

    # ── Timestamp ──
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this result was assembled (IST).",
    )

    @model_validator(mode="after")
    def compute_total_tokens(self) -> AnalysisResult:
        """Auto-compute total_tokens from input + output."""
        if self.total_tokens == 0 and (self.input_tokens > 0 or self.output_tokens > 0):
            self.total_tokens = self.input_tokens + self.output_tokens
        return self
