"""
SwingAdvisorBot — Module 5: Memory & Personalization
models.py — Pydantic models for user memory and RAG pipeline

This module transforms SwingAdvisorBot from a generic tool into
Vijay's personal finance advisor. Every model here represents
a piece of persistent memory — the advisor remembers everything.

Models:
  UserProfile       → User identity, capital, risk tolerance, stats
  TradeRecord       → Every trade with full lifecycle (open → closed)
  LearningProgress  → Concepts taught, quiz scores, repetition
  WatchlistItem     → User's watchlist entries with alert levels
  DailyStats        → Daily performance snapshot
  RetrievedChunk    → A single chunk returned from Pinecone
  MemoryContext     → Assembled context for Claude prompt (≤300 tokens)
  VerificationResult → Output from 2-round verification

Enums:
  TradeStatus       → open, closed, stopped_out
  ExitReason        → target_hit, stop_hit, manual
  SetupSource       → m4_generated, manual
  MemoryNamespace   → trade_memory, market_patterns, etc.

Data flow:
  SQLite → structured facts (profile, trades, stats)
  Pinecone → semantic search (embeddings + metadata)
  ContextBuilder → assemble ≤300 tokens for Claude
  VerificationEngine → 2-round check on generated advice

All financial amounts are Decimal. IST timestamps.
exclude_none=True on all serialization.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, computed_field
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def _new_id() -> str:
    """Generate a short UUID for primary keys."""
    return uuid.uuid4().hex[:16]


# ═══════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════


class TradeStatus(str, enum.Enum):
    """Lifecycle status of a trade."""

    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"


class ExitReason(str, enum.Enum):
    """Why a trade was closed."""

    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    MANUAL = "manual"


class SetupSource(str, enum.Enum):
    """Where the trade setup originated."""

    M4_GENERATED = "m4_generated"
    MANUAL = "manual"


class MemoryNamespace(str, enum.Enum):
    """Pinecone namespace identifiers for different data types."""

    TRADE_MEMORY = "trade_memory"
    MARKET_PATTERNS = "market_patterns"
    CONVERSATIONS = "conversations"
    LESSONS = "lessons"
    KNOWLEDGE_BASE = "knowledge_base"


# ═══════════════════════════════════════════════════════════
# USER PROFILE — Who the advisor is talking to
# ═══════════════════════════════════════════════════════════


class UserProfile(BaseModel):
    """Complete user profile stored in SQLite.

    This is the advisor's memory of who Vijay is —
    capital, risk appetite, track record, preferences.

    Loaded before every Claude call. The profile summary
    always fits in the first 80 tokens of memory context.

    Example:
        {
          "user_id": "XCU700",
          "name": "Vijay",
          "capital": "50000.00",
          "risk_tolerance": "moderate",
          "total_trades": 15,
          "winning_trades": 9,
          "total_pnl": "3200.00"
        }
    """

    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID",
    )
    name: str = Field(
        default="Vijay",
        description="User's display name",
    )
    capital: Decimal = Field(
        default=Decimal("50000.00"),
        description="Total trading capital in INR",
    )
    risk_tolerance: str = Field(
        default="moderate",
        description="conservative / moderate / aggressive",
    )
    total_trades: int = Field(
        default=0,
        ge=0,
        description="Total trades taken (closed + open)",
    )
    winning_trades: int = Field(
        default=0,
        ge=0,
        description="Trades closed with positive P&L",
    )
    total_pnl: Decimal = Field(
        default=Decimal("0.00"),
        description="Cumulative P&L across all closed trades",
    )
    open_positions_count: int = Field(
        default=0,
        ge=0,
        description="Number of currently open positions",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this profile was first created (IST)",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="Last profile update (IST)",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def win_rate(self) -> float:
        """Win rate as percentage. 0.0 if no trades."""
        if self.total_trades == 0:
            return 0.0
        return round(self.winning_trades / self.total_trades * 100, 1)

    def to_context_summary(self) -> str:
        """Build profile summary for Claude prompt (~80 tokens).

        This is always the first section of memory context.
        """
        return (
            f"User: {self.name}. "
            f"Capital: ₹{self.capital:,.0f}. "
            f"Risk: {self.risk_tolerance}. "
            f"Win rate: {self.win_rate}% ({self.winning_trades}/{self.total_trades}). "
            f"Open positions: {self.open_positions_count}. "
            f"Total P&L: ₹{self.total_pnl:,.2f}."
        )


# ═══════════════════════════════════════════════════════════
# TRADE RECORD — Every trade the user takes
# ═══════════════════════════════════════════════════════════


class TradeRecord(BaseModel):
    """A single trade with full lifecycle tracking.

    Stored in SQLite (structured) and Pinecone (embedding).
    Financial fields are Decimal — stored as string in SQLite.

    Lifecycle:
      Entry: status=open, exit fields null
      Exit:  status=closed/stopped_out, exit fields populated

    Example (open):
        {"ticker": "HDFCBANK", "entry_price": "769.55",
         "shares": 13, "status": "open"}

    Example (closed):
        {"ticker": "HDFCBANK", "entry_price": "769.55",
         "exit_price": "888.24", "pnl_rupees": "1543.00",
         "status": "closed", "exit_reason": "target_hit"}
    """

    trade_id: str = Field(
        default_factory=_new_id,
        description="Unique trade identifier",
    )
    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID",
    )
    ticker: str = Field(
        ...,
        description="NSE ticker symbol",
    )
    sector: str = Field(
        default="Other",
        description="Business sector",
    )

    # ── Entry ──
    entry_price: Decimal = Field(
        ...,
        description="Entry price in INR",
    )
    stop_loss: Decimal = Field(
        ...,
        description="Stop loss price in INR",
    )
    target_price: Decimal = Field(
        ...,
        description="Target price in INR",
    )
    shares: int = Field(
        ...,
        ge=1,
        description="Number of shares",
    )
    entry_date: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When position was opened (IST)",
    )

    # ── Exit (nullable — populated on close) ──
    exit_price: Optional[Decimal] = Field(
        default=None,
        description="Exit price in INR",
    )
    exit_date: Optional[datetime] = Field(
        default=None,
        description="When position was closed (IST)",
    )
    exit_reason: Optional[ExitReason] = Field(
        default=None,
        description="Why the trade was closed",
    )

    # ── Lifecycle ──
    status: TradeStatus = Field(
        default=TradeStatus.OPEN,
        description="Current trade status",
    )
    pnl_rupees: Optional[Decimal] = Field(
        default=None,
        description="Realized P&L in INR (after close)",
    )
    pnl_pct: Optional[Decimal] = Field(
        default=None,
        description="Realized P&L as percentage",
    )

    # ── Context at time of entry ──
    market_mood: Optional[str] = Field(
        default=None,
        description="M2 market mood when trade was entered",
    )
    vix_at_entry: Optional[Decimal] = Field(
        default=None,
        description="India VIX when trade was entered",
    )
    setup_source: SetupSource = Field(
        default=SetupSource.M4_GENERATED,
        description="Where the setup came from",
    )
    notes: Optional[str] = Field(
        default=None,
        description="User or advisor notes on this trade",
    )

    def to_embedding_text(self) -> str:
        """Build text for Pinecone embedding (~1 chunk).

        Contains enough context for semantic similarity search.
        """
        outcome = ""
        if self.status != TradeStatus.OPEN and self.exit_price is not None:
            direction = "profit" if (self.pnl_rupees or 0) > 0 else "loss"
            outcome = (
                f" Exited at ₹{self.exit_price} ({self.exit_reason.value if self.exit_reason else 'unknown'}). "
                f"P&L: ₹{self.pnl_rupees} ({direction})."
            )

        mood = f" Market was {self.market_mood}." if self.market_mood else ""
        vix = f" VIX was {self.vix_at_entry}." if self.vix_at_entry else ""

        return (
            f"Bought {self.ticker} ({self.sector}) at ₹{self.entry_price} "
            f"on {self.entry_date.strftime('%Y-%m-%d')}. "
            f"Stop: ₹{self.stop_loss}, Target: ₹{self.target_price}. "
            f"{self.shares} shares.{outcome}{mood}{vix}"
        )

    def to_embedding_metadata(self) -> dict:
        """Build metadata dict for Pinecone storage."""
        meta: dict = {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "sector": self.sector,
            "status": self.status.value,
            "entry_date": self.entry_date.isoformat(),
            "entry_price": str(self.entry_price),
        }
        if self.pnl_rupees is not None:
            meta["pnl_rupees"] = str(self.pnl_rupees)
        if self.market_mood:
            meta["market_mood"] = self.market_mood
        if self.exit_reason:
            meta["exit_reason"] = self.exit_reason.value
        return meta


# ═══════════════════════════════════════════════════════════
# LEARNING PROGRESS — What the user has been taught
# ═══════════════════════════════════════════════════════════


class LearningProgress(BaseModel):
    """Tracks a single concept the user has been taught.

    Used by M7 (Education) to avoid repeating lessons
    and to track quiz performance over time.
    """

    progress_id: str = Field(
        default_factory=_new_id,
        description="Unique progress record ID",
    )
    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID",
    )
    concept: str = Field(
        ...,
        description="Concept name, e.g. 'stop_loss', 'position_sizing'",
    )
    taught_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When first taught (IST)",
    )
    quiz_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Quiz score 0-100 (null if not quizzed)",
    )
    times_taught: int = Field(
        default=1,
        ge=1,
        description="How many times this concept was taught",
    )
    last_taught: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="Most recent teaching (IST)",
    )


# ═══════════════════════════════════════════════════════════
# WATCHLIST ITEM — Stocks the user is watching
# ═══════════════════════════════════════════════════════════


class WatchlistItem(BaseModel):
    """A stock on the user's watchlist."""

    watchlist_id: str = Field(
        default_factory=_new_id,
        description="Unique watchlist entry ID",
    )
    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID",
    )
    ticker: str = Field(
        ...,
        description="NSE ticker symbol",
    )
    added_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When added to watchlist (IST)",
    )
    alert_price: Optional[Decimal] = Field(
        default=None,
        description="Price level to alert user",
    )
    notes: Optional[str] = Field(
        default=None,
        description="User notes for this watchlist entry",
    )


# ═══════════════════════════════════════════════════════════
# DAILY STATS — Daily performance snapshot
# ═══════════════════════════════════════════════════════════


class DailyStats(BaseModel):
    """Daily trading performance record."""

    stat_id: str = Field(
        default_factory=_new_id,
        description="Unique stat record ID",
    )
    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID",
    )
    date: str = Field(
        ...,
        description="Date in YYYY-MM-DD format (IST)",
    )
    total_trades: int = Field(
        default=0,
        ge=0,
        description="Trades closed on this day",
    )
    winning_trades: int = Field(
        default=0,
        ge=0,
        description="Winning trades on this day",
    )
    total_pnl: Decimal = Field(
        default=Decimal("0.00"),
        description="Total P&L for this day",
    )
    win_rate: Decimal = Field(
        default=Decimal("0.00"),
        description="Win rate for this day",
    )
    best_trade: Optional[str] = Field(
        default=None,
        description="Best performing ticker",
    )
    worst_trade: Optional[str] = Field(
        default=None,
        description="Worst performing ticker",
    )


# ═══════════════════════════════════════════════════════════
# RETRIEVED CHUNK — Single Pinecone search result
# ═══════════════════════════════════════════════════════════


class RetrievedChunk(BaseModel):
    """A single chunk returned from Pinecone similarity search.

    Contains the original text, similarity score,
    metadata, and which namespace it came from.
    """

    content: str = Field(
        ...,
        description="Original text content of the chunk",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score from Pinecone",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Pinecone metadata (ticker, date, etc.)",
    )
    namespace: str = Field(
        ...,
        description="Pinecone namespace this chunk came from",
    )


# ═══════════════════════════════════════════════════════════
# MEMORY CONTEXT — Assembled context for Claude prompt
# ═══════════════════════════════════════════════════════════


class MemoryContext(BaseModel):
    """Assembled memory context injected into Claude prompts.

    Hard limit: 300 tokens. This budget was reserved in M2.

    Contains profile summary + relevant chunks,
    trimmed to fit within budget.

    Example:
        {
          "text": "User: Vijay. Capital: ₹50,000. ...",
          "token_estimate": 280,
          "chunks_used": 3,
          "user_id": "XCU700"
        }
    """

    text: str = Field(
        default="",
        description="Assembled context string for Claude prompt",
    )
    token_estimate: int = Field(
        default=0,
        ge=0,
        description="Estimated token count (must be ≤300)",
    )
    chunks_used: int = Field(
        default=0,
        ge=0,
        description="Number of retrieved chunks included",
    )
    user_id: str = Field(
        default="XCU700",
        description="User this context was built for",
    )
    profile_included: bool = Field(
        default=False,
        description="Whether user profile summary is included",
    )

    @property
    def is_empty(self) -> bool:
        """Check if context has any content."""
        return len(self.text.strip()) == 0

    @property
    def within_budget(self) -> bool:
        """Check if context is within 300 token budget."""
        return self.token_estimate <= 300


# ═══════════════════════════════════════════════════════════
# VERIFICATION RESULT — 2-round check output
# ═══════════════════════════════════════════════════════════


class VerificationResult(BaseModel):
    """Output from 2-round verification of generated advice.

    Round 1: Generate advice (M2/M4)
    Round 2: Verify against user history + risk rules
    """

    verified: bool = Field(
        ...,
        description="Whether advice passed verification",
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="List of issues detected",
    )
    corrected_advice: Optional[str] = Field(
        default=None,
        description="Corrected version if issues found",
    )
    verification_note: Optional[str] = Field(
        default=None,
        description="Brief explanation of verification outcome",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Verification confidence (< 0.7 → flag for review)",
    )
    original_advice: Optional[str] = Field(
        default=None,
        description="Original advice before verification",
    )

    @property
    def needs_review(self) -> bool:
        """Whether this result should be flagged for user review."""
        return self.confidence < 0.7

    @property
    def final_advice(self) -> Optional[str]:
        """Return corrected advice if available, else original."""
        if self.corrected_advice:
            return self.corrected_advice
        return self.original_advice
