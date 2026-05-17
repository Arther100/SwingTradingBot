"""
SwingAdvisorBot — Module 3: Risk Management Engine
models.py — Pydantic models for risk assessment

All financial values use Decimal for precision.
A trader who loses 50% needs 100% gain to recover.
Risk management is not optional — it is survival.

Models:
  TradeProposal   → Input: what the user wants to trade
  RiskReport      → Output: APPROVED / REJECTED / REDUCE_SIZE
  PortfolioRiskReport → Portfolio-level risk snapshot
  VixGateStatus   → Quick VIX gate check result
  OpenPosition    → Single open position record

Enums:
  RiskVerdict     → APPROVED, REJECTED, REDUCE_SIZE
  RiskTolerance   → conservative, moderate, aggressive
  VixGateResult   → open, closed

Data flow:
  TradeProposal (from M4)
    → RiskAssessmentAgent (10-step CoT)
    → RiskReport (to M4 for trade setup)
"""

from __future__ import annotations

import enum
import json
from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


# ═══════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════


class RiskVerdict(str, enum.Enum):
    """Risk assessment verdict for a trade proposal.

    APPROVED    → Trade meets all risk criteria. Proceed.
    REJECTED    → Trade violates one or more risk rules. Do not proceed.
    REDUCE_SIZE → Trade direction is correct but position is too large.
    """

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REDUCE_SIZE = "REDUCE_SIZE"


class RiskTolerance(str, enum.Enum):
    """User's self-declared risk tolerance level.

    Determines VIX gate thresholds and max risk per trade:
      conservative → VIX < 15, max 1% risk per trade
      moderate     → VIX < 20, max 2% risk per trade
      aggressive   → VIX < 25, max 3% risk per trade
    """

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class VixGateResult(str, enum.Enum):
    """VIX gate status — whether new swing trades are allowed.

    open   → VIX within tolerance, new trades allowed
    closed → VIX too high, no new swing trades
    """

    OPEN = "open"
    CLOSED = "closed"


# ═══════════════════════════════════════════════════════════
# INPUT MODELS
# ═══════════════════════════════════════════════════════════


class OpenPosition(BaseModel):
    """A single open swing trade position.

    Used for portfolio-level risk calculations:
    sector exposure, total capital at risk, open trade count.

    Example:
        OpenPosition(
            ticker="HDFCBANK",
            sector="Banking",
            entry_price=Decimal("1623.00"),
            quantity=13,
            stop_loss=Decimal("1548.00"),
            target=Decimal("1900.00"),
            entry_date=datetime(2026, 5, 10, tzinfo=IST),
        )
    """

    ticker: str = Field(
        ...,
        description="NSE ticker symbol, e.g. HDFCBANK",
    )
    sector: str = Field(
        default="Other",
        description="Business sector, e.g. Banking, IT, Energy",
    )
    entry_price: Decimal = Field(
        ...,
        description="Entry price in INR (Decimal precision)",
    )
    quantity: int = Field(
        ...,
        ge=1,
        description="Number of shares held",
    )
    stop_loss: Decimal = Field(
        ...,
        description="Stop loss price in INR",
    )
    target: Decimal = Field(
        ...,
        description="Target price in INR",
    )
    entry_date: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When the position was opened (IST)",
    )

    @property
    def position_value(self) -> Decimal:
        """Total position value = entry_price × quantity."""
        return self.entry_price * self.quantity

    @property
    def risk_amount(self) -> Decimal:
        """Total risk = (entry_price - stop_loss) × quantity."""
        return (self.entry_price - self.stop_loss) * self.quantity


class TradeProposal(BaseModel):
    """Input to the risk engine — what the user wants to trade.

    Comes from Module 4 (Trade Setup) or directly from user input.
    All prices are Decimal for financial precision.

    Example:
        TradeProposal(
            ticker="HDFCBANK",
            entry_price=Decimal("1623.00"),
            target_price=Decimal("1900.00"),
            stop_loss=Decimal("1548.00"),
            user_id="XCU700",
        )
    """

    ticker: str = Field(
        ...,
        description="NSE ticker symbol, e.g. HDFCBANK, TCS, RELIANCE",
    )
    entry_price: Decimal = Field(
        ...,
        gt=0,
        description="Proposed entry price in INR",
    )
    target_price: Decimal = Field(
        ...,
        gt=0,
        description="Proposed target price in INR",
    )
    stop_loss: Decimal = Field(
        ...,
        gt=0,
        description="Proposed stop loss price in INR",
    )
    user_id: str = Field(
        default="XCU700",
        description="Zerodha client ID",
    )
    requested_shares: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of shares the user wants to buy. "
            "If None, position calculator determines optimal size."
        ),
    )

    @property
    def risk_per_share(self) -> Decimal:
        """Risk per share = entry_price - stop_loss."""
        return self.entry_price - self.stop_loss

    @property
    def gain_per_share(self) -> Decimal:
        """Gain per share = target_price - entry_price."""
        return self.target_price - self.entry_price


# ═══════════════════════════════════════════════════════════
# OUTPUT MODELS
# ═══════════════════════════════════════════════════════════


class RiskReport(BaseModel):
    """Complete risk assessment output — the gatekeeper's verdict.

    Every field is populated with exact rupee amounts.
    Every verdict includes plain English reasoning.
    No trade passes without a complete RiskReport.

    Token target: APPROVED ~300, REJECTED ~250, REDUCE_SIZE ~280.

    Example (APPROVED):
        {
            "ticker": "HDFCBANK",
            "verdict": "APPROVED",
            "position_size_shares": 13,
            "position_size_rupees": "21099.00",
            "risk_per_share": "75.00",
            "total_risk_rupees": "975.00",
            "risk_reward_ratio": "1:3.69",
            "advisor_note": "Solid risk management..."
        }
    """

    # ── Core Verdict ──
    ticker: str = Field(
        ...,
        description="NSE ticker symbol being assessed",
    )
    verdict: RiskVerdict = Field(
        ...,
        description="APPROVED, REJECTED, or REDUCE_SIZE",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description=(
            "Specific reason for rejection: vix_gate_failed, "
            "risk_reward_below_minimum, risk_pct_exceeded, "
            "position_size_exceeded, sector_overexposure, "
            "max_trades_reached, insufficient_capital"
        ),
    )

    # ── Position Sizing ──
    position_size_shares: int = Field(
        default=0,
        ge=0,
        description="Approved number of shares",
    )
    position_size_rupees: Optional[Decimal] = Field(
        default=None,
        description="Total position value in INR = shares × entry_price",
    )
    position_pct_of_capital: Optional[Decimal] = Field(
        default=None,
        description="Position as percentage of total capital",
    )

    # ── Risk Metrics ──
    risk_per_share: Optional[Decimal] = Field(
        default=None,
        description="Risk per share = entry - stop_loss (INR)",
    )
    total_risk_rupees: Optional[Decimal] = Field(
        default=None,
        description="Total risk = shares × risk_per_share (INR)",
    )
    risk_pct_of_capital: Optional[Decimal] = Field(
        default=None,
        description="Total risk as percentage of capital",
    )

    # ── Reward Metrics ──
    potential_gain_rupees: Optional[Decimal] = Field(
        default=None,
        description="Total potential gain = shares × gain_per_share (INR)",
    )
    risk_reward_ratio: Optional[str] = Field(
        default=None,
        description="Risk/reward ratio as string, e.g. '1:3.69'",
    )

    # ── VIX Assessment ──
    vix_value: Optional[Decimal] = Field(
        default=None,
        description="India VIX value at time of assessment",
    )
    vix_limit: Optional[Decimal] = Field(
        default=None,
        description="VIX gate threshold for user's risk tolerance",
    )
    vix_signal: Optional[str] = Field(
        default=None,
        description="VIX classification: low_fear, moderate_fear, high_fear, extreme_fear",
    )

    # ── Checks ──
    checks_passed: list[str] = Field(
        default_factory=list,
        description=(
            "List of passed check names: vix_gate_passed, "
            "risk_pct_within_limit, risk_reward_above_minimum, "
            "position_size_within_limit, sector_exposure_within_limit, "
            "open_trades_within_limit"
        ),
    )
    checks_failed: list[str] = Field(
        default_factory=list,
        description="List of failed check names with reasons",
    )

    # ── REDUCE_SIZE specific fields ──
    requested_shares: Optional[int] = Field(
        default=None,
        description="Original requested shares (for REDUCE_SIZE)",
    )
    approved_shares: Optional[int] = Field(
        default=None,
        description="Approved (reduced) shares (for REDUCE_SIZE)",
    )
    requested_risk_rupees: Optional[Decimal] = Field(
        default=None,
        description="Risk at requested size (for REDUCE_SIZE)",
    )
    approved_risk_rupees: Optional[Decimal] = Field(
        default=None,
        description="Risk at approved size (for REDUCE_SIZE)",
    )
    risk_pct_at_requested: Optional[Decimal] = Field(
        default=None,
        description="Risk % at requested size (for REDUCE_SIZE)",
    )
    risk_pct_at_approved: Optional[Decimal] = Field(
        default=None,
        description="Risk % at approved size (for REDUCE_SIZE)",
    )

    # ── REJECTED suggestion fields ──
    suggested_target: Optional[Decimal] = Field(
        default=None,
        description="Suggested target price for acceptable R/R (for REJECTED)",
    )
    minimum_required: Optional[str] = Field(
        default=None,
        description="Minimum required ratio, e.g. '1:2.0' (for REJECTED)",
    )
    gain_per_share: Optional[Decimal] = Field(
        default=None,
        description="Gain per share at proposed target (for REJECTED context)",
    )

    # ── Sector Exposure (for sector rejection) ──
    sector: Optional[str] = Field(
        default=None,
        description="Sector name for exposure check",
    )
    current_exposure_pct: Optional[Decimal] = Field(
        default=None,
        description="Current sector exposure before this trade (%)",
    )
    max_exposure_pct: Optional[Decimal] = Field(
        default=None,
        description="Maximum allowed sector exposure (%)",
    )
    suggested_alternatives: Optional[list[str]] = Field(
        default=None,
        description="Alternative tickers from different sectors",
    )

    # ── Reasoning ──
    cot_reasoning: Optional[str] = Field(
        default=None,
        description="Full 10-step Chain of Thought reasoning trail",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description=(
            "Plain English advisor note — personalised, uses user's name, "
            "explains verdict with exact rupee amounts"
        ),
    )

    # ── Metadata ──
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this risk assessment was performed (IST)",
    )

    def to_prompt_context(self) -> str:
        """Return trimmed version for M4 Claude prompt.

        Includes: verdict, key numbers, advisor_note only.
        Excludes: full cot_reasoning (too verbose for M4).
        Serializes with exclude_none=True.

        Returns:
            JSON string under 300 tokens for APPROVED,
            under 250 tokens for REJECTED.
        """
        summary: dict = {
            "ticker": self.ticker,
            "verdict": self.verdict.value,
            "position_shares": self.position_size_shares,
            "position_rupees": str(self.position_size_rupees)
            if self.position_size_rupees is not None
            else None,
            "risk_rupees": str(self.total_risk_rupees)
            if self.total_risk_rupees is not None
            else None,
            "risk_reward": self.risk_reward_ratio,
            "advisor_note": self.advisor_note,
        }

        if self.rejection_reason:
            summary["rejection_reason"] = self.rejection_reason
            summary["suggested_fix"] = (
                str(self.suggested_target)
                if self.suggested_target is not None
                else str(self.approved_shares)
                if self.approved_shares is not None
                else None
            )

        # Strip None values
        summary = {k: v for k, v in summary.items() if v is not None}
        return json.dumps(summary, ensure_ascii=False)


class PortfolioRiskReport(BaseModel):
    """Portfolio-level risk snapshot.

    Shows total capital at risk, sector exposures,
    open trade count. Used by check_portfolio_risk MCP tool.

    Example:
        {
            "total_capital": "50000.00",
            "total_invested": "21099.00",
            "available_capital": "28901.00",
            "total_risk_rupees": "975.00",
            "total_risk_pct": "1.95",
            "sector_exposures": {"Banking": "42.20"},
            "open_trade_count": 1,
            "max_trades": 5
        }
    """

    # ── Capital Summary ──
    total_capital: Decimal = Field(
        ...,
        description="Total trading capital in INR",
    )
    total_invested: Decimal = Field(
        default=Decimal("0.00"),
        description="Total value of open positions in INR",
    )
    available_capital: Decimal = Field(
        ...,
        description="Capital available for new trades in INR",
    )

    # ── Risk Summary ──
    total_risk_rupees: Decimal = Field(
        default=Decimal("0.00"),
        description="Total capital at risk across all open positions (INR)",
    )
    total_risk_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Total risk as percentage of capital",
    )

    # ── Sector Exposure ──
    sector_exposures: dict[str, Decimal] = Field(
        default_factory=dict,
        description=(
            "Sector exposure percentages. "
            "Key: sector name, Value: exposure % of capital"
        ),
    )

    # ── Trade Count ──
    open_trade_count: int = Field(
        default=0,
        ge=0,
        description="Number of currently open positions",
    )
    max_trades: int = Field(
        default=5,
        description="Maximum allowed concurrent trades",
    )

    # ── Positions Detail ──
    positions: list[OpenPosition] = Field(
        default_factory=list,
        description="List of all open positions",
    )

    # ── Metadata ──
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this portfolio snapshot was taken (IST)",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description="Plain English portfolio risk summary",
    )


class VixGateStatus(BaseModel):
    """Quick VIX gate check result.

    Call this FIRST before generating any trade setups.
    If gate is closed → skip M4 entirely.

    Example (gate open):
        {
            "vix_value": "14.2",
            "vix_limit": "20.0",
            "tolerance": "moderate",
            "gate": "open",
            "vix_signal": "low_fear",
            "advisor_note": "VIX at 14.2 — low fear. Safe for new trades."
        }
    """

    vix_value: Decimal = Field(
        ...,
        description="Current India VIX value",
    )
    vix_limit: Decimal = Field(
        ...,
        description="VIX gate threshold for user's risk tolerance",
    )
    tolerance: RiskTolerance = Field(
        default=RiskTolerance.MODERATE,
        description="User's risk tolerance level",
    )
    gate: VixGateResult = Field(
        ...,
        description="Gate status: open or closed",
    )
    vix_signal: str = Field(
        ...,
        description="VIX classification: low_fear, moderate_fear, high_fear, extreme_fear",
    )
    advisor_note: str = Field(
        ...,
        description="Plain English VIX assessment",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this VIX check was performed (IST)",
    )
