"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
models.py — All Pydantic v2 data models for backtesting

Models:
  OHLCVBar              → Single daily price bar (from Kite)
  TradeCosts            → Realistic cost calculator (brokerage + STT + slippage)
  TradeSimulation       → One simulated historical trade (entry → exit)
  BacktestConfig        → Parameters for a backtest run
  PerformanceMetrics    → All quantitative metrics in one place
  BacktestResult        → Per-signal per-ticker result with advisor verdict
  PortfolioBacktestResult → Aggregate result across all tickers
  SignalWeight          → M4 confidence weight (updated by backtesting)
  WalkForwardResult     → In-sample vs out-of-sample comparison
  BacktestRun           → Metadata about a complete backtest execution

Enums:
  SignalType      → breakout_watch, accumulation_zone, unusual_activity, fii_buying
  ExitReason      → target_hit, stop_hit, timeout
  AdvisorVerdict  → VALID_SIGNAL, WEAK_SIGNAL, INVALID_SIGNAL, STRATEGY_VALIDATED

Data flow:
  Kite OHLCV → indicator_builder → signal_replayer
    → trade_simulator → metrics_calculator
    → BacktestResult → report_generator → Telegram

No look-ahead bias. Costs included. Walk-forward tested.
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

# Pydantic v2.12 rejects a field named 'date' when the type annotation is also
# 'date' (from datetime). The field name shadows the type in the class namespace.
# Using a private alias for the type avoids the clash while keeping the field
# name and runtime behaviour identical.
_Date = date

from pydantic import BaseModel, Field, model_validator
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


# ═══════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════


class SignalType(str, enum.Enum):
    """M1 advisor flag / signal types that the backtest evaluates.

    Maps directly to AdvisorFlag values from module1_data_layer.
    """

    BREAKOUT_WATCH = "breakout_watch"
    ACCUMULATION_ZONE = "accumulation_zone"
    UNUSUAL_ACTIVITY = "unusual_activity"
    FII_BUYING = "fii_buying"
    ALL = "all"  # aggregate across all signal types


class ExitReason(str, enum.Enum):
    """Why a simulated trade was closed."""

    TARGET_HIT = "target_hit"       # High touched or exceeded target price
    STOP_HIT = "stop_hit"           # Low touched or went below stop price
    TIMEOUT = "timeout"             # max_hold_days reached, exit at close
    NEVER_ENTERED = "never_entered" # Position could not be entered (no next-day open)


class AdvisorVerdict(str, enum.Enum):
    """Signal quality verdict based on backtest results.

    Used in both per-signal and portfolio-level results.
    """

    VALID_SIGNAL = "VALID SIGNAL"           # win_rate >= 52% AND profit_factor >= 1.3
    WEAK_SIGNAL = "WEAK SIGNAL"             # win_rate 45-52% OR profit_factor 1.0-1.3
    INVALID_SIGNAL = "INVALID SIGNAL"       # win_rate < 45% OR profit_factor < 1.0
    STRATEGY_VALIDATED = "STRATEGY VALIDATED"  # portfolio-level pass
    INSUFFICIENT_DATA = "INSUFFICIENT DATA" # too few trades to conclude


# ═══════════════════════════════════════════════════════════
# PRICE BAR
# ═══════════════════════════════════════════════════════════


class OHLCVBar(BaseModel):
    """Single daily OHLCV price bar from Kite historical data.

    Fields match Kite historical_data() response exactly.
    date is stored as date (not datetime) for daily interval.
    """

    date: _Date = Field(..., description="Trading date (IST)")
    open: float = Field(..., description="Opening price (INR)")
    high: float = Field(..., description="Intraday high (INR)")
    low: float = Field(..., description="Intraday low (INR)")
    close: float = Field(..., description="Closing price (INR)")
    volume: int = Field(default=0, ge=0, description="Total traded volume")

    @model_validator(mode="after")
    def validate_ohlcv(self) -> "OHLCVBar":
        """Verify OHLCV internal consistency."""
        if self.high < self.low:
            raise ValueError(
                f"high ({self.high}) must be >= low ({self.low}) "
                f"on {self.date}"
            )
        if self.high < self.open or self.high < self.close:
            raise ValueError(
                f"high ({self.high}) must be >= open ({self.open}) "
                f"and close ({self.close}) on {self.date}"
            )
        if self.low > self.open or self.low > self.close:
            raise ValueError(
                f"low ({self.low}) must be <= open ({self.open}) "
                f"and close ({self.close}) on {self.date}"
            )
        return self


# ═══════════════════════════════════════════════════════════
# TRADE COSTS
# ═══════════════════════════════════════════════════════════


class TradeCosts(BaseModel):
    """Zerodha-realistic cost model for backtesting.

    Every simulated trade deducts these costs from P&L.
    Using Decimal for accuracy on financial arithmetic.

    Breakdown per trade (example: 13 shares @ ₹769):
      Brokerage: ₹40 (₹20 buy + ₹20 sell)
      STT:       ₹10 (0.1% on sell side)
      Slippage:  ₹20 buy (0.1%) + ₹17 sell (0.1%)
      Total:     ₹87
    """

    brokerage_per_trade: Decimal = Field(
        default=Decimal("40.00"),
        description="Round-trip brokerage (₹20 buy + ₹20 sell — Zerodha flat)",
    )
    stt_rate: Decimal = Field(
        default=Decimal("0.001"),
        description="Securities Transaction Tax — 0.1% on sell side value",
    )
    slippage_rate: Decimal = Field(
        default=Decimal("0.001"),
        description="0.1% each side — buyer pays more, seller gets less",
    )

    def calculate(
        self,
        entry_price: Decimal,
        exit_price: Decimal,
        shares: int,
    ) -> Decimal:
        """Calculate total round-trip cost for a trade.

        Args:
            entry_price: Price paid per share on entry.
            exit_price:  Price received per share on exit.
            shares:      Number of shares traded.

        Returns:
            Total cost in ₹ (Decimal, rounded to 2dp).
        """
        entry_value = entry_price * shares
        exit_value = exit_price * shares

        stt = exit_value * self.stt_rate
        slippage_buy = entry_value * self.slippage_rate
        slippage_sell = exit_value * self.slippage_rate

        total = self.brokerage_per_trade + stt + slippage_buy + slippage_sell
        return total.quantize(Decimal("0.01"))


# ═══════════════════════════════════════════════════════════
# TRADE SIMULATION — one historical trade
# ═══════════════════════════════════════════════════════════


class TradeSimulation(BaseModel):
    """One simulated historical trade from signal detection to exit.

    Entry rules (no look-ahead bias):
      - Signal generated at close of signal_date
      - Enter at open of the next trading day (entry_date)
      - Target = entry × (1 + target_pct)
      - Stop  = entry × (1 - stop_pct)
      - Check each subsequent bar: high >= target → WIN
                                    low  <= stop  → LOSS
      - After max_hold_days → EXIT at close (timeout)

    Costs applied:
      - Slippage on entry (paid more than open)
      - Slippage on exit (received less than target)
      - Brokerage (flat ₹40 round trip)
      - STT (0.1% on sell value)
    """

    # ── Identity ──
    ticker: str = Field(..., description="NSE ticker symbol")
    signal_type: SignalType = Field(
        ..., description="Which M4 signal generated this trade"
    )
    signal_date: date = Field(
        ..., description="Date the signal was generated (close of day)"
    )

    # ── Entry ──
    entry_date: date = Field(
        ..., description="Date entered (next trading day after signal_date)"
    )
    entry_price: Decimal = Field(
        ..., description="Actual entry price including slippage (INR)"
    )
    shares: int = Field(
        default=0, ge=0, description="Shares purchased (from M3 position sizing)"
    )
    target_price: Decimal = Field(
        ..., description="Target exit price (entry × target_rr_multiplier)"
    )
    stop_price: Decimal = Field(
        ..., description="Stop loss price (entry × (1 - stop_pct))"
    )

    # ── Exit ──
    exit_date: Optional[date] = Field(
        default=None, description="Date the position was closed"
    )
    exit_price: Optional[Decimal] = Field(
        default=None, description="Actual exit price including slippage (INR)"
    )
    exit_reason: ExitReason = Field(
        default=ExitReason.NEVER_ENTERED,
        description="Why the trade was closed",
    )
    hold_days: int = Field(
        default=0, ge=0, description="Calendar days between entry and exit"
    )

    # ── P&L ──
    gross_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Profit/loss before costs in ₹",
    )
    total_costs: Decimal = Field(
        default=Decimal("0"),
        description="Total brokerage + STT + slippage in ₹",
    )
    net_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Net profit/loss after all costs in ₹",
    )
    return_pct: float = Field(
        default=0.0,
        description="Net return as % of entry position value",
    )
    is_win: bool = Field(
        default=False,
        description="True if net_pnl > 0",
    )

    # ── Context ──
    nifty_return_pct: Optional[float] = Field(
        default=None,
        description="Nifty 50 return over same hold period (for alpha calc)",
    )


# ═══════════════════════════════════════════════════════════
# BACKTEST CONFIG
# ═══════════════════════════════════════════════════════════


class BacktestConfig(BaseModel):
    """Parameters that define a backtest run.

    Immutable once created — ensures reproducible results.
    """

    # ── Universe ──
    tickers: list[str] = Field(
        default_factory=list,
        description="NSE tickers to backtest",
    )
    signal_types: list[SignalType] = Field(
        default_factory=lambda: [
            SignalType.BREAKOUT_WATCH,
            SignalType.ACCUMULATION_ZONE,
            SignalType.UNUSUAL_ACTIVITY,
        ],
        description="Which signal types to backtest",
    )

    # ── Period ──
    from_date: date = Field(..., description="Backtest start date (inclusive)")
    to_date: date = Field(..., description="Backtest end date (inclusive)")

    # ── Walk-forward split ──
    in_sample_end_date: Optional[date] = Field(
        default=None,
        description=(
            "End of in-sample period. "
            "None = single period (no walk-forward). "
            "Typically set to 10 months from start — "
            "e.g. from_date=May-25, in_sample_end=Feb-26, to_date=May-26."
        ),
    )

    # ── Trade rules ──
    starting_capital: Decimal = Field(
        default=Decimal("50000"),
        description="Starting capital in ₹ (matches Vijay's account)",
    )
    risk_pct_per_trade: Decimal = Field(
        default=Decimal("0.02"),
        description="Risk 2% of capital per trade (M3 rule)",
    )
    target_rr_ratio: float = Field(
        default=3.0,
        description="Target R/R ratio — 3 = target at 3× risk",
    )
    stop_pct: float = Field(
        default=0.05,
        description="Stop loss as fraction of entry price (5%)",
    )
    max_hold_days: int = Field(
        default=10,
        ge=1,
        description="Maximum holding period before forced exit",
    )

    # ── Costs ──
    costs: TradeCosts = Field(
        default_factory=TradeCosts,
        description="Cost model (brokerage + STT + slippage)",
    )

    # ── Benchmark ──
    benchmark_ticker: str = Field(
        default="NIFTY 50",
        description="Benchmark index for alpha calculation",
    )
    risk_free_rate: float = Field(
        default=0.065,
        description="Annual risk-free rate (India FD rate ~6.5%)",
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "BacktestConfig":
        if self.from_date >= self.to_date:
            raise ValueError(
                f"from_date ({self.from_date}) must be before "
                f"to_date ({self.to_date})"
            )
        if self.in_sample_end_date:
            if not (self.from_date < self.in_sample_end_date < self.to_date):
                raise ValueError(
                    "in_sample_end_date must be strictly between "
                    "from_date and to_date"
                )
        return self


# ═══════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════


class PerformanceMetrics(BaseModel):
    """Quantitative performance metrics for a set of trades.

    Reused in both per-signal results and portfolio-level aggregates.
    All metrics are explained in plain English (for Vijay's reports).
    """

    # ── Trade counts ──
    total_trades: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    timeouts: int = Field(default=0, ge=0, description="Trades exited by timeout (no win/loss)")

    # ── Win rate ──
    win_rate: float = Field(
        default=0.0,
        description="wins / total_trades × 100. Target: >= 52%",
    )

    # ── Return metrics ──
    avg_win_pct: float = Field(
        default=0.0,
        description="Average return % on winning trades",
    )
    avg_loss_pct: float = Field(
        default=0.0,
        description="Average return % on losing trades (negative number)",
    )
    avg_hold_days: float = Field(
        default=0.0,
        description="Mean holding period in days. Target: 3-8 for swing trading.",
    )
    total_return_pct: float = Field(
        default=0.0,
        description="Total portfolio return % over backtest period",
    )

    # ── Risk metrics ──
    profit_factor: float = Field(
        default=0.0,
        description=(
            "total_gross_profit / total_gross_loss. "
            "Target: >= 1.3. Above 2.0 = excellent. "
            "Plain English: For every ₹1 lost, bot made ₹X."
        ),
    )
    max_drawdown_pct: float = Field(
        default=0.0,
        description=(
            "Largest peak-to-trough capital decline as %. "
            "Target: < 15%. Above 20% = too risky."
        ),
    )
    max_drawdown_period: Optional[str] = Field(
        default=None,
        description="Month/period of worst drawdown e.g. 'Nov 2025'",
    )
    sharpe_ratio: float = Field(
        default=0.0,
        description=(
            "(avg_return - risk_free_rate) / std_deviation. "
            "Target: >= 1.0. Above 1.5 = excellent."
        ),
    )

    # ── Benchmark ──
    nifty_return_pct: Optional[float] = Field(
        default=None,
        description="Nifty 50 return over same period (for alpha calc)",
    )
    alpha: Optional[float] = Field(
        default=None,
        description=(
            "strategy_return - nifty_return. "
            "Target: positive (beating the index)."
        ),
    )

    # ── Best/worst months ──
    best_month: Optional[str] = Field(
        default=None,
        description="Best performing month e.g. 'Feb 2026 (+8.4%)'",
    )
    worst_month: Optional[str] = Field(
        default=None,
        description="Worst performing month e.g. 'Nov 2025 (-3.1%)'",
    )
    avg_trades_per_month: Optional[float] = Field(
        default=None,
        description="Mean trade count per calendar month",
    )


# ═══════════════════════════════════════════════════════════
# BACKTEST RESULT — per signal per ticker
# ═══════════════════════════════════════════════════════════


class BacktestResult(BaseModel):
    """Complete backtest result for one signal type on one ticker.

    Example (VALID SIGNAL):
        signal_type=breakout_watch, ticker=HDFCBANK
        57.9% win rate, 1.89 profit factor → VALID SIGNAL

    Example (WEAK SIGNAL):
        signal_type=unusual_activity, ticker=SBIN
        41.2% win rate, 0.84 profit factor → WEAK SIGNAL

    Carries full trade list so any claim can be verified.
    """

    # ── Identity ──
    signal_type: SignalType = Field(
        ..., description="Which signal was backtested"
    )
    ticker: str = Field(..., description="NSE ticker symbol")
    period_start: date = Field(..., description="Backtest start date")
    period_end: date = Field(..., description="Backtest end date")

    # ── Signal counts ──
    total_signals: int = Field(
        default=0, ge=0,
        description="Total times the signal fired during the period",
    )
    trades_taken: int = Field(
        default=0, ge=0,
        description="Signals that became trades (VIX gate open)",
    )
    trades_skipped: int = Field(
        default=0, ge=0,
        description="Signals skipped (VIX gate closed or no capital)",
    )
    skip_reason: Optional[str] = Field(
        default=None,
        description="Most common skip reason e.g. 'VIX gate closed'",
    )

    # ── Full metrics ──
    metrics: PerformanceMetrics = Field(
        default_factory=PerformanceMetrics,
        description="All quantitative performance metrics",
    )

    # ── Notable trades ──
    best_trade: Optional[TradeSimulation] = Field(
        default=None,
        description="Highest return trade in the period",
    )
    worst_trade: Optional[TradeSimulation] = Field(
        default=None,
        description="Lowest return (most negative) trade in the period",
    )

    # ── Walk-forward results (if walk-forward was run) ──
    in_sample_metrics: Optional[PerformanceMetrics] = Field(
        default=None,
        description="Metrics on in-sample (training) period",
    )
    out_of_sample_metrics: Optional[PerformanceMetrics] = Field(
        default=None,
        description="Metrics on out-of-sample (test) period. "
                    "Compare to in_sample_metrics — divergence = overfit.",
    )
    is_overfit: Optional[bool] = Field(
        default=None,
        description=(
            "True if in-sample win_rate > out-of-sample win_rate by >= 5pp. "
            "Overfit strategies should not be traded live."
        ),
    )

    # ── Verdict ──
    advisor_verdict: AdvisorVerdict = Field(
        default=AdvisorVerdict.INSUFFICIENT_DATA,
        description="Signal quality verdict",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description=(
            "2-3 sentence plain English interpretation. "
            "References actual % numbers. Gives actionable recommendation."
        ),
    )

    # ── Timestamp ──
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this result was calculated (IST)",
    )

    # ── Raw trades (stored separately in DB, summarised here) ──
    total_gross_profit: Decimal = Field(
        default=Decimal("0"),
        description="Sum of all winning trade net P&L in ₹",
    )
    total_gross_loss: Decimal = Field(
        default=Decimal("0"),
        description="Sum of all losing trade net P&L in ₹ (negative)",
    )
    ending_capital: Optional[Decimal] = Field(
        default=None,
        description="Capital at end of period starting from starting_capital",
    )


# ═══════════════════════════════════════════════════════════
# PORTFOLIO BACKTEST RESULT — aggregate across all tickers
# ═══════════════════════════════════════════════════════════


class PortfolioBacktestResult(BaseModel):
    """Aggregate backtest result across all tickers and signal types.

    This is the top-level output Vijay sees:
    "Your strategy returned 36.8% vs Nifty's 12.4%"

    Example:
        {
          "total_return_pct": 36.8,
          "nifty_return_pct": 12.4,
          "alpha": 24.4,
          "win_rate": 54.0,
          "advisor_verdict": "STRATEGY VALIDATED"
        }
    """

    # ── Period ──
    period_start: date = Field(..., description="Backtest start date")
    period_end: date = Field(..., description="Backtest end date")

    # ── Capital ──
    starting_capital: Decimal = Field(
        ..., description="Starting capital in ₹"
    )
    ending_capital: Decimal = Field(
        ..., description="Ending capital in ₹ after all trades + costs"
    )

    # ── Aggregate metrics ──
    metrics: PerformanceMetrics = Field(
        default_factory=PerformanceMetrics,
        description="Portfolio-level performance metrics",
    )

    # ── Per-signal breakdown ──
    signal_results: list[BacktestResult] = Field(
        default_factory=list,
        description="Individual BacktestResult for each signal × ticker combination",
    )

    # ── Per-ticker summary ──
    tickers_tested: list[str] = Field(
        default_factory=list,
        description="All tickers included in this backtest",
    )
    best_ticker: Optional[str] = Field(
        default=None,
        description="Ticker with highest total return over period",
    )
    worst_ticker: Optional[str] = Field(
        default=None,
        description="Ticker with lowest total return over period",
    )

    # ── Verdict ──
    advisor_verdict: AdvisorVerdict = Field(
        default=AdvisorVerdict.INSUFFICIENT_DATA,
        description="Overall strategy verdict",
    )
    advisor_note: Optional[str] = Field(
        default=None,
        description="Claude-generated portfolio-level interpretation (plain English)",
    )
    telegram_text: Optional[str] = Field(
        default=None,
        description="Pre-formatted Telegram HTML message for delivery to Vijay",
    )

    # ── Timestamp ──
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this result was calculated (IST)",
    )


# ═══════════════════════════════════════════════════════════
# SIGNAL WEIGHT — M4 confidence weight (updated by backtesting)
# ═══════════════════════════════════════════════════════════


class SignalWeight(BaseModel):
    """Backtest-driven confidence weight for one signal type.

    After backtesting, win_rate and profit_factor determine
    a multiplier that adjusts the default M4 confidence score
    contribution. Stored in SQLite and loaded by M4 at startup.

    Example:
        breakout_watch: default=30, win_rate=58% → multiplier=1.1 → current=33
        unusual_activity: default=25, win_rate=41% → multiplier=0.5 → current=12.5
    """

    signal_type: SignalType = Field(..., description="Which signal this weight applies to")
    default_weight: float = Field(
        ...,
        description="Original M4 confidence weight before any backtesting",
    )
    current_weight: float = Field(
        ...,
        description="Current evidence-based weight after backtest adjustment",
    )
    multiplier: float = Field(
        default=1.0,
        description="current_weight / default_weight",
    )

    # ── Evidence ──
    win_rate: Optional[float] = Field(
        default=None,
        description="Win rate from most recent backtest (%)",
    )
    profit_factor: Optional[float] = Field(
        default=None,
        description="Profit factor from most recent backtest",
    )
    sample_size: int = Field(
        default=0,
        ge=0,
        description="Number of trades the weight is based on",
    )

    # ── Metadata ──
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(IST),
        description="When this weight was last updated (IST)",
    )
    backtest_period: Optional[str] = Field(
        default=None,
        description="Period the backtest covered e.g. 'May 2025 – May 2026'",
    )


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD RESULT
# ═══════════════════════════════════════════════════════════


class WalkForwardResult(BaseModel):
    """Side-by-side comparison of in-sample vs out-of-sample results.

    The honest test — most retail backtests skip this.
    Ours doesn't.

    If in-sample win_rate = 58% but out-of-sample = 48%
    → delta = -10pp → strategy is OVERFIT → do not trade.

    If in-sample win_rate = 55% and out-of-sample = 52%
    → delta = -3pp → strategy is ROBUST → worth trading.
    """

    signal_type: SignalType = Field(...)
    ticker: str = Field(...)

    # ── Period boundaries ──
    in_sample_start: date = Field(...)
    in_sample_end: date = Field(...)
    out_of_sample_start: date = Field(...)
    out_of_sample_end: date = Field(...)

    # ── Results ──
    in_sample: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    out_of_sample: PerformanceMetrics = Field(default_factory=PerformanceMetrics)

    # ── Computed comparison ──
    win_rate_delta: float = Field(
        default=0.0,
        description="out_of_sample.win_rate - in_sample.win_rate (negative = degraded)",
    )
    profit_factor_delta: float = Field(
        default=0.0,
        description="out_of_sample.profit_factor - in_sample.profit_factor",
    )
    is_overfit: bool = Field(
        default=False,
        description=(
            "True if win_rate_delta <= -5pp (in-sample was significantly better). "
            "Overfit = strategy found patterns that don't generalise."
        ),
    )
    is_robust: bool = Field(
        default=False,
        description=(
            "True if |win_rate_delta| < 5pp AND out_of_sample.win_rate >= 50%. "
            "Robust = results hold on unseen data."
        ),
    )
    verdict: str = Field(
        default="",
        description="ROBUST / OVERFIT / DEGRADED / INSUFFICIENT DATA",
    )
