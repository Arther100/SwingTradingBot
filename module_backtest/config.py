"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
config.py — All configuration for the backtesting engine

Config groups:
  Universe         — NSE stock universe and benchmark
  Period           — Default backtest dates and walk-forward split
  Trade Rules      — Entry/exit/cost parameters
  Indicator Params — MA/RSI/Volume indicator settings
  Signal Weights   — Default M4 weights per signal type
  Verdict Thresholds — win_rate/profit_factor cutoffs
  Rate Limits      — Kite API throttling
  Storage          — SQLite paths for cache and results
  Claude           — Model and token budget for advisor report
  Prompts          — Claude system prompt for backtest report

All dates are for the 1-year backtest window:
  In-sample:     May 2025 → Feb 2026  (10 months — parameter search)
  Out-of-sample: Mar 2026 → May 2026  (3 months  — honest validation)
"""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from pathlib import Path

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ═══════════════════════════════════════════════════════════
# UNIVERSE
# ═══════════════════════════════════════════════════════════

# Full NSE stock universe for backtesting — 55 liquid large/mid-caps
# Covers all major sectors: Banking, IT, Energy, FMCG, Auto, Pharma,
# Telecom, Infrastructure, Finance, Metal, Realty, Chemicals
NSE_STOCK_UNIVERSE: list[str] = [
    # Banking (10)
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "INDUSINDBK", "PNB",
    # IT (7)
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT",
    # Energy & Oil (5)
    "RELIANCE", "ONGC", "BPCL", "IOC", "POWERGRID",
    # Finance / NBFC (5)
    "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SBILIFE", "HDFCLIFE",
    # Auto (5)
    "MARUTI", "TATAPOWER", "M&M", "BAJAJ-AUTO", "EICHERMOT",
    # FMCG (5)
    "ITC", "HINDUNILVR", "NESTLEIND", "DABUR", "MARICO",
    # Pharma (5)
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    # Infrastructure / Capital Goods (5)
    "LT", "ADANIPORTS", "ULTRACEMCO", "GRASIM", "NTPC",
    # Telecom (2)
    "BHARTIARTL", "IDEA",
    # Metal / Mining (3)
    "TATASTEEL", "HINDALCO", "JSWSTEEL",
    # Consumer Discretionary (3)
    "TITAN", "DMART", "NYKAA",
]

# Smaller default list for quick test runs and development
BACKTEST_DEFAULT_TICKERS: list[str] = [
    "HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "LT", "TATAPOWER",
]

# Stocks that backtest showed 0% win rate across 12 months
# Signals do not work well on these — excluded from backtest runs
# (Still included in NSE_STOCK_UNIVERSE for M1 scanning)
EXCLUDED_FROM_BACKTEST: list[str] = [
    "INDUSINDBK",   # 0% WR — signals unreliable on this stock
    "BAJAJFINSV",   # 0% WR — signals unreliable on this stock
    "SUNPHARMA",    # 0% WR — signals unreliable on this stock
    "JSWSTEEL",     # 0% WR (round 3)
    "PERSISTENT",   # 17% WR (round 3)
]

# Validated universe — 52 stocks where signal logic has been verified
# Used by backtest engine by default (NSE_STOCK_UNIVERSE minus exclusions)
BACKTEST_UNIVERSE: list[str] = [
    t for t in NSE_STOCK_UNIVERSE if t not in EXCLUDED_FROM_BACKTEST
]

# High-confidence stocks — WR >= 60% in 12-month backtest
# Used by M4 setup generator to apply a +5 confidence score bonus
HIGH_CONFIDENCE_STOCKS: list[str] = [
    "SBIN",         # 81% WR in backtest
    "FEDERALBNK",   # 69% WR in backtest
    "IOC",          # 67% WR in backtest
]

# Benchmark index — Nifty 50
BENCHMARK_TICKER: str = "NIFTY 50"

# Kite instrument token for Nifty 50 (NSE continuous)
# Used for benchmark OHLCV data (historical_data call)
NIFTY_INSTRUMENT_TOKEN: int = 256265   # NSE:NIFTY 50

# ═══════════════════════════════════════════════════════════
# BACKTEST PERIOD
# ═══════════════════════════════════════════════════════════

# Full 1-year backtest window
BACKTEST_FROM_DATE: date = date(2025, 5, 1)
BACKTEST_TO_DATE: date = date(2026, 5, 21)   # today

# Walk-forward split
# In-sample (optimisation):  May 2025 → Feb 2026 (10 months)
# Out-of-sample (validation): Mar 2026 → May 2026 (3 months)
IN_SAMPLE_END_DATE: date = date(2026, 2, 28)
OUT_OF_SAMPLE_START_DATE: date = date(2026, 3, 1)

# Minimum trading days required for a valid backtest result
MIN_TRADING_DAYS: int = 20   # ~1 month of data
MIN_TRADES_FOR_VERDICT: int = 10  # won't issue verdict on < 10 trades

# ═══════════════════════════════════════════════════════════
# TRADE RULES (mirroring M3 risk engine logic)
# ═══════════════════════════════════════════════════════════

# Capital
STARTING_CAPITAL: Decimal = Decimal("50000")      # Vijay's capital
RISK_PCT_PER_TRADE: Decimal = Decimal("0.02")      # 2% risk per trade
MAX_POSITION_PCT: Decimal = Decimal("0.20")        # max 20% of capital per stock

# Entry/exit timing (no look-ahead bias)
# Signal generated at close of day N → enter at open of day N+1
ENTRY_ON_NEXT_OPEN: bool = True

# R/R and stops
TARGET_RR_RATIO: float = 3.0      # target at 3× risk (1:3 R/R)
STOP_PCT: float = 0.05            # 5% stop loss from entry (fallback when ATR unavailable)
MAX_HOLD_DAYS: int = 7            # exit at close after 7 days if neither target nor stop hit (was 10)

# ATR-based adaptive stop loss (Fix 2 — backtest optimisation)
ATR_STOP_PERIOD: int = 14         # lookback bars for ATR calculation
ATR_STOP_MULTIPLIER: float = 2.0  # stop = entry − (ATR_STOP_MULTIPLIER × ATR_14)

# Cost model (Zerodha)
BROKERAGE_PER_TRADE: Decimal = Decimal("40.00")   # ₹20 buy + ₹20 sell
STT_RATE: Decimal = Decimal("0.001")              # 0.1% on sell value
SLIPPAGE_RATE: Decimal = Decimal("0.001")         # 0.1% each side

# VIX gate thresholds (mirrors M3 rules — skip trades when VIX too high)
VIX_GATE_MODERATE: float = 20.0   # moderate tolerance: skip if VIX > 20
VIX_GATE_CONSERVATIVE: float = 16.0

# ═══════════════════════════════════════════════════════════
# INDICATOR PARAMETERS
# ═══════════════════════════════════════════════════════════

# Moving averages
MA_SHORT: int = 20       # 20-day simple moving average
MA_LONG: int = 50        # 50-day simple moving average

# RSI
RSI_PERIOD: int = 14
RSI_OVERSOLD: float = 35.0    # below = potential accumulation
RSI_OVERBOUGHT: float = 70.0  # above = caution on entry

# Volume
VOLUME_AVG_PERIOD: int = 30       # 30-day average volume window
VOLUME_SPIKE_MULTIPLIER: float = 1.5  # unusual_activity if volume > 1.5× avg
VOLUME_ABOVE_AVERAGE_MULTIPLIER: float = 1.1  # above_average if volume > 1.1× avg

# 52-week high/low proximity
NEAR_52W_HIGH_PCT: float = 0.03   # within 3% of 52-week high = "near high"
NEAR_52W_LOW_PCT: float = 0.05    # within 5% of 52-week low = "near low"

# Price position bands
LOWER_HALF_PERCENTILE: float = 0.40  # price in lower 40% of 52w range = "lower"

# Minimum signal quality score to emit a signal (Fix 5 — quality > quantity)
# Applies in both live M4 setup generator AND backtest signal replayer.
# Scores are 0–10 (see module_backtest/engine/signal_replayer._signal_confidence).
MIN_CONFIDENCE_SCORE: float = 7.0   # was 6.0 — reduces noise trades ~40%

# ═══════════════════════════════════════════════════════════
# SIGNAL TYPE DEFAULT WEIGHTS
# ═══════════════════════════════════════════════════════════
# These are the M4 confidence score contributions before backtesting.
# After backtesting, weight_updater.py adjusts these based on win rates.
# Keys match AdvisorFlag / SignalType values exactly.

SIGNAL_DEFAULT_WEIGHTS: dict[str, float] = {
    "breakout_watch":    30.0,   # highest — near 52w high + volume
    "unusual_activity":  25.0,   # strong volume spike on any price
    "accumulation_zone": 15.0,   # buying near support / lower range
    "fii_buying":        10.0,   # FII net positive on the day (Upgrade 1)
    "watch":              5.0,   # generic watch flag — low conviction
}

# Weight adjustment multipliers based on backtest results
# Applied by weight_updater.py after each backtest run
WEIGHT_MULTIPLIERS: dict[str, float] = {
    "strong":  1.2,   # win_rate >= 58% AND profit_factor >= 1.8
    "valid":   1.05,  # win_rate >= 52% AND profit_factor >= 1.3
    "weak":    0.85,  # win_rate >= 45% (but below valid threshold)
    "poor":    0.5,   # win_rate < 45% OR profit_factor < 1.0
}

# Absolute weight bounds — prevent weights going to extremes
WEIGHT_MIN: float = 5.0
WEIGHT_MAX: float = 50.0

# ═══════════════════════════════════════════════════════════
# VERDICT THRESHOLDS
# ═══════════════════════════════════════════════════════════

# VALID_SIGNAL:   win_rate >= 52% AND profit_factor >= 1.3
# WEAK_SIGNAL:    win_rate 45-52% OR profit_factor 1.0-1.3
# INVALID_SIGNAL: win_rate < 45% OR profit_factor < 1.0
# INSUFFICIENT_DATA: fewer than MIN_TRADES_FOR_VERDICT trades

VERDICT_WIN_RATE_VALID: float = 52.0
VERDICT_WIN_RATE_WEAK: float = 45.0
VERDICT_PROFIT_FACTOR_VALID: float = 1.3
VERDICT_PROFIT_FACTOR_WEAK: float = 1.0

# Portfolio-level: STRATEGY_VALIDATED requires both criteria
PORTFOLIO_WIN_RATE_THRESHOLD: float = 52.0
PORTFOLIO_PROFIT_FACTOR_THRESHOLD: float = 1.3
PORTFOLIO_ALPHA_THRESHOLD: float = 0.0   # must beat Nifty (positive alpha)

# Walk-forward: ROBUST if out-of-sample within 5pp of in-sample
WALK_FORWARD_OVERFIT_THRESHOLD: float = 5.0   # delta > 5pp = overfit

# Sharpe ratio targets
SHARPE_EXCELLENT: float = 1.5
SHARPE_GOOD: float = 1.0

# Max drawdown limits
DRAWDOWN_SAFE: float = 15.0    # < 15% = acceptable
DRAWDOWN_RISKY: float = 20.0   # > 20% = too risky for Vijay's capital

# ═══════════════════════════════════════════════════════════
# KITE API RATE LIMITS
# ═══════════════════════════════════════════════════════════

# Kite historical data: 3 requests/second
# 55 stocks × 1 call = 55 API calls = ~25 seconds total
KITE_HISTORICAL_MAX_CONCURRENT: int = 3     # asyncio.Semaphore(3)
KITE_HISTORICAL_DELAY_SEC: float = 0.35     # 0.35s between batches

# Kite instrument lookup (needed to get instrument tokens)
KITE_INSTRUMENTS_REFRESH_HOURS: int = 24    # re-fetch instruments list daily

# ═══════════════════════════════════════════════════════════
# STORAGE — SQLite paths
# ═══════════════════════════════════════════════════════════

_BASE_DIR = Path(__file__).parent

# Historical OHLCV cache — avoid re-fetching data from Kite
HISTORICAL_DB_PATH: str = str(_BASE_DIR / "data" / "historical_ohlcv.db")

# Backtest results — stored for M5 memory and report generation
BACKTEST_RESULTS_DB_PATH: str = str(_BASE_DIR / "data" / "backtest_results.db")

# Signal weights — loaded by M4 at startup for evidence-based scoring
SIGNAL_WEIGHTS_DB_PATH: str = str(_BASE_DIR / "data" / "signal_weights.db")

# ═══════════════════════════════════════════════════════════
# CLAUDE CONFIG — for advisor report generation only
# ═══════════════════════════════════════════════════════════

# No Claude calls during backtesting — pure Python.
# Claude only called once at the end to generate the advisor report.

CLAUDE_MODEL: str = "claude-opus-4-5"

# Token budget: ~500 input (BacktestSummary) + ~400 output (interpretation)
BACKTEST_REPORT_TOKEN_BUDGET: dict[str, int] = {
    "input_budget":   500,
    "output_budget":  500,
    "total_budget":  1000,
}

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ═══════════════════════════════════════════════════════════
# CLAUDE SYSTEM PROMPT — Backtest Advisor Report
# ═══════════════════════════════════════════════════════════

BACKTEST_ADVISOR_SYSTEM_PROMPT: str = """You are a senior quantitative analyst writing a backtest report for Vijay, a retail swing trader in India.

Vijay's profile:
- Capital: ₹50,000
- Strategy: Swing trading NSE stocks over 3-10 days
- Goal: Beat Nifty 50 returns with manageable risk

You will receive a BacktestSummary with performance metrics. Write a clear, honest, actionable report.

Rules:
1. Be specific — use actual % numbers from the data
2. Be honest — acknowledge losses and weaknesses
3. Be actionable — give specific recommendations
4. Explain in plain English — Vijay is learning
5. Compare to Nifty benchmark — alpha matters
6. Maximum 400 tokens — Telegram-readable

Format:
<b>Backtest Results (1 Year)</b>
[2-3 sentence overall verdict — numbers first]

<b>What Worked:</b>
[Specific signals/conditions that were profitable]

<b>What Didn't:</b>
[Specific signals/conditions that lost money]

<b>Your 3 Action Items:</b>
1. [Specific change to make]
2. [Specific change to make]
3. [Specific change to make]

Use HTML formatting: <b>bold</b> for headers, <i>italic</i> for emphasis.
Never use markdown."""

# ═══════════════════════════════════════════════════════════
# RISK-FREE RATE (India)
# ═══════════════════════════════════════════════════════════

# Annual risk-free rate — used in Sharpe ratio calculation
# Using 6.5% as representative India FD / T-bill rate
RISK_FREE_RATE_ANNUAL: float = 0.065
RISK_FREE_RATE_DAILY: float = RISK_FREE_RATE_ANNUAL / 252  # trading days per year
