"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
engine/walk_forward.py — Walk-forward validation to detect overfitting

The walk-forward test splits the backtest period into:
  In-sample:      May 2025 → Feb 2026 (10 months — "training" / optimisation)
  Out-of-sample:  Mar 2026 → May 2026 (3 months  — "test" / honest validation)

Why this matters:
  Most retail backtests fit parameters to historical data (overfitting).
  A strategy that shows 58% win rate in-sample but only 42% out-of-sample
  is NOT a real edge — it memorised the past and will fail live.

  This module catches that. If out-of-sample win_rate drops ≥ 5pp from
  in-sample win_rate, the strategy is flagged as OVERFIT.

Zero look-ahead guarantee (critical):
  Indicators are ALWAYS built from the full bar history (to preserve
  warm-up accuracy for bars near the split boundary). However, signals
  and trades are strictly partitioned by date:
    in-sample  window: signal_date ∈ [in_sample_start,  in_sample_end]
    out-of-sample:     signal_date ∈ [oos_start,        oos_end]
  The trade-simulation uses only bars that are available as of entry_date —
  there is no look-ahead at any point.

Verdict:
  ROBUST          — |win_rate_delta| < 5pp AND oos win_rate ≥ 50%
  DEGRADED        — oos win_rate < in-sample but delta < 5pp
  OVERFIT         — win_rate_delta ≤ −5pp (in-sample was ≥5pp better)
  INSUFFICIENT DATA — fewer than MIN_TRADES_FOR_VERDICT in either window

Usage:
    bars       = await historical_fetcher.fetch("HDFCBANK", from_date, to_date)
    nifty_bars = await historical_fetcher.fetch_nifty(from_date, to_date)
    result = WalkForwardEngine.run(
        ticker="HDFCBANK",
        bars=bars,
        signal_type=SignalType.BREAKOUT_WATCH,
        config=BacktestConfig(...),
        nifty_bars=nifty_bars,
    )
    print(result.verdict)   # "ROBUST" or "OVERFIT"
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from module_backtest.config import (
    BACKTEST_FROM_DATE,
    BACKTEST_TO_DATE,
    IN_SAMPLE_END_DATE,
    MIN_TRADES_FOR_VERDICT,
    OUT_OF_SAMPLE_START_DATE,
    WALK_FORWARD_OVERFIT_THRESHOLD,
)
from module_backtest.engine.metrics_calculator import MetricsCalculator
from module_backtest.engine.signal_replayer import SignalReplayer
from module_backtest.engine.trade_simulator import TradeSimulator
from module_backtest.models import (
    BacktestConfig,
    ExitReason,
    OHLCVBar,
    PerformanceMetrics,
    SignalType,
    WalkForwardResult,
)

logger = logging.getLogger("swing_advisor.backtest.walk_forward")

# ── Verdict strings ───────────────────────────────────────
_VERDICT_ROBUST = "ROBUST"
_VERDICT_DEGRADED = "DEGRADED"
_VERDICT_OVERFIT = "OVERFIT"
_VERDICT_INSUFFICIENT = "INSUFFICIENT DATA"


class WalkForwardEngine:
    """Runs walk-forward validation for a single (ticker, signal_type) pair.

    Stateless — all methods are @classmethods.
    No I/O — operates purely on already-fetched OHLCVBar lists.
    """

    # ── Public API ──────────────────────────────────────────

    @classmethod
    def run(
        cls,
        ticker: str,
        bars: list[OHLCVBar],
        signal_type: SignalType,
        config: Optional[BacktestConfig] = None,
        nifty_bars: Optional[list[OHLCVBar]] = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation for one ticker + signal_type.

        Args:
            ticker:      NSE ticker symbol.
            bars:        Full chronological OHLCVBar list (full period).
            signal_type: Which signal type to validate.
            config:      BacktestConfig with period dates and trade rules.
                         If None, uses module defaults.
            nifty_bars:  Optional Nifty 50 bars for benchmark comparison.

        Returns:
            WalkForwardResult with in/out-of-sample metrics and verdict.
        """
        # Resolve split dates from config or module defaults
        is_start, is_end, oos_start, oos_end = cls._resolve_dates(config)

        # Build simulator and calculator (shared across both windows)
        simulator = TradeSimulator(config)
        calculator = MetricsCalculator(
            starting_capital=(
                config.starting_capital
                if config
                else __import__(
                    "module_backtest.config", fromlist=["STARTING_CAPITAL"]
                ).STARTING_CAPITAL
            )
        )

        # ── In-sample window ──────────────────────────────────
        is_signals = SignalReplayer.replay_date_range(
            ticker, bars, is_start, is_end, signal_types=[signal_type], nifty_bars=nifty_bars
        )
        is_trades_all = simulator.simulate_all(is_signals, bars, nifty_bars)
        is_trades = [
            t for t in is_trades_all
            if t.exit_reason != ExitReason.NEVER_ENTERED
        ]
        is_metrics = calculator.compute(is_trades_all)

        # ── Out-of-sample window ──────────────────────────────
        oos_signals = SignalReplayer.replay_date_range(
            ticker, bars, oos_start, oos_end, signal_types=[signal_type], nifty_bars=nifty_bars
        )
        oos_trades_all = simulator.simulate_all(oos_signals, bars, nifty_bars)
        oos_trades = [
            t for t in oos_trades_all
            if t.exit_reason != ExitReason.NEVER_ENTERED
        ]
        oos_metrics = calculator.compute(oos_trades_all)

        # ── Compute deltas ────────────────────────────────────
        win_rate_delta = round(
            oos_metrics.win_rate - is_metrics.win_rate, 2
        )
        profit_factor_delta = round(
            oos_metrics.profit_factor - is_metrics.profit_factor, 4
        )

        # ── Verdict logic ─────────────────────────────────────
        verdict_str = cls._compute_verdict(
            is_metrics, oos_metrics, win_rate_delta
        )
        is_overfit = verdict_str == _VERDICT_OVERFIT
        is_robust = verdict_str == _VERDICT_ROBUST

        logger.info(
            f"[{ticker}:{signal_type.value}] Walk-forward — "
            f"IS win_rate={is_metrics.win_rate:.1f}% ({len(is_trades)} trades), "
            f"OOS win_rate={oos_metrics.win_rate:.1f}% ({len(oos_trades)} trades), "
            f"delta={win_rate_delta:+.1f}pp → {verdict_str}"
        )

        return WalkForwardResult(
            signal_type=signal_type,
            ticker=ticker,
            in_sample_start=is_start,
            in_sample_end=is_end,
            out_of_sample_start=oos_start,
            out_of_sample_end=oos_end,
            in_sample=is_metrics,
            out_of_sample=oos_metrics,
            win_rate_delta=win_rate_delta,
            profit_factor_delta=profit_factor_delta,
            is_overfit=is_overfit,
            is_robust=is_robust,
            verdict=verdict_str,
        )

    @classmethod
    def run_all_signals(
        cls,
        ticker: str,
        bars: list[OHLCVBar],
        config: Optional[BacktestConfig] = None,
        nifty_bars: Optional[list[OHLCVBar]] = None,
    ) -> list[WalkForwardResult]:
        """Run walk-forward for all 4 signal types on one ticker.

        Args:
            ticker:     NSE ticker symbol.
            bars:       Full chronological OHLCVBar list.
            config:     BacktestConfig (optional — module defaults used if None).
            nifty_bars: Optional Nifty 50 bars.

        Returns:
            List of WalkForwardResult, one per SignalType (4 results).
        """
        signal_types = [
            SignalType.BREAKOUT_WATCH,
            SignalType.ACCUMULATION_ZONE,
            SignalType.UNUSUAL_ACTIVITY,
            SignalType.FII_BUYING,
        ]
        results = []
        for st in signal_types:
            result = cls.run(ticker, bars, st, config=config, nifty_bars=nifty_bars)
            results.append(result)
        return results

    @classmethod
    def summary_table(cls, results: list[WalkForwardResult]) -> str:
        """Produce a compact ASCII table for logging / Telegram preview.

        Example output:
            HDFCBANK Walk-Forward Summary
            Signal              IS Win%  OOS Win%  Delta   Verdict
            breakout_watch      58.3%    54.1%     -4.2pp  ROBUST
            accumulation_zone   52.0%    41.0%     -11.0pp OVERFIT
            unusual_activity    47.5%    49.0%     +1.5pp  DEGRADED
            fii_buying          60.0%    58.0%     -2.0pp  ROBUST
        """
        if not results:
            return "(no walk-forward results)"

        ticker = results[0].ticker
        lines = [
            f"{ticker} Walk-Forward Summary",
            f"{'Signal':<22} {'IS Win%':>8}  {'OOS Win%':>9}  {'Delta':>7}  Verdict",
            "-" * 62,
        ]
        for r in results:
            sig = r.signal_type.value
            lines.append(
                f"{sig:<22} {r.in_sample.win_rate:>7.1f}%  "
                f"{r.out_of_sample.win_rate:>8.1f}%  "
                f"{r.win_rate_delta:>+6.1f}pp  {r.verdict}"
            )
        return "\n".join(lines)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _resolve_dates(
        config: Optional[BacktestConfig],
    ) -> tuple[date, date, date, date]:
        """Extract walk-forward split dates from config or module defaults."""
        if config is not None and config.in_sample_end_date is not None:
            is_start = config.from_date
            is_end = config.in_sample_end_date
            # out-of-sample starts day after in-sample ends
            oos_start = OUT_OF_SAMPLE_START_DATE  # from config module
            oos_end = config.to_date
        else:
            is_start = BACKTEST_FROM_DATE
            is_end = IN_SAMPLE_END_DATE
            oos_start = OUT_OF_SAMPLE_START_DATE
            oos_end = BACKTEST_TO_DATE

        return is_start, is_end, oos_start, oos_end

    @staticmethod
    def _compute_verdict(
        is_metrics: PerformanceMetrics,
        oos_metrics: PerformanceMetrics,
        win_rate_delta: float,
    ) -> str:
        """Determine ROBUST / DEGRADED / OVERFIT / INSUFFICIENT DATA."""
        is_n = is_metrics.total_trades
        oos_n = oos_metrics.total_trades

        if is_n < MIN_TRADES_FOR_VERDICT or oos_n < MIN_TRADES_FOR_VERDICT:
            return _VERDICT_INSUFFICIENT

        # Overfit: out-of-sample degraded by ≥ WALK_FORWARD_OVERFIT_THRESHOLD
        if win_rate_delta <= -WALK_FORWARD_OVERFIT_THRESHOLD:
            return _VERDICT_OVERFIT

        # Robust: small delta AND oos win_rate ≥ 50%
        if (
            abs(win_rate_delta) < WALK_FORWARD_OVERFIT_THRESHOLD
            and oos_metrics.win_rate >= 50.0
        ):
            return _VERDICT_ROBUST

        # Otherwise: some degradation but not catastrophic
        return _VERDICT_DEGRADED
