"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
engine/metrics_calculator.py — Quantitative performance metrics computation

Computes PerformanceMetrics from a list[TradeSimulation] using only
standard library math (no pandas/numpy).

Metrics computed:
  Trade counts     — total, wins, losses, timeouts
  Win rate         — wins / total_trades × 100
  Return metrics   — avg_win_pct, avg_loss_pct, avg_hold_days, total_return_pct
  Profit factor    — gross_profit / gross_loss
  Max drawdown     — peak-to-trough on sequential equity curve
  Sharpe ratio     — (mean_return - risk_free) / std(returns) × sqrt(252/avg_hold)
  Benchmark        — avg nifty_return_pct across trades; alpha = strategy - nifty
  Monthly          — best/worst month, avg_trades_per_month

Verdict logic (from config.py thresholds):
  VALID_SIGNAL      — win_rate >= 52% AND profit_factor >= 1.3
  WEAK_SIGNAL       — win_rate 45–52% OR profit_factor 1.0–1.3
  INVALID_SIGNAL    — win_rate < 45% AND profit_factor < 1.0
  INSUFFICIENT_DATA — fewer than MIN_TRADES_FOR_VERDICT (10) entered trades

NEVER_ENTERED trades are excluded from all metric calculations.
They are counted separately and available from the BacktestResult layer.

Usage:
    calculator = MetricsCalculator()
    metrics = calculator.compute(trades)
    verdict, note = calculator.verdict(metrics)
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Optional

from module_backtest.config import (
    MIN_TRADES_FOR_VERDICT,
    RISK_FREE_RATE_ANNUAL,
    VERDICT_PROFIT_FACTOR_VALID,
    VERDICT_PROFIT_FACTOR_WEAK,
    VERDICT_WIN_RATE_VALID,
    VERDICT_WIN_RATE_WEAK,
)
from module_backtest.models import (
    AdvisorVerdict,
    ExitReason,
    PerformanceMetrics,
    TradeSimulation,
)


class MetricsCalculator:
    """Computes PerformanceMetrics from a list of completed trades.

    Stateless — all state lives in the input TradeSimulation list.
    Create once and reuse for multiple compute() calls.

    Usage:
        calculator = MetricsCalculator(starting_capital=Decimal("50000"))
        metrics = calculator.compute(trades)
        verdict, note = calculator.verdict(metrics)
    """

    def __init__(
        self,
        starting_capital: Decimal = Decimal("50000"),
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
    ) -> None:
        self._starting_capital = starting_capital
        self._risk_free_annual = risk_free_rate
        self._risk_free_daily = risk_free_rate / 252.0

    # ── Public API ──────────────────────────────────────────

    def compute(self, trades: list[TradeSimulation]) -> PerformanceMetrics:
        """Compute full PerformanceMetrics from a trade list.

        Args:
            trades: All TradeSimulation objects including NEVER_ENTERED.
                    NEVER_ENTERED trades are filtered out before computation.

        Returns:
            PerformanceMetrics with all fields populated.
            Returns empty metrics (all zeros/None) for empty or all-never-entered input.
        """
        # Filter to entered trades only
        entered = [
            t for t in trades
            if t.exit_reason != ExitReason.NEVER_ENTERED
        ]

        if not entered:
            return PerformanceMetrics()

        # ── Partition by outcome ──────────────────────────────
        wins = [t for t in entered if t.is_win]
        losses = [t for t in entered if not t.is_win and t.exit_reason == ExitReason.STOP_HIT]
        timeouts = [t for t in entered if t.exit_reason == ExitReason.TIMEOUT and not t.is_win]

        n = len(entered)
        n_wins = len(wins)

        win_rate = n_wins / n * 100.0 if n > 0 else 0.0

        # ── Return percentages ────────────────────────────────
        win_returns = [t.return_pct for t in wins]
        loss_returns = [t.return_pct for t in losses]
        all_returns = [t.return_pct for t in entered]
        hold_days_list = [t.hold_days for t in entered]

        avg_win_pct = _mean(win_returns)
        avg_loss_pct = _mean(loss_returns)
        avg_hold_days = _mean(hold_days_list)

        # ── Total return ──────────────────────────────────────
        total_net_pnl = sum(t.net_pnl for t in entered)
        total_return_pct = (
            float(total_net_pnl / self._starting_capital * 100)
            if self._starting_capital > 0
            else 0.0
        )

        # ── Profit factor ─────────────────────────────────────
        gross_profit = sum(t.gross_pnl for t in entered if t.gross_pnl > 0)
        gross_loss = abs(sum(t.gross_pnl for t in entered if t.gross_pnl < 0))
        profit_factor = (
            float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit)
        )

        # ── Max drawdown ──────────────────────────────────────
        max_dd, dd_period = self._max_drawdown(entered)

        # ── Sharpe ratio ──────────────────────────────────────
        sharpe = self._sharpe(all_returns, avg_hold_days)

        # ── Benchmark ─────────────────────────────────────────
        nifty_returns = [
            t.nifty_return_pct
            for t in entered
            if t.nifty_return_pct is not None
        ]
        avg_nifty: Optional[float] = _mean(nifty_returns) if nifty_returns else None

        # Alpha = strategy total return vs sum of nifty trade returns
        # (strategy capital deployed period vs nifty same periods)
        strategy_avg = _mean(all_returns)
        alpha: Optional[float] = None
        if avg_nifty is not None:
            alpha = round(strategy_avg - avg_nifty, 4)

        # ── Monthly analysis ──────────────────────────────────
        best_month, worst_month, avg_trades_per_month = self._monthly_stats(entered)

        return PerformanceMetrics(
            total_trades=n,
            wins=n_wins,
            losses=len(losses),
            timeouts=len(timeouts),
            win_rate=round(win_rate, 2),
            avg_win_pct=round(avg_win_pct, 4),
            avg_loss_pct=round(avg_loss_pct, 4),
            avg_hold_days=round(avg_hold_days, 2),
            total_return_pct=round(total_return_pct, 4),
            profit_factor=round(profit_factor, 4),
            max_drawdown_pct=round(max_dd, 4),
            max_drawdown_period=dd_period,
            sharpe_ratio=round(sharpe, 4),
            nifty_return_pct=round(avg_nifty, 4) if avg_nifty is not None else None,
            alpha=round(alpha, 4) if alpha is not None else None,
            best_month=best_month,
            worst_month=worst_month,
            avg_trades_per_month=round(avg_trades_per_month, 1) if avg_trades_per_month else None,
        )

    def verdict(
        self,
        metrics: PerformanceMetrics,
    ) -> tuple[AdvisorVerdict, str]:
        """Determine AdvisorVerdict and explanation note from metrics.

        Returns:
            (verdict, note) — note is a 1-2 sentence plain English explanation.

        Thresholds (from config.py):
          VALID_SIGNAL:      win_rate >= 52% AND profit_factor >= 1.3
          WEAK_SIGNAL:       win_rate 45–52% OR profit_factor 1.0–1.3
          INVALID_SIGNAL:    win_rate < 45% AND profit_factor < 1.0
          INSUFFICIENT_DATA: < MIN_TRADES_FOR_VERDICT (10) entered trades
        """
        n = metrics.total_trades
        wr = metrics.win_rate
        pf = metrics.profit_factor

        if n < MIN_TRADES_FOR_VERDICT:
            return (
                AdvisorVerdict.INSUFFICIENT_DATA,
                f"Only {n} trades — need at least {MIN_TRADES_FOR_VERDICT} "
                f"for a statistically meaningful verdict.",
            )

        valid_wr = wr >= VERDICT_WIN_RATE_VALID
        valid_pf = pf >= VERDICT_PROFIT_FACTOR_VALID
        weak_wr = wr >= VERDICT_WIN_RATE_WEAK
        weak_pf = pf >= VERDICT_PROFIT_FACTOR_WEAK

        if valid_wr and valid_pf:
            return (
                AdvisorVerdict.VALID_SIGNAL,
                f"Win rate {wr:.1f}% (≥{VERDICT_WIN_RATE_VALID}%) and profit "
                f"factor {pf:.2f} (≥{VERDICT_PROFIT_FACTOR_VALID}) — "
                f"signal has edge over {n} trades.",
            )

        if weak_wr or weak_pf:
            reason = []
            if not valid_wr:
                reason.append(f"win rate {wr:.1f}% below {VERDICT_WIN_RATE_VALID}%")
            if not valid_pf:
                reason.append(
                    f"profit factor {pf:.2f} below {VERDICT_PROFIT_FACTOR_VALID}"
                )
            return (
                AdvisorVerdict.WEAK_SIGNAL,
                f"Signal shows some edge but {' and '.join(reason)}. "
                f"Use cautiously — reduce position size.",
            )

        return (
            AdvisorVerdict.INVALID_SIGNAL,
            f"Win rate {wr:.1f}% and profit factor {pf:.2f} are both below "
            f"minimum thresholds. This signal does not have a reliable edge "
            f"over {n} trades.",
        )

    def portfolio_verdict(
        self,
        metrics: PerformanceMetrics,
    ) -> tuple[AdvisorVerdict, str]:
        """Portfolio-level verdict — requires positive alpha in addition to win_rate/PF.

        Used by backtest_engine for PortfolioBacktestResult.
        Returns STRATEGY_VALIDATED when ALL three criteria pass.
        """
        from module_backtest.config import (
            PORTFOLIO_ALPHA_THRESHOLD,
            PORTFOLIO_PROFIT_FACTOR_THRESHOLD,
            PORTFOLIO_WIN_RATE_THRESHOLD,
        )

        n = metrics.total_trades
        wr = metrics.win_rate
        pf = metrics.profit_factor
        alpha = metrics.alpha

        if n < MIN_TRADES_FOR_VERDICT:
            return (
                AdvisorVerdict.INSUFFICIENT_DATA,
                f"Only {n} trades across portfolio — insufficient data.",
            )

        passes_wr = wr >= PORTFOLIO_WIN_RATE_THRESHOLD
        passes_pf = pf >= PORTFOLIO_PROFIT_FACTOR_THRESHOLD
        passes_alpha = alpha is None or alpha >= PORTFOLIO_ALPHA_THRESHOLD

        if passes_wr and passes_pf and passes_alpha:
            alpha_str = f", alpha {alpha:+.1f}%" if alpha is not None else ""
            return (
                AdvisorVerdict.STRATEGY_VALIDATED,
                f"Portfolio-level: {wr:.1f}% win rate, {pf:.2f} profit "
                f"factor{alpha_str} — strategy validated across "
                f"{n} trades.",
            )

        # Fall back to individual verdict logic
        return self.verdict(metrics)

    # ── Internal calculations ───────────────────────────────

    def _max_drawdown(
        self, trades: list[TradeSimulation]
    ) -> tuple[float, Optional[str]]:
        """Compute max drawdown on sequential equity curve.

        Trades are sorted by entry_date. Equity starts at starting_capital
        and each net_pnl is applied in order.

        Returns:
            (max_drawdown_pct, drawdown_period_str)
            drawdown_period_str = 'Mon YYYY' of the trough bar, or None.
        """
        if not trades:
            return 0.0, None

        sorted_trades = sorted(trades, key=lambda t: t.entry_date)

        equity = float(self._starting_capital)
        peak = equity
        max_dd = 0.0
        dd_date: Optional[date] = None

        for trade in sorted_trades:
            equity += float(trade.net_pnl)
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
                    dd_date = trade.exit_date or trade.entry_date

        dd_period: Optional[str] = None
        if dd_date is not None:
            import calendar
            dd_period = f"{calendar.month_abbr[dd_date.month]} {dd_date.year}"

        return max_dd, dd_period

    def _sharpe(self, returns: list[float], avg_hold_days: float) -> float:
        """Compute annualised Sharpe ratio from per-trade return percentages.

        Method:
          1. Convert annual risk-free rate to per-trade rate:
               rf_per_trade = risk_free_annual / 252 × avg_hold_days
          2. Excess returns = trade_return - rf_per_trade
          3. Sharpe_per_trade = mean(excess) / std(excess)
          4. Annualise: × sqrt(252 / avg_hold_days)
              (number of non-overlapping trade periods per year)

        Returns 0.0 for fewer than 2 trades or zero std deviation.
        """
        if len(returns) < 2:
            return 0.0

        hold = max(avg_hold_days, 1.0)
        rf_per_trade = self._risk_free_daily * hold

        excess = [r - rf_per_trade for r in returns]
        mean_excess = _mean(excess)
        std_excess = _std(excess)

        if std_excess == 0.0:
            return 0.0

        sharpe_per_trade = mean_excess / std_excess
        annualisation_factor = math.sqrt(252.0 / hold)
        return sharpe_per_trade * annualisation_factor

    @staticmethod
    def _monthly_stats(
        trades: list[TradeSimulation],
    ) -> tuple[Optional[str], Optional[str], Optional[float]]:
        """Compute best/worst month and average trades per month.

        Groups by (year, month) of entry_date. Returns:
          best_month  — 'Mon YYYY (+X.X%)' highest P&L month
          worst_month — 'Mon YYYY (-X.X%)' lowest P&L month
          avg_trades_per_month — float
        """
        import calendar

        if not trades:
            return None, None, None

        # Group net_pnl by (year, month)
        monthly_pnl: dict[tuple[int, int], float] = defaultdict(float)
        monthly_count: dict[tuple[int, int], int] = defaultdict(int)

        for t in trades:
            key = (t.entry_date.year, t.entry_date.month)
            monthly_pnl[key] += float(t.net_pnl)
            monthly_count[key] += 1

        if not monthly_pnl:
            return None, None, None

        # Find best and worst
        best_key = max(monthly_pnl, key=lambda k: monthly_pnl[k])
        worst_key = min(monthly_pnl, key=lambda k: monthly_pnl[k])

        def _fmt(key: tuple[int, int], pnl: float) -> str:
            mon = calendar.month_abbr[key[1]]
            sign = "+" if pnl >= 0 else ""
            return f"{mon} {key[0]} ({sign}₹{abs(pnl):,.0f})"

        best_month_str = _fmt(best_key, monthly_pnl[best_key])
        worst_month_str = _fmt(worst_key, monthly_pnl[worst_key])

        n_months = len(monthly_pnl)
        avg_trades = sum(monthly_count.values()) / n_months

        return best_month_str, worst_month_str, avg_trades


# ═══════════════════════════════════════════════════════════
# PURE-PYTHON STATISTICS HELPERS
# ═══════════════════════════════════════════════════════════


def _mean(values: list[float]) -> float:
    """Arithmetic mean. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    """Population standard deviation. Returns 0.0 for fewer than 2 values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)
