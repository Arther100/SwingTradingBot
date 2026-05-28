"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
engine/trade_simulator.py — Historical trade simulation with realistic costs

Converts a list of ReplayedSignal into list[TradeSimulation] by:
  1. Finding the next trading bar after signal_date (entry bar)
  2. Calculating entry, target, and stop prices
  3. Walking forward bar-by-bar to find the exit condition
  4. Applying TradeCosts (brokerage + STT + slippage) to compute net P&L
  5. Computing nifty_return_pct over the same hold period for alpha

Trade rules (no look-ahead bias):
  - Signal fires at close of day N
  - Entry at open of day N+1 (next available trading bar)
  - Target = entry × (1 + stop_pct × target_rr_ratio) = entry × 1.15 at default settings
  - Stop  = entry × (1 - stop_pct)                    = entry × 0.95 at default settings
  - Each subsequent bar:
      if bar.high >= target_price → EXIT at target (TARGET_HIT)
      if bar.low  <= stop_price  → EXIT at stop   (STOP_HIT)
  - After max_hold_days (10) → EXIT at close           (TIMEOUT)
  - If no entry bar available   → NEVER_ENTERED (signal skipped)

P&L calculation:
  gross_pnl = (exit_price - entry_price) × shares
  total_costs = TradeCosts.calculate(entry_price, exit_price, shares)
              = brokerage + STT (0.1% sell) + slippage_buy (0.1%) + slippage_sell (0.1%)
  net_pnl   = gross_pnl − total_costs
  return_pct = net_pnl / (entry_price × shares) × 100

Position sizing (M3 risk rules):
  risk_amount     = starting_capital × risk_pct_per_trade  (₹1,000 at defaults)
  risk_per_share  = entry_price × stop_pct
  shares_by_risk  = floor(risk_amount / risk_per_share)
  max_position    = starting_capital × MAX_POSITION_PCT    (₹10,000 at defaults)
  shares_by_cap   = floor(max_position / entry_price)
  shares          = min(shares_by_risk, shares_by_cap)     — capped, never zero

One-bar-per-signal rule:
  Each signal produces at most one TradeSimulation. If the SAME bar fires multiple
  signal types, each becomes its own independent simulation (as the backtest
  measures each signal type's performance independently).
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from math import floor
from typing import Optional

from module_backtest.config import (
    ATR_STOP_MULTIPLIER,
    ATR_STOP_PERIOD,
    MAX_HOLD_DAYS,
    MAX_POSITION_PCT,
    RISK_PCT_PER_TRADE,
    STARTING_CAPITAL,
    STOP_PCT,
    TARGET_RR_RATIO,
)
from module_backtest.engine.signal_replayer import ReplayedSignal
from module_backtest.models import (
    BacktestConfig,
    ExitReason,
    OHLCVBar,
    SignalType,
    TradeCosts,
    TradeSimulation,
)

logger = logging.getLogger("swing_advisor.backtest.trade_simulator")


class TradeSimulator:
    """Simulates trades from replayed signals over historical OHLCV bars.

    Usage:
        bars = await historical_fetcher.fetch("HDFCBANK", from_date, to_date)
        nifty_bars = await historical_fetcher.fetch_nifty(from_date, to_date)
        signals = SignalReplayer.replay("HDFCBANK", bars)
        simulator = TradeSimulator()
        trades = simulator.simulate_all(signals, bars, nifty_bars)

    TradeSimulator is stateless between calls. The same instance can be reused
    for different tickers or signal batches.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """Initialise simulator with optional custom BacktestConfig.

        If config is None, uses defaults from module_backtest.config.
        """
        if config is not None:
            self._stop_pct = config.stop_pct
            self._target_rr = config.target_rr_ratio
            self._max_hold = config.max_hold_days
            self._starting_capital = config.starting_capital
            self._risk_pct = config.risk_pct_per_trade
            self._costs = config.costs
        else:
            self._stop_pct = STOP_PCT
            self._target_rr = TARGET_RR_RATIO
            self._max_hold = MAX_HOLD_DAYS
            self._starting_capital = STARTING_CAPITAL
            self._risk_pct = RISK_PCT_PER_TRADE
            self._costs = TradeCosts()

    # ── Public API ──────────────────────────────────────────

    def simulate_all(
        self,
        signals: list[ReplayedSignal],
        bars: list[OHLCVBar],
        nifty_bars: Optional[list[OHLCVBar]] = None,
    ) -> list[TradeSimulation]:
        """Simulate all signals against OHLCV bars.

        Args:
            signals:    Replayed signals (from SignalReplayer.replay).
            bars:       Chronologically sorted OHLCVBar for the same ticker.
            nifty_bars: Optional Nifty 50 bars for nifty_return_pct computation.
                        If None, nifty_return_pct is left as None in TradeSimulation.

        Returns:
            list[TradeSimulation] — one per signal, including NEVER_ENTERED
            signals (these can be filtered out downstream if desired).
            Results are in the same order as the input signals.
        """
        if not signals or not bars:
            return []

        # Build date → bar lookup for O(1) access
        bar_map: dict[date, OHLCVBar] = {b.date: b for b in bars}
        sorted_dates: list[date] = sorted(bar_map.keys())

        nifty_map: dict[date, OHLCVBar] = {}
        if nifty_bars:
            nifty_map = {b.date: b for b in nifty_bars}

        results: list[TradeSimulation] = []
        for signal in signals:
            trade = self._simulate_one(signal, bar_map, sorted_dates, nifty_map)
            results.append(trade)

        entered = sum(1 for t in results if t.exit_reason != ExitReason.NEVER_ENTERED)
        logger.debug(
            f"[{signals[0].ticker if signals else '?'}] "
            f"simulate_all: {len(signals)} signals → "
            f"{entered} trades entered, "
            f"{len(signals) - entered} never-entered"
        )
        return results

    def simulate_signal_type(
        self,
        signals: list[ReplayedSignal],
        bars: list[OHLCVBar],
        signal_type: SignalType,
        nifty_bars: Optional[list[OHLCVBar]] = None,
    ) -> list[TradeSimulation]:
        """Simulate only signals of a specific type.

        Convenience wrapper for per-signal-type backtest runs.
        """
        filtered = [s for s in signals if s.signal_type == signal_type]
        return self.simulate_all(filtered, bars, nifty_bars=nifty_bars)

    # ── Core simulation logic ───────────────────────────────

    def _simulate_one(
        self,
        signal: ReplayedSignal,
        bar_map: dict[date, OHLCVBar],
        sorted_dates: list[date],
        nifty_map: dict[date, OHLCVBar],
    ) -> TradeSimulation:
        """Simulate one signal → one TradeSimulation."""

        # ── Step 1: Find entry bar (first available bar after signal_date) ──
        entry_bar = self._next_bar(signal.signal_date, sorted_dates, bar_map)
        if entry_bar is None:
            return self._never_entered(signal)

        # ── Step 2: Calculate entry price ──
        entry_price = Decimal(str(round(entry_bar.open, 4)))

        # ── Step 3: Position sizing (M3 rules) ──
        shares = self._position_size(entry_price)
        if shares < 1:
            return self._never_entered(signal)

        # ── Step 4: Per-signal ATR-based target and stop prices ──
        stop_atr_mult, target_atr_mult, sig_max_hold = self._get_signal_params(signal.signal_type)
        atr = self._calculate_atr(sorted_dates, bar_map, entry_bar.date)
        if atr is not None and atr > 0:
            raw_stop = entry_price - atr * Decimal(str(stop_atr_mult))
            # Clamp stop: never < 1% or > 15% below entry
            min_stop = entry_price * Decimal("0.85")
            max_stop = entry_price * Decimal("0.99")
            stop_price = max(min_stop, min(raw_stop, max_stop)).quantize(Decimal("0.01"))
            # Target: entry + target_atr_mult × ATR (direct, not via fixed RR ratio)
            target_price = (entry_price + atr * Decimal(str(target_atr_mult))).quantize(
                Decimal("0.01")
            )
        else:
            # Fallback to fixed stop when ATR not available (< 14 bars warm-up)
            stop_pct_d = Decimal(str(self._stop_pct))
            target_rr_d = Decimal(str(self._target_rr))
            target_price = (entry_price * (1 + stop_pct_d * target_rr_d)).quantize(
                Decimal("0.01")
            )
            stop_price = (entry_price * (1 - stop_pct_d)).quantize(Decimal("0.01"))

        # Round 4: Partial exit level for UNUSUAL_ACTIVITY (exit 50% at 2.5×ATR)
        partial_exit_level: Optional[Decimal] = None
        if signal.signal_type == SignalType.UNUSUAL_ACTIVITY and atr is not None and atr > 0:
            partial_exit_level = (entry_price + atr * Decimal("2.5")).quantize(Decimal("0.01"))

        # ── Step 5: Walk forward to find exit ──
        exit_date, exit_price, exit_reason, hold_days, partial_hit = self._walk_forward(
            entry_bar.date, entry_price, target_price, stop_price,
            sorted_dates, bar_map, max_hold=sig_max_hold,
            partial_exit_price=partial_exit_level,
        )

        # ── Step 6: P&L computation ──
        if partial_hit is not None:
            # Blended P&L: 50% exited at partial_hit, 50% at final exit_price
            partial_shares = shares // 2
            remaining_shares = shares - partial_shares
            gross_pnl = (
                (partial_hit - entry_price) * partial_shares
                + (exit_price - entry_price) * remaining_shares
            )
        else:
            gross_pnl = (exit_price - entry_price) * shares
        total_costs = self._costs.calculate(entry_price, exit_price, shares)
        net_pnl = gross_pnl - total_costs

        entry_position_value = entry_price * shares
        return_pct = (
            float(net_pnl / entry_position_value * 100)
            if entry_position_value > 0
            else 0.0
        )

        is_win = net_pnl > Decimal("0")

        # ── Step 7: Nifty benchmark return ──
        nifty_ret = self._nifty_return(
            entry_bar.date, exit_date, nifty_map
        )

        return TradeSimulation(
            ticker=signal.ticker,
            signal_type=signal.signal_type,
            signal_date=signal.signal_date,
            entry_date=entry_bar.date,
            entry_price=entry_price,
            shares=shares,
            target_price=target_price,
            stop_price=stop_price,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            hold_days=hold_days,
            gross_pnl=gross_pnl.quantize(Decimal("0.01")),
            total_costs=total_costs,
            net_pnl=net_pnl.quantize(Decimal("0.01")),
            return_pct=round(return_pct, 4),
            is_win=is_win,
            nifty_return_pct=nifty_ret,
        )

    def _walk_forward(
        self,
        entry_date: date,
        entry_price: Decimal,
        target_price: Decimal,
        stop_price: Decimal,
        sorted_dates: list[date],
        bar_map: dict[date, OHLCVBar],
        max_hold: Optional[int] = None,
        partial_exit_price: Optional[Decimal] = None,
    ) -> tuple[date, Decimal, ExitReason, int, Optional[Decimal]]:
        """Walk bar-by-bar from entry_date+1 to find exit condition.

        Returns: (exit_date, exit_price, exit_reason, hold_days, partial_exit_hit)

        Args:
            max_hold:           Override for max hold days. Falls back to self._max_hold if None.
            partial_exit_price: If set, exit 50% of position at this price (raises stop to BE).
                                Used for UNUSUAL_ACTIVITY partial profit-lock at 2.5×ATR.

        Priority when target and stop both touched in same bar:
          Target wins (optimistic but standard for EOD bar simulation).
          In reality this depends on intraday order; EOD data can't tell us.
        """
        effective_max_hold = max_hold if max_hold is not None else self._max_hold
        # Find starting index (entry bar) in sorted_dates
        try:
            entry_idx = sorted_dates.index(entry_date)
        except ValueError:
            return entry_date, entry_price, ExitReason.TIMEOUT, 0, None

        hold_days = 0
        last_bar = bar_map[entry_date]
        last_bar_price = Decimal(str(round(last_bar.close, 4)))
        highest_price = entry_price  # track peak for momentum exit
        partial_exit_hit: Optional[Decimal] = None  # Round 4: partial exit tracking

        # Walk from the bar AFTER entry_date
        for i in range(entry_idx + 1, len(sorted_dates)):
            bar_date = sorted_dates[i]
            bar = bar_map[bar_date]
            hold_days = i - entry_idx

            bar_high = Decimal(str(round(bar.high, 4)))
            bar_low = Decimal(str(round(bar.low, 4)))
            bar_close = Decimal(str(round(bar.close, 4)))

            # Update peak price seen during the trade
            if bar_high > highest_price:
                highest_price = bar_high

            # Round 4: Partial exit — lock 50% profit, raise stop to breakeven
            if (
                partial_exit_price is not None
                and partial_exit_hit is None
                and bar_high >= partial_exit_price
            ):
                partial_exit_hit = partial_exit_price
                stop_price = entry_price  # move stop to breakeven for remaining 50%

            # Check target first (target wins over stop in same bar)
            if bar_high >= target_price:
                return bar_date, target_price, ExitReason.TARGET_HIT, hold_days, partial_exit_hit

            # Check stop
            if bar_low <= stop_price:
                return bar_date, stop_price, ExitReason.STOP_HIT, hold_days, partial_exit_hit

            # Momentum exit: give back 50% of peak unrealised gain
            if self._check_momentum_exit(entry_price, bar_close, highest_price):
                logger.debug(
                    f"Momentum exit: entry={entry_price} peak={highest_price} "
                    f"close={bar_close} hold={hold_days}d"
                )
                return bar_date, bar_close, ExitReason.TIMEOUT, hold_days, partial_exit_hit

            # Timeout check
            if hold_days >= effective_max_hold:
                return bar_date, bar_close, ExitReason.TIMEOUT, hold_days, partial_exit_hit

            last_bar_price = bar_close

        # Ran out of bars before reaching max_hold_days or hitting target/stop
        last_date = sorted_dates[min(entry_idx + hold_days, len(sorted_dates) - 1)]
        return last_date, last_bar_price, ExitReason.TIMEOUT, hold_days, partial_exit_hit

    # ── Per-signal exit parameters ────────────────────────────────────

    @staticmethod
    def _get_signal_params(signal_type: SignalType) -> tuple[float, float, int]:
        """Return (stop_atr_mult, target_atr_mult, max_hold_days) per signal type.

        Different signals resolve on different timescales:
          breakout_watch   — patient hold, wider target (6×ATR stop, 6×ATR target)
          fii_buying       — medium hold (2×ATR stop, 5×ATR target)
          unusual_activity — fast mover, tight exits (1.5×ATR stop, 3.5×ATR target)
          accumulation     — medium hold matching fii_buying

        ATR formula:
          stop  = entry − stop_atr_mult  × ATR
          target = entry + target_atr_mult × ATR  (direct, not via RR ratio)
        """
        _params: dict[SignalType, tuple[float, float, int]] = {
            SignalType.BREAKOUT_WATCH:    (2.0, 6.0, 7),
            SignalType.FII_BUYING:        (2.0, 5.0, 6),
            SignalType.UNUSUAL_ACTIVITY:  (1.5, 4.5, 5),  # Round 4: target 3.5→4.5
            SignalType.ACCUMULATION_ZONE: (2.0, 5.0, 7),
        }
        return _params.get(signal_type, (2.0, 5.0, 7))

    # ── ATR calculation ─────────────────────────────────────

    @staticmethod
    def _calculate_atr(
        sorted_dates: list[date],
        bar_map: dict[date, OHLCVBar],
        entry_date: date,
        period: int = ATR_STOP_PERIOD,
    ) -> Optional[Decimal]:
        """Compute ATR over `period` bars immediately before entry_date.

        Uses Wilder-compatible simple average of True Ranges (not EMA).
        True Range = max(H−L, |H−prev_close|, |L−prev_close|).

        Returns None if fewer than `period` bars are available before entry.
        """
        try:
            entry_idx = sorted_dates.index(entry_date)
        except ValueError:
            return None

        if entry_idx < period:
            return None  # not enough history for ATR warm-up

        true_ranges: list[Decimal] = []
        for i in range(entry_idx - period + 1, entry_idx + 1):
            if i <= 0:
                continue
            bar = bar_map[sorted_dates[i]]
            prev_bar = bar_map[sorted_dates[i - 1]]
            high = Decimal(str(round(float(bar.high), 4)))
            low = Decimal(str(round(float(bar.low), 4)))
            prev_close = Decimal(str(round(float(prev_bar.close), 4)))
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        if not true_ranges:
            return None

        return sum(true_ranges, Decimal("0")) / len(true_ranges)

    # ── Momentum exit ───────────────────────────────────────

    @staticmethod
    def _check_momentum_exit(
        entry_price: Decimal,
        current_price: Decimal,
        highest_price: Decimal,
    ) -> bool:
        """Exit early if trade gives back 50% or more of its peak unrealised gain.

        Protects profits on trades that ran well but are now reversing.
        Only triggers when the trade was profitable at some point
        (highest_price > entry_price).

        Formula:
          peak_gain    = highest_price − entry_price
          current_gain = current_price − entry_price
          giveback_pct = (peak_gain − current_gain) / peak_gain
          exit if giveback_pct >= 0.5  (gave back half the gain)
        """
        if highest_price <= entry_price:
            return False  # trade never profitable — no profit to protect
        peak_gain = highest_price - entry_price
        # Minimum peak gain threshold: only protect gains ≥ 1.5% of entry.
        # Prevents this from acting as a hair-trigger breakeven stop on noise.
        min_meaningful_gain = entry_price * Decimal("0.015")
        if peak_gain < min_meaningful_gain:
            return False  # peak was too small — let normal stop/target handle it
        current_gain = current_price - entry_price
        if current_gain <= Decimal("0"):
            # Trade has fallen below entry after a meaningful peak → exit
            return True
        giveback_pct = (peak_gain - current_gain) / peak_gain
        return giveback_pct >= Decimal("0.5")

    # ── Position sizing ─────────────────────────────────────

    def _position_size(self, entry_price: Decimal) -> int:
        """Compute shares using M3 risk rules.

        risk_amount    = starting_capital × risk_pct_per_trade
        risk_per_share = entry_price × stop_pct
        shares_by_risk = floor(risk_amount / risk_per_share)
        max_position   = starting_capital × MAX_POSITION_PCT
        shares         = min(by_risk, by_cap)
        """
        if entry_price <= 0:
            return 0

        risk_amount = self._starting_capital * self._risk_pct
        stop_pct_d = Decimal(str(self._stop_pct))
        risk_per_share = entry_price * stop_pct_d

        if risk_per_share <= 0:
            return 0

        shares_by_risk = int(floor(float(risk_amount / risk_per_share)))

        max_position = self._starting_capital * Decimal(str(MAX_POSITION_PCT))
        shares_by_cap = int(floor(float(max_position / entry_price)))

        return max(1, min(shares_by_risk, shares_by_cap))

    # ── Nifty benchmark ─────────────────────────────────────

    @staticmethod
    def _nifty_return(
        entry_date: date,
        exit_date: date,
        nifty_map: dict[date, OHLCVBar],
    ) -> Optional[float]:
        """Compute Nifty 50 return over the same hold period.

        Uses close prices. Returns None if either date is missing.
        """
        if not nifty_map:
            return None
        entry_bar = nifty_map.get(entry_date)
        exit_bar = nifty_map.get(exit_date)
        if entry_bar is None or exit_bar is None:
            return None
        if entry_bar.close <= 0:
            return None
        return round(
            (exit_bar.close - entry_bar.close) / entry_bar.close * 100, 4
        )

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _next_bar(
        signal_date: date,
        sorted_dates: list[date],
        bar_map: dict[date, OHLCVBar],
    ) -> Optional[OHLCVBar]:
        """Return the first trading bar strictly after signal_date."""
        for d in sorted_dates:
            if d > signal_date:
                return bar_map[d]
        return None

    @staticmethod
    def _never_entered(signal: ReplayedSignal) -> TradeSimulation:
        """Build a placeholder TradeSimulation for a signal that could not be entered."""
        placeholder_price = Decimal(str(round(signal.close, 4)))
        return TradeSimulation(
            ticker=signal.ticker,
            signal_type=signal.signal_type,
            signal_date=signal.signal_date,
            entry_date=signal.signal_date,  # same day — marker only
            entry_price=placeholder_price,
            shares=0,
            target_price=placeholder_price,
            stop_price=placeholder_price,
            exit_date=signal.signal_date,
            exit_price=placeholder_price,
            exit_reason=ExitReason.NEVER_ENTERED,
            hold_days=0,
        )
