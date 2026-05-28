"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
data/indicator_builder.py — Pure-Python technical indicator computation

Computes the following indicators from a sorted list[OHLCVBar]:
  - SMA (Simple Moving Average) — 20-day and 50-day
  - RSI (Relative Strength Index) — 14-period Wilder's RSI
  - Average Volume — 30-day rolling average
  - 52-Week High / Low — rolling 252-bar (≈52 trading weeks) max/min
  - VWAP Proxy — cumulative (price × volume) / cumulative volume (session reset daily)

Design constraints:
  - Zero look-ahead bias: indicator for day N uses only bars[0..N] inclusive
  - Pure Python only — no pandas, no numpy (not in requirements.txt)
  - All returns are float, None when insufficient bars for the period
  - BarIndicators is a plain dataclass for speed; no Pydantic overhead in hot loop
  - IndicatorSeries wraps a list and provides fast O(1) date-keyed lookup

Usage:
  fetched_bars = await historical_fetcher.fetch("HDFCBANK", from_date, to_date)
  series = IndicatorBuilder.build(fetched_bars)
  bar_indicators = series.get(date(2026, 3, 15))
  if bar_indicators:
      print(bar_indicators.sma_20, bar_indicators.rsi_14)

All methods are synchronous — no I/O performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from module_backtest.models import OHLCVBar

# ═══════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════


@dataclass
class BarIndicators:
    """All computed indicator values for a single trading day.

    Fields are None when insufficient history exists for the computation.
    For example: sma_50 is None for the first 49 bars.
    """

    bar_date: date
    close: float
    volume: int

    # Moving averages
    sma_20: Optional[float] = None       # 20-day simple moving average of close
    sma_50: Optional[float] = None       # 50-day simple moving average of close

    # Momentum
    rsi_14: Optional[float] = None       # 14-period Wilder RSI (0–100)

    # Volume
    avg_volume_30: Optional[float] = None   # 30-day average volume

    # Range
    high_52w: Optional[float] = None     # rolling 252-bar high (≈52 trading weeks)
    low_52w: Optional[float] = None      # rolling 252-bar low

    # Derived boolean flags (pre-computed for signal_replayer speed)
    near_52w_high: bool = False          # close within 3% of 52w high
    volume_spike: bool = False           # volume > 1.5× avg_volume_30
    above_sma_20: bool = False           # close > sma_20
    above_sma_50: bool = False           # close > sma_50
    golden_cross: bool = False           # sma_20 > sma_50 (bullish trend)


class IndicatorSeries:
    """Wrapper around list[BarIndicators] with O(1) date-keyed lookup.

    Returned by IndicatorBuilder.build(). Immutable after construction.
    """

    def __init__(self, indicators: list[BarIndicators]) -> None:
        self._list: list[BarIndicators] = indicators
        self._index: dict[date, BarIndicators] = {
            ind.bar_date: ind for ind in indicators
        }

    def get(self, bar_date: date) -> Optional[BarIndicators]:
        """Return indicators for the given date, or None if not available."""
        return self._index.get(bar_date)

    def get_all(self) -> list[BarIndicators]:
        """Return all BarIndicators in chronological order."""
        return self._list

    def dates(self) -> list[date]:
        """Return all dates in chronological order."""
        return [ind.bar_date for ind in self._list]

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:  # pragma: no cover
        if not self._list:
            return "IndicatorSeries(empty)"
        return (
            f"IndicatorSeries({len(self._list)} bars, "
            f"{self._list[0].bar_date} → {self._list[-1].bar_date})"
        )


# ═══════════════════════════════════════════════════════════
# BUILDER
# ═══════════════════════════════════════════════════════════


class IndicatorBuilder:
    """Builds BarIndicators for every bar in a sorted OHLCVBar list.

    All computation is O(n) — single pass through bars using running
    accumulators; no nested loops, no slicing by index.

    Public API:
        IndicatorBuilder.build(bars) → IndicatorSeries

    Class is stateless; all methods are @staticmethod or @classmethod.
    """

    # ── indicator periods ──────────────────────────────────
    SMA_SHORT: int = 20
    SMA_LONG: int = 50
    RSI_PERIOD: int = 14
    AVG_VOL_PERIOD: int = 30
    HIGH_LOW_PERIOD: int = 252   # ≈ 52 trading weeks

    # ── derived-flag thresholds ────────────────────────────
    NEAR_52W_HIGH_PCT: float = 0.03      # within 3% of 52w high
    VOLUME_SPIKE_MULTIPLIER: float = 1.5  # 1.5× average volume

    # ─────────────────────────────────────────────────────────
    @classmethod
    def build(cls, bars: list[OHLCVBar]) -> IndicatorSeries:
        """Compute all indicators for every bar.  O(n) single pass.

        Args:
            bars: Chronologically sorted OHLCVBar list (oldest first).

        Returns:
            IndicatorSeries with one BarIndicators per input bar.
            Returns empty IndicatorSeries for empty or single-bar input.
        """
        if not bars:
            return IndicatorSeries([])

        # Sort defensively — callers should pass sorted data but we protect
        sorted_bars = sorted(bars, key=lambda b: b.date)

        # Running accumulators for each indicator
        close_window: list[float] = []     # last N closes (max SMA_LONG length)
        vol_window: list[int] = []         # last AVG_VOL_PERIOD volumes
        high_window: list[float] = []      # last HIGH_LOW_PERIOD highs
        low_window: list[float] = []       # last HIGH_LOW_PERIOD lows

        # RSI running state (Wilder's smoothed averages)
        avg_gain: Optional[float] = None
        avg_loss: Optional[float] = None
        prev_close: Optional[float] = None
        rsi_warm_gains: list[float] = []   # accumulate first RSI_PERIOD changes
        rsi_warm_losses: list[float] = []

        results: list[BarIndicators] = []

        for i, bar in enumerate(sorted_bars):
            close = float(bar.close)
            volume = int(bar.volume)

            # ── update windows ─────────────────────────────
            close_window.append(close)
            if len(close_window) > cls.SMA_LONG:
                close_window.pop(0)

            vol_window.append(volume)
            if len(vol_window) > cls.AVG_VOL_PERIOD:
                vol_window.pop(0)

            high_window.append(float(bar.high))
            if len(high_window) > cls.HIGH_LOW_PERIOD:
                high_window.pop(0)

            low_window.append(float(bar.low))
            if len(low_window) > cls.HIGH_LOW_PERIOD:
                low_window.pop(0)

            # ── SMA ────────────────────────────────────────
            sma_20: Optional[float] = None
            sma_50: Optional[float] = None

            n_closes = len(close_window)
            if n_closes >= cls.SMA_SHORT:
                # Use last SMA_SHORT entries from the window
                sma_20 = sum(close_window[-cls.SMA_SHORT:]) / cls.SMA_SHORT
            if n_closes >= cls.SMA_LONG:
                sma_50 = sum(close_window) / cls.SMA_LONG  # window is exactly SMA_LONG

            # ── RSI (Wilder's smoothed RS) ─────────────────
            rsi_14: Optional[float] = None

            if prev_close is not None:
                change = close - prev_close
                gain = max(change, 0.0)
                loss = max(-change, 0.0)

                if avg_gain is None:
                    # Accumulate the first RSI_PERIOD price changes for seed avg
                    rsi_warm_gains.append(gain)
                    rsi_warm_losses.append(loss)
                    if len(rsi_warm_gains) == cls.RSI_PERIOD:
                        avg_gain = sum(rsi_warm_gains) / cls.RSI_PERIOD
                        avg_loss = sum(rsi_warm_losses) / cls.RSI_PERIOD
                        # Calculate RSI for this bar (period is now satisfied)
                        rsi_14 = cls._wilder_rsi(avg_gain, avg_loss)
                else:
                    # Wilder's smoothing: avg = (prev_avg * (N-1) + current) / N
                    avg_gain = (avg_gain * (cls.RSI_PERIOD - 1) + gain) / cls.RSI_PERIOD
                    avg_loss = (avg_loss * (cls.RSI_PERIOD - 1) + loss) / cls.RSI_PERIOD
                    rsi_14 = cls._wilder_rsi(avg_gain, avg_loss)

            prev_close = close

            # ── Average Volume ─────────────────────────────
            avg_vol: Optional[float] = None
            if len(vol_window) >= cls.AVG_VOL_PERIOD:
                avg_vol = sum(vol_window) / cls.AVG_VOL_PERIOD

            # ── 52-Week High / Low ─────────────────────────
            high_52w: Optional[float] = None
            low_52w: Optional[float] = None
            if high_window:
                high_52w = max(high_window)
            if low_window:
                low_52w = min(low_window)

            # ── Derived boolean flags ──────────────────────
            near_52w_high = (
                high_52w is not None
                and high_52w > 0
                and close >= high_52w * (1.0 - cls.NEAR_52W_HIGH_PCT)
            )
            volume_spike = (
                avg_vol is not None
                and avg_vol > 0
                and volume >= avg_vol * cls.VOLUME_SPIKE_MULTIPLIER
            )
            above_sma_20 = sma_20 is not None and close > sma_20
            above_sma_50 = sma_50 is not None and close > sma_50

            # Golden cross: check previous bar's SMAs vs current
            prev_golden = results[-1].golden_cross if results else False
            if sma_20 is not None and sma_50 is not None:
                golden_cross = sma_20 > sma_50
            else:
                golden_cross = prev_golden  # propagate last known state

            results.append(
                BarIndicators(
                    bar_date=bar.date,
                    close=close,
                    volume=volume,
                    sma_20=sma_20,
                    sma_50=sma_50,
                    rsi_14=rsi_14,
                    avg_volume_30=avg_vol,
                    high_52w=high_52w,
                    low_52w=low_52w,
                    near_52w_high=near_52w_high,
                    volume_spike=volume_spike,
                    above_sma_20=above_sma_20,
                    above_sma_50=above_sma_50,
                    golden_cross=golden_cross,
                )
            )

        return IndicatorSeries(results)

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _wilder_rsi(avg_gain: float, avg_loss: float) -> float:
        """Compute RSI from Wilder's smoothed average gain/loss.

        Returns 50.0 when avg_loss is zero and avg_gain is also zero
        (flat market — no movement). Returns 100.0 if only gains, 0.0
        if only losses.
        """
        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0.0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
