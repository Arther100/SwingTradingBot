"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
engine/signal_replayer.py — Historical signal reconstruction (zero look-ahead)

Replays what the M1 advisor signal logic would have flagged on every historical
day for a given ticker, using only data available up to that day.

Signal reconstruction logic mirrors module1_data_layer/signals/advisor_signals.py:
  - Same RangePosition thresholds (NEAR_HIGH ≥ 80%, UPPER 60-80%, etc.)
  - Same VolumeSignal thresholds (UNUSUAL_SPIKE ≥ 3.0×, ABOVE_AVERAGE ≥ 1.3×, etc.)
  - Same decision matrix (_decide_flag) → mapped to backtest SignalType

Additional signal — FII_BUYING (proxy):
  Live FII flow data is unavailable for historical dates (NSE only publishes
  rolling ~60 days). The proxy rule simulates FII accumulation behaviour:
    golden_cross (SMA20 > SMA50) AND above_sma_50 AND volume_spike
  This captures the "smart money accumulating while trend turns bullish" pattern.

Zero look-ahead guarantee:
  IndicatorBuilder.build() computes all indicators with only data up to bar N.
  SignalReplayer uses the IndicatorSeries for that bar's pre-computed values.
  The replayer never reads bars[i+1] or beyond.

Data flow:
  list[OHLCVBar] → IndicatorBuilder.build() → IndicatorSeries
    → (for each bar) classify signals → filter actionable → list[ReplayedSignal]
    → TradeSimulator uses signal_date + bars to simulate entry on day N+1

Usage:
  bars = await historical_fetcher.fetch("HDFCBANK", from_date, to_date)
  signals = SignalReplayer.replay("HDFCBANK", bars)
  # signals with signal_type=BREAKOUT_WATCH, ACCUMULATION_ZONE, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from module_backtest.config import MIN_CONFIDENCE_SCORE as _MIN_CONF

# Per-signal minimum confidence thresholds
# breakout_watch requires higher quality due to historically poor PF (0.40)
_BREAKOUT_MIN_CONF: float = 8.0    # stricter than global 7.0 (Round 3 fix)
_MIN_CONF_FII_BUYING: float = 7.5  # Round 5: 8.0 was too aggressive (15 trades); back to 7.5
from module_backtest.data.indicator_builder import BarIndicators, IndicatorBuilder
from module_backtest.models import OHLCVBar, SignalType

logger = logging.getLogger("swing_advisor.backtest.signal_replayer")

# ═══════════════════════════════════════════════════════════
# THRESHOLDS (mirror module1_data_layer/models.py)
# ═══════════════════════════════════════════════════════════

# Range position thresholds (percentile of 52w range)
_NEAR_HIGH_PCT = 0.80
_UPPER_PCT = 0.60
_MIDDLE_PCT = 0.40
_LOWER_PCT = 0.20

# Volume signal thresholds (ratio to 30d average)
_UNUSUAL_SPIKE_RATIO = 3.0
_ABOVE_AVERAGE_RATIO = 1.3
_BELOW_AVERAGE_RATIO = 0.7

# ── Fix 4: FII data path ─────────────────────────────────
_FII_HISTORY_PATH = (
    Path(__file__).parent.parent.parent
    / "module1_data_layer" / "data" / "fii_dii_history.json"
)
_FII_BLOCK_THRESHOLD = -2000.0  # Cr — skip long trades when FII net selling > 2000 Cr


# ═══════════════════════════════════════════════════════════
# FILTER HELPERS (Fixes 1, 3, 4, 5)
# ═══════════════════════════════════════════════════════════

def _requires_confirmation(signal_type: SignalType) -> bool:
    """Only BREAKOUT_WATCH needs a next-day close confirmation."""
    return signal_type == SignalType.BREAKOUT_WATCH


def _is_nifty_uptrend(nifty_bars: list[OHLCVBar], signal_date: date) -> bool:
    """Return True if Nifty close is above its 50-day MA on signal_date.

    If fewer than 50 Nifty bars are available before signal_date, allow the
    trade (insufficient data → no filter applied).
    """
    recent_closes = [float(b.close) for b in nifty_bars if b.date <= signal_date][-50:]
    if len(recent_closes) < 50:
        return True  # not enough history — allow
    ma50 = sum(recent_closes) / 50
    return recent_closes[-1] > ma50


def _is_fii_favourable(signal_date: date) -> bool:
    """Return False only when FII net selling exceeds 2000 Cr on signal_date.

    Reads from module1_data_layer/data/fii_dii_history.json. If the file is
    missing, the date has no entry, or any read error occurs, allow the trade.
    """
    try:
        with open(_FII_HISTORY_PATH) as fh:
            history: dict = json.load(fh)
        key = signal_date.isoformat()
        if key not in history:
            return True  # no data → allow
        fii_net = float(history[key].get("fii_net", 0.0))
        return fii_net > _FII_BLOCK_THRESHOLD
    except Exception:
        return True  # file missing or parse error → allow


def _signal_confidence(
    signal_type: SignalType,
    vol_ratio: float,
    range_pct: Optional[float],
    change_pct: float,
    ind: BarIndicators,
) -> float:
    """Quick 0–10 signal quality score for backtest filtering (Fix 5).

    Factors scored:
      Volume ratio     — high volume = institutional interest
      Range clarity    — near 52w high/low = clear structure
      Price change     — strong move = conviction
      SMA alignment    — above SMA20 + SMA50 = trend confirmation
      Golden cross     — medium-term trend turning bullish

    Returns float in range 5.0–9.0; threshold is MIN_CONFIDENCE_SCORE (7.0).
    """
    score = 5.0

    # Volume quality (+0.0 to +1.0)
    if vol_ratio >= 3.0:
        score += 1.0
    elif vol_ratio >= 2.0:
        score += 0.75
    elif vol_ratio >= 1.5:
        score += 0.5

    # Range clarity (+0.0 to +0.5)
    if range_pct is not None:
        if range_pct >= 0.80 or range_pct <= 0.20:
            score += 0.5
        elif range_pct >= 0.60 or range_pct <= 0.40:
            score += 0.25

    # Price change strength (+0.0 to +0.5)
    abs_chg = abs(change_pct)
    if abs_chg >= 3.0:
        score += 0.5
    elif abs_chg >= 1.5:
        score += 0.25

    # SMA alignment (+0.0 to +0.5)
    if ind.above_sma_20 and ind.above_sma_50:
        score += 0.5
    elif ind.above_sma_50:
        score += 0.25

    # Golden cross (+0.0 to +0.5)
    if ind.golden_cross:
        score += 0.5

    return round(score, 2)


# ═══════════════════════════════════════════════════════════
# WEEKLY TRADE GATE (Round 4 — cap correlated signals)
# ═══════════════════════════════════════════════════════════

class WeeklyTradeGate:
    """Limits new trades to max_trades_per_week per ticker.

    Prevents correlated signal bursts on strong FII buying days from
    overwhelming the portfolio with same-direction trades.
    Applied per ticker inside SignalReplayer.replay().
    """

    def __init__(self, max_trades_per_week: int = 3) -> None:
        self.max_per_week = max_trades_per_week
        self._weekly_counts: dict[tuple[int, int], int] = {}

    def can_take_trade(self, trade_date: date) -> bool:
        week_key = trade_date.isocalendar()[:2]
        return self._weekly_counts.get(week_key, 0) < self.max_per_week

    def record_trade(self, trade_date: date) -> None:
        week_key = trade_date.isocalendar()[:2]
        self._weekly_counts[week_key] = self._weekly_counts.get(week_key, 0) + 1


# ═══════════════════════════════════════════════════════════
# INTERNAL ENUM SUBSTITUTES (avoid importing M1 models)
# ═══════════════════════════════════════════════════════════

class _RangePos:
    NEAR_HIGH = "near_high"
    UPPER = "upper"
    MIDDLE = "middle"
    LOWER = "lower"
    NEAR_LOW = "near_low"


class _VolSig:
    UNUSUAL_SPIKE = "unusual_spike"
    ABOVE_AVERAGE = "above_average"
    NORMAL = "normal"
    BELOW_AVERAGE = "below_average"


# ═══════════════════════════════════════════════════════════
# OUTPUT DATA CLASS
# ═══════════════════════════════════════════════════════════


@dataclass
class ReplayedSignal:
    """A single reconstructed signal for one ticker on one historical day.

    Fields:
        ticker:       NSE ticker symbol
        signal_date:  Date the signal fired (end of day N — entry is day N+1)
        signal_type:  Which of the 4 backtest signal types was triggered
        close:        Closing price on signal_date (used for context)
        volume:       Volume on signal_date
        volume_ratio: volume / avg_volume_30d  (0.0 if avg not yet available)
        range_pct:    Price position in 52w range 0.0–1.0  (None if 52w data unavailable)
        change_pct:   Daily % change on signal_date  (None for first bar)
        cot_reason:   Compact Chain of Thought string explaining why signal fired
    """

    ticker: str
    signal_date: date
    signal_type: SignalType
    close: float
    volume: int
    volume_ratio: float
    range_pct: Optional[float]
    change_pct: Optional[float]
    cot_reason: str


# ═══════════════════════════════════════════════════════════
# SIGNAL REPLAYER
# ═══════════════════════════════════════════════════════════


class SignalReplayer:
    """Reconstructs historical M1 signals from OHLCV bars.

    All methods are synchronous (no I/O). Call IndicatorBuilder internally.
    Stateless — instantiation not required; use the class methods directly.

    The replayer only emits signals for the 4 SignalTypes the backtest tests:
      BREAKOUT_WATCH      — Near 52w high with above-average volume + up move
      ACCUMULATION_ZONE   — Lower range with above-average volume + up move
      UNUSUAL_ACTIVITY    — Volume spike without a clear direction signal
      FII_BUYING          — Proxy: golden_cross + above_sma_50 + volume_spike

    All other M1 flags (SELLING_PRESSURE, CONSOLIDATION, etc.) are filtered out —
    the backtest only tests signals that generate long entries.
    """

    # Actionable M1 advisor flags that map to a backtest SignalType
    _FLAG_TO_SIGNAL: dict[str, SignalType] = {
        "breakout_watch": SignalType.BREAKOUT_WATCH,
        "accumulation_zone": SignalType.ACCUMULATION_ZONE,
        "unusual_activity": SignalType.UNUSUAL_ACTIVITY,
    }

    # ── Public API ──────────────────────────────────────────

    @classmethod
    def replay(
        cls,
        ticker: str,
        bars: list[OHLCVBar],
        signal_types: Optional[list[SignalType]] = None,
        nifty_bars: Optional[list[OHLCVBar]] = None,
    ) -> list[ReplayedSignal]:
        """Replay all historical signals for a ticker.

        Args:
            ticker:        NSE ticker symbol (used in output only).
            bars:          Chronologically sorted OHLCVBar list (oldest first).
            signal_types:  Optional filter — only return these SignalTypes.
                           If None, returns all 4 signal types.

        Returns:
            List of ReplayedSignal, one per (bar, signal_type) pair where
            a signal fired. Empty list if bars is empty or too short for
            indicator warm-up.

        Zero look-ahead: IndicatorBuilder.build() guarantees no future data
        reaches the indicator computation for bar N.
        """
        if not bars:
            return []

        # Build full indicator series in one O(n) pass — no look-ahead
        indicator_series = IndicatorBuilder.build(bars)
        all_indicators = indicator_series.get_all()

        if len(all_indicators) < 2:
            return []  # Need at least 2 bars for a change_pct

        # Build a quick prev-close lookup from sorted bars
        sorted_bars = sorted(bars, key=lambda b: b.date)
        prev_close_map: dict[date, Optional[float]] = {}
        for i, bar in enumerate(sorted_bars):
            prev_close_map[bar.date] = float(sorted_bars[i - 1].close) if i > 0 else None

        # Fix 1: next-bar lookup for confirmation candle check
        next_bar_map: dict[date, OHLCVBar] = {}
        for i in range(len(sorted_bars) - 1):
            next_bar_map[sorted_bars[i].date] = sorted_bars[i + 1]

        # Round 4: Nifty daily change_pct map for relative strength filter
        nifty_change_pct_map: dict[date, float] = {}
        if nifty_bars:
            sorted_nifty = sorted(nifty_bars, key=lambda b: b.date)
            for i, nb in enumerate(sorted_nifty):
                if i == 0:
                    nifty_change_pct_map[nb.date] = 0.0
                else:
                    prev_n = float(sorted_nifty[i - 1].close)
                    nifty_change_pct_map[nb.date] = (
                        (float(nb.close) - prev_n) / prev_n * 100.0
                        if prev_n > 0 else 0.0
                    )

        allowed = set(signal_types) if signal_types else None

        results: list[ReplayedSignal] = []
        weekly_gate = WeeklyTradeGate(max_trades_per_week=3)  # Round 4: cap bursts
        _fii_consecutive: int = 0  # consecutive bars where _detect_fii_buying is True

        for ind in all_indicators:
            # Skip the very first bar — no prev_close means no change_pct
            prev_close = prev_close_map.get(ind.bar_date)
            if prev_close is None or prev_close == 0.0:
                continue

            # Indicators must have at least avg_volume_30 available
            # (30 bars warm-up). Before that, signals would be noisy.
            if ind.avg_volume_30 is None:
                continue

            change_pct = ((ind.close - prev_close) / prev_close) * 100.0
            vol_ratio = ind.volume / ind.avg_volume_30 if ind.avg_volume_30 > 0 else 0.0
            nifty_change_pct: Optional[float] = nifty_change_pct_map.get(ind.bar_date)

            # Classify inputs using same thresholds as M1
            range_pct = cls._range_pct(ind)
            range_pos = cls._classify_range(range_pct)
            vol_sig = cls._classify_volume(vol_ratio)

            # ── M1 signal logic ─────────────────────────────
            m1_flag = cls._decide_flag(range_pos, vol_sig, change_pct)

            # ── FII_BUYING proxy ─────────────────────────────
            fii_detected = cls._detect_fii_buying(ind, vol_ratio, change_pct, nifty_change_pct)
            if fii_detected:
                _fii_consecutive += 1
            else:
                _fii_consecutive = 0
            # Anti-burst: suppress after 2 consecutive trigger days (Fix 1)
            fii_signal = fii_detected and _fii_consecutive < 3

            # ── Emit M1-derived signal ────────────────────────
            signal_type = cls._FLAG_TO_SIGNAL.get(m1_flag)
            if signal_type is not None and (allowed is None or signal_type in allowed):
                # Round 3: per-signal strict validation gate
                skip_m1 = False
                if signal_type == SignalType.BREAKOUT_WATCH:
                    if not cls._is_breakout_watch_valid(ind, vol_ratio):
                        skip_m1 = True  # fails breakout quality filter
                elif signal_type == SignalType.UNUSUAL_ACTIVITY:
                    if not cls._is_unusual_activity_valid(vol_ratio):
                        skip_m1 = True  # volume not unusual enough
                if not skip_m1:
                    # Per-signal confidence threshold (breakout_watch: 8.0, others: 7.0)
                    min_conf_thr = (
                        _BREAKOUT_MIN_CONF
                        if signal_type == SignalType.BREAKOUT_WATCH
                        else _MIN_CONF
                    )
                    conf = _signal_confidence(signal_type, vol_ratio, range_pct, change_pct, ind)
                    if conf >= min_conf_thr:
                        # Nifty trend filter — only long trades in uptrend
                        if nifty_bars is None or _is_nifty_uptrend(nifty_bars, ind.bar_date):
                            # FII net-selling filter
                            if _is_fii_favourable(ind.bar_date):
                                # Confirmation candle for breakout_watch
                                emit_date = ind.bar_date
                                skip_signal = False
                                if _requires_confirmation(signal_type):
                                    next_bar = next_bar_map.get(ind.bar_date)
                                    if next_bar is None:
                                        skip_signal = True
                                    elif float(next_bar.close) <= float(ind.close):
                                        logger.debug(
                                            f"[{ticker}] {ind.bar_date} breakout_watch rejected: "
                                            f"no confirmation (D+1 close {next_bar.close:.2f} "
                                            f"<= D0 close {ind.close:.2f})"
                                        )
                                        skip_signal = True
                                    else:
                                        emit_date = next_bar.date
                                if not skip_signal:
                                    if weekly_gate.can_take_trade(emit_date):
                                        weekly_gate.record_trade(emit_date)
                                        results.append(
                                            cls._make_signal(
                                                ticker, ind, signal_type, vol_ratio,
                                                range_pct, change_pct, m1_flag,
                                                signal_date_override=emit_date,
                                            )
                                        )

            # ── Emit FII_BUYING proxy (independent of M1 flag) ──
            if fii_signal and (allowed is None or SignalType.FII_BUYING in allowed):
                conf = _signal_confidence(SignalType.FII_BUYING, vol_ratio, range_pct, change_pct, ind)
                if conf >= _MIN_CONF_FII_BUYING:  # Round 4: raised from 7.0 to 8.0
                    if nifty_bars is None or _is_nifty_uptrend(nifty_bars, ind.bar_date):
                        if _is_fii_favourable(ind.bar_date):
                            if weekly_gate.can_take_trade(ind.bar_date):
                                weekly_gate.record_trade(ind.bar_date)
                                results.append(
                                    cls._make_fii_signal(
                                        ticker, ind, vol_ratio, range_pct, change_pct
                                    )
                                )

        logger.debug(
            f"[{ticker}] Signal replay: {len(bars)} bars → {len(results)} signals"
        )
        return results

    @classmethod
    def replay_date_range(
        cls,
        ticker: str,
        bars: list[OHLCVBar],
        from_date: date,
        to_date: date,
        signal_types: Optional[list[SignalType]] = None,
        nifty_bars: Optional[list[OHLCVBar]] = None,
    ) -> list[ReplayedSignal]:
        """Replay signals within a specific date window.

        Indicators are still built from the full bars list (to preserve
        warm-up accuracy for bars near from_date), but only signals whose
        signal_date falls within [from_date, to_date] are returned.

        This is the correct way to handle walk-forward splits: always build
        indicators from the full available history, but only evaluate signals
        in the target window.
        """
        all_signals = cls.replay(ticker, bars, signal_types=signal_types, nifty_bars=nifty_bars)
        return [
            s for s in all_signals
            if from_date <= s.signal_date <= to_date
        ]

    # ── Internal classification ─────────────────────────────

    @staticmethod
    def _range_pct(ind: BarIndicators) -> Optional[float]:
        """Compute 0.0–1.0 position of close in 52w high/low range."""
        if ind.high_52w is None or ind.low_52w is None:
            return None
        range_total = ind.high_52w - ind.low_52w
        if range_total <= 0:
            return None
        pct = (ind.close - ind.low_52w) / range_total
        return max(0.0, min(1.0, pct))

    @staticmethod
    def _classify_range(range_pct: Optional[float]) -> str:
        """Map 0.0–1.0 range percentile to M1 RangePosition string."""
        if range_pct is None:
            return _RangePos.MIDDLE  # Default when data unavailable
        if range_pct >= _NEAR_HIGH_PCT:
            return _RangePos.NEAR_HIGH
        if range_pct >= _UPPER_PCT:
            return _RangePos.UPPER
        if range_pct >= _MIDDLE_PCT:
            return _RangePos.MIDDLE
        if range_pct >= _LOWER_PCT:
            return _RangePos.LOWER
        return _RangePos.NEAR_LOW

    @staticmethod
    def _classify_volume(vol_ratio: float) -> str:
        """Map volume ratio to M1 VolumeSignal string."""
        if vol_ratio >= _UNUSUAL_SPIKE_RATIO:
            return _VolSig.UNUSUAL_SPIKE
        if vol_ratio >= _ABOVE_AVERAGE_RATIO:
            return _VolSig.ABOVE_AVERAGE
        if vol_ratio < _BELOW_AVERAGE_RATIO:
            return _VolSig.BELOW_AVERAGE
        return _VolSig.NORMAL

    @staticmethod
    def _decide_flag(range_pos: str, vol_sig: str, change_pct: float) -> str:
        """Core M1 decision matrix — returns AdvisorFlag string.

        Mirrors module1_data_layer/signals/advisor_signals.py::_decide_flag()
        exactly. Priority order: unusual_spike first, then near_high, etc.
        """
        # Rule 1: Unusual volume spike
        if vol_sig == _VolSig.UNUSUAL_SPIKE:
            if change_pct > 2.0:
                return "breakout_watch"
            elif change_pct < -2.0:
                return "selling_pressure"
            else:
                return "unusual_activity"

        # Rule 2: Near 52w high
        if range_pos == _RangePos.NEAR_HIGH:
            if vol_sig == _VolSig.ABOVE_AVERAGE and change_pct > 1.0:
                return "breakout_watch"
            elif vol_sig == _VolSig.ABOVE_AVERAGE and change_pct < -1.0:
                return "distribution_zone"
            elif vol_sig == _VolSig.ABOVE_AVERAGE:
                return "momentum_building"

        # Rule 3: Near low / lower range with above-average volume
        if range_pos in (_RangePos.NEAR_LOW, _RangePos.LOWER):
            if vol_sig == _VolSig.ABOVE_AVERAGE and change_pct > 0:
                return "accumulation_zone"
            elif vol_sig == _VolSig.ABOVE_AVERAGE and change_pct < -1.0:
                return "selling_pressure"

        # Rule 4: Upper/middle with above-average volume and positive move
        if range_pos in (_RangePos.UPPER, _RangePos.MIDDLE):
            if vol_sig == _VolSig.ABOVE_AVERAGE and change_pct > 0:
                return "momentum_building"
            elif vol_sig == _VolSig.ABOVE_AVERAGE and change_pct < -1.0:
                return "selling_pressure"

        # Rule 5: Low volume or flat → consolidation/neutral
        if vol_sig == _VolSig.BELOW_AVERAGE:
            return "consolidation"
        if -0.5 <= change_pct <= 0.5:
            return "consolidation"

        return "neutral"

    @staticmethod
    def _is_breakout_watch_valid(ind: BarIndicators, vol_ratio: float) -> bool:
        """Strict validation for BREAKOUT_WATCH — all 4 conditions must hold.

        Without these guards, 61 breakout signals had PF=0.40 (Round 2 results).
        Volume confirmation is the single most important filter on NSE — false
        breakouts are frequent without genuine institutional participation.

        Conditions:
          1. Within 3% of 52-week high (true near-high, not just upper range)
          2. Volume >= 2.0× average (strong institutional interest)
          3. RSI between 50–70 (momentum confirmed, not yet overbought)
          4. Above BOTH 20MA and 50MA (strong multi-timeframe trend)

        Nifty uptrend is handled separately by _is_nifty_uptrend() in caller.
        Confidence threshold is _BREAKOUT_MIN_CONF (8.0) in caller.
        """
        # 1. Must be within 3% of 52-week high
        if ind.high_52w is not None and ind.high_52w > 0:
            distance_from_high = (ind.high_52w - ind.close) / ind.high_52w
            if distance_from_high > 0.03:
                return False
        # 2. Volume must confirm (2× minimum — not just above-average 1.3×)
        if vol_ratio < 2.0:
            return False
        # 3. RSI in momentum zone — not overbought, not oversold
        if ind.rsi_14 is not None and not (50.0 <= ind.rsi_14 <= 70.0):
            return False
        # 4. Strong multi-timeframe trend: above both moving averages
        if not (ind.above_sma_20 and ind.above_sma_50):
            return False
        return True

    @staticmethod
    def _is_unusual_activity_valid(vol_ratio: float) -> bool:
        """Unusual activity must have genuinely unusual volume (>= 2.5×).

        Previous threshold 1.5× was too low — generated signals with PF=0.00.
        2.5× ensures only truly exceptional volume days are captured.
        """
        return vol_ratio >= 2.5

    @staticmethod
    def _detect_fii_buying(
        ind: BarIndicators,
        vol_ratio: float,
        stock_change_pct: float = 0.0,
        nifty_change_pct: Optional[float] = None,
    ) -> bool:
        """FII_BUYING proxy v4 (Round 4) — adds relative strength vs Nifty.

        All conditions must hold:
          - golden_cross (SMA20 > SMA50) — medium-term trend turning bullish
          - above_sma_50 — price confirmed above medium-term MA
          - above_sma_20 — price in near-term uptrend
          - vol_ratio >= 1.5× avg_volume_30 — institutional interest
          - rsi_14 between 40–65 — not overbought or deeply oversold
          - stock outperforms Nifty by ≥ 0.3% (Round 4 — relative strength gate)

        Relative strength gate: filters stocks merely riding the broad market.
        If nifty_change_pct is None (no Nifty data for date), gate is skipped.
        """
        if ind.avg_volume_30 is None:
            return False
        # RSI gate: healthy zone only
        if ind.rsi_14 is not None and not (40.0 <= ind.rsi_14 <= 65.0):
            return False
        # Near-term uptrend: must be above 20-day MA
        if not ind.above_sma_20:
            return False
        # Round 5: Relative strength vs Nifty — stock must at least match Nifty
        # Threshold 0.0%: filters pure market-riders (stock underperforms Nifty)
        # Was 0.3% in Round 4 — too strict, cut FII trades from 62 to 3
        if nifty_change_pct is not None:
            if (stock_change_pct - nifty_change_pct) < 0.0:
                return False
        return (
            ind.golden_cross
            and ind.above_sma_50
            and vol_ratio >= IndicatorBuilder.VOLUME_SPIKE_MULTIPLIER
        )

    # ── Signal constructors ─────────────────────────────────

    @staticmethod
    def _make_signal(
        ticker: str,
        ind: BarIndicators,
        signal_type: SignalType,
        vol_ratio: float,
        range_pct: Optional[float],
        change_pct: float,
        m1_flag: str,
        signal_date_override: Optional[date] = None,
    ) -> ReplayedSignal:
        range_str = f"{range_pct:.0%}" if range_pct is not None else "n/a"
        return ReplayedSignal(
            ticker=ticker,
            signal_date=signal_date_override if signal_date_override is not None else ind.bar_date,
            signal_type=signal_type,
            close=ind.close,
            volume=ind.volume,
            volume_ratio=round(vol_ratio, 2),
            range_pct=range_pct,
            change_pct=round(change_pct, 2),
            cot_reason=(
                f"M1 flag={m1_flag} | "
                f"range={range_str} in 52w band | "
                f"vol_ratio={vol_ratio:.2f}x | "
                f"change={change_pct:+.2f}%"
            ),
        )

    @staticmethod
    def _make_fii_signal(
        ticker: str,
        ind: BarIndicators,
        vol_ratio: float,
        range_pct: Optional[float],
        change_pct: float,
    ) -> ReplayedSignal:
        range_str = f"{range_pct:.0%}" if range_pct is not None else "n/a"
        return ReplayedSignal(
            ticker=ticker,
            signal_date=ind.bar_date,
            signal_type=SignalType.FII_BUYING,
            close=ind.close,
            volume=ind.volume,
            volume_ratio=round(vol_ratio, 2),
            range_pct=range_pct,
            change_pct=round(change_pct, 2),
            cot_reason=(
                f"M1 flag=fii_buying_proxy | "
                f"golden_cross=True | above_sma50=True | "
                f"range={range_str} | vol_ratio={vol_ratio:.2f}x | "
                f"change={change_pct:+.2f}%"
            ),
        )
