"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
reporting/weight_updater.py — Evidence-based M4 signal weight updater

After each backtest run, this module adjusts the confidence score weights
that module4_setup_generator uses when building trade setups. Signals with
a strong win-rate history get a higher weight; poor signals are penalised.

Weight update logic:
  1. Aggregate win_rate and profit_factor per signal_type across all tickers
     (weighted by number of trades — more trades = more reliable evidence)
  2. Determine multiplier tier from backtest performance:
       STRONG → ×1.2  (win_rate ≥ 58% AND profit_factor ≥ 1.8)
       VALID  → ×1.05 (win_rate ≥ 52% AND profit_factor ≥ 1.3)
       WEAK   → ×0.85 (win_rate ≥ 45%, below valid threshold)
       POOR   → ×0.5  (win_rate < 45% OR profit_factor < 1.0)
  3. new_weight = default_weight × multiplier
  4. Clamp: max(WEIGHT_MIN=5, min(WEIGHT_MAX=50, new_weight))
  5. Store SignalWeight to SQLite via signal_weight_cache
  6. M4 loads these weights at startup via:
       signal_weight_cache.get_weight_dict() → {signal_type: current_weight}

Example (from test spec):
  breakout_watch:  win_rate=58%, PF=1.9 → STRONG tier → 30 × 1.2 = 36.0
  unusual_activity: win_rate=41%, PF=0.8 → POOR tier  → 25 × 0.5 = 12.5

INSUFFICIENT_DATA signals retain their current_weight unchanged
(sample_size < MIN_TRADES_FOR_VERDICT = 10).

Usage:
    updater = WeightUpdater()
    updated_weights = updater.compute_new_weights(
        signal_results=portfolio_result.signal_results,
        backtest_period="May 2025 – May 2026",
    )
    # Returns list[SignalWeight]
    saved = await updater.update_and_store(
        signal_results=portfolio_result.signal_results,
        backtest_period="May 2025 – May 2026",
    )
    # Saves to SQLite and returns list[SignalWeight]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from module_backtest.config import (
    MIN_TRADES_FOR_VERDICT,
    SIGNAL_DEFAULT_WEIGHTS,
    WEIGHT_MAX,
    WEIGHT_MIN,
    WEIGHT_MULTIPLIERS,
)
from module_backtest.models import (
    AdvisorVerdict,
    BacktestResult,
    ExitReason,
    SignalType,
    SignalWeight,
)

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.backtest.weight_updater")

# Thresholds for strong tier (above the VALID threshold)
_STRONG_WIN_RATE: float = 58.0
_STRONG_PROFIT_FACTOR: float = 1.8

# Threshold to move from WEAK to POOR
_POOR_WIN_RATE: float = 45.0
_POOR_PROFIT_FACTOR: float = 1.0


class WeightUpdater:
    """Computes and stores evidence-based M4 signal weights from backtest results.

    Stateless — create once and reuse.

    Usage:
        updater = WeightUpdater()
        # Pure computation (no I/O):
        weights = updater.compute_new_weights(signal_results, backtest_period)
        # Computation + SQLite storage:
        weights = await updater.update_and_store(signal_results, backtest_period)
    """

    # ── Public API ──────────────────────────────────────────

    def compute_new_weights(
        self,
        signal_results: list[BacktestResult],
        backtest_period: str = "",
    ) -> list[SignalWeight]:
        """Compute new weights from backtest results (no I/O).

        Args:
            signal_results: List of BacktestResult from the portfolio run.
                            May contain multiple results per signal_type
                            (one per ticker) — these are aggregated.
            backtest_period: Human-readable period string e.g. 'May 2025–May 2026'.

        Returns:
            list[SignalWeight] — one per signal_type with updated weights.
            Signal types not present in signal_results retain default weights.
        """
        # ── Aggregate metrics per signal_type ────────────────
        # Weighted by number of trades to give more trust to larger samples
        agg = self._aggregate_by_signal_type(signal_results)

        results: list[SignalWeight] = []

        all_signal_types = [
            SignalType.BREAKOUT_WATCH,
            SignalType.ACCUMULATION_ZONE,
            SignalType.UNUSUAL_ACTIVITY,
            SignalType.FII_BUYING,
        ]

        for st in all_signal_types:
            key = st.value
            default_w = SIGNAL_DEFAULT_WEIGHTS.get(key, 10.0)

            if key not in agg or agg[key]["n"] < MIN_TRADES_FOR_VERDICT:
                # Insufficient data — keep default weight unchanged
                sw = SignalWeight(
                    signal_type=st,
                    default_weight=default_w,
                    current_weight=default_w,
                    multiplier=1.0,
                    win_rate=agg[key]["win_rate"] if key in agg else None,
                    profit_factor=agg[key]["profit_factor"] if key in agg else None,
                    sample_size=agg[key]["n"] if key in agg else 0,
                    updated_at=datetime.now(IST),
                    backtest_period=backtest_period or None,
                )
                results.append(sw)
                logger.debug(
                    f"[{key}] Insufficient data (n={agg.get(key, {}).get('n', 0)}) "
                    f"— weight unchanged at {default_w}"
                )
                continue

            stats = agg[key]
            win_rate = stats["win_rate"]
            profit_factor = stats["profit_factor"]
            n = stats["n"]

            tier, multiplier = _get_tier_and_multiplier(win_rate, profit_factor)
            new_weight = _clamp(default_w * multiplier)

            sw = SignalWeight(
                signal_type=st,
                default_weight=default_w,
                current_weight=new_weight,
                multiplier=round(multiplier, 4),
                win_rate=round(win_rate, 2),
                profit_factor=round(profit_factor, 4),
                sample_size=n,
                updated_at=datetime.now(IST),
                backtest_period=backtest_period or None,
            )
            results.append(sw)

            logger.info(
                f"[{key}] {tier.upper()} tier — "
                f"win_rate={win_rate:.1f}%, PF={profit_factor:.2f}, n={n} → "
                f"weight: {default_w} × {multiplier} = {new_weight} "
                f"(clamped to [{WEIGHT_MIN}, {WEIGHT_MAX}])"
            )

        return results

    async def update_and_store(
        self,
        signal_results: list[BacktestResult],
        backtest_period: str = "",
    ) -> list[SignalWeight]:
        """Compute new weights and persist to SQLite.

        Args:
            signal_results: Backtest results from portfolio run.
            backtest_period: Human-readable period description.

        Returns:
            list[SignalWeight] that were stored.
        """
        weights = self.compute_new_weights(signal_results, backtest_period)

        from module_backtest.data.data_cache import signal_weight_cache  # lazy import

        stored = 0
        for sw in weights:
            try:
                signal_weight_cache.store_signal_weight(sw)
                stored += 1
            except Exception as exc:
                logger.error(f"Failed to store weight for {sw.signal_type.value}: {exc}")

        logger.info(
            f"Weight update complete: {stored}/{len(weights)} weights saved to SQLite"
        )
        return weights

    def get_current_weight_dict(self) -> dict[str, float]:
        """Return current weights from SQLite as {signal_type_str: weight}.

        Falls back to default weights if SQLite is empty or unavailable.
        This is the method M4 calls at startup.
        """
        try:
            from module_backtest.data.data_cache import signal_weight_cache

            weight_dict = signal_weight_cache.get_weight_dict()
            if weight_dict:
                logger.debug(f"Loaded {len(weight_dict)} signal weights from SQLite")
                return weight_dict
        except Exception as exc:
            logger.warning(f"Could not load weights from SQLite — using defaults: {exc}")

        return dict(SIGNAL_DEFAULT_WEIGHTS)

    # ── Internal helpers ────────────────────────────────────

    @staticmethod
    def _aggregate_by_signal_type(
        signal_results: list[BacktestResult],
    ) -> dict[str, dict]:
        """Aggregate win_rate and profit_factor per signal_type across tickers.

        Uses weighted average where weight = number of trades.
        This gives more trust to tickers with more trading history.

        Returns:
            dict[signal_type_str, {win_rate, profit_factor, n}]
        """
        # Accumulate weighted sums
        weighted_wr: dict[str, float] = defaultdict(float)
        weighted_pf: dict[str, float] = defaultdict(float)
        total_n: dict[str, int] = defaultdict(int)

        for result in signal_results:
            key = result.signal_type.value
            n = result.metrics.total_trades
            if n < 1:
                continue

            weighted_wr[key] += result.metrics.win_rate * n
            weighted_pf[key] += result.metrics.profit_factor * n
            total_n[key] += n

        aggregated: dict[str, dict] = {}
        for key in total_n:
            n = total_n[key]
            if n > 0:
                aggregated[key] = {
                    "win_rate": weighted_wr[key] / n,
                    "profit_factor": weighted_pf[key] / n,
                    "n": n,
                }

        return aggregated


# ═══════════════════════════════════════════════════════════
# TIER LOGIC — pure functions
# ═══════════════════════════════════════════════════════════


def _get_tier_and_multiplier(
    win_rate: float,
    profit_factor: float,
) -> tuple[str, float]:
    """Determine performance tier and multiplier.

    Priority (highest to lowest):
      strong → valid → weak → poor

    Returns:
        (tier_name, multiplier_value)
    """
    # Strong: best evidence tier — both win rate and PF excellent
    if win_rate >= _STRONG_WIN_RATE and profit_factor >= _STRONG_PROFIT_FACTOR:
        return "strong", WEIGHT_MULTIPLIERS["strong"]

    # Valid: solid edge — meets minimum thresholds for live trading
    if win_rate >= 52.0 and profit_factor >= 1.3:
        return "valid", WEIGHT_MULTIPLIERS["valid"]

    # Weak: marginal edge — win rate acceptable but PF below threshold,
    # OR win rate marginally below valid threshold
    if win_rate >= _POOR_WIN_RATE and profit_factor >= _POOR_PROFIT_FACTOR:
        return "weak", WEIGHT_MULTIPLIERS["weak"]

    # Poor: no reliable edge — penalise this signal type
    return "poor", WEIGHT_MULTIPLIERS["poor"]


def _clamp(value: float) -> float:
    """Clamp weight to [WEIGHT_MIN, WEIGHT_MAX] and round to 2dp."""
    return round(max(WEIGHT_MIN, min(WEIGHT_MAX, value)), 2)
