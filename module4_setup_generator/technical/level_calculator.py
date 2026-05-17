"""
SwingAdvisorBot — Module 4: Trade Setup Generator
technical/level_calculator.py — Calculate entry, stop loss, and target levels

This is pure math — no API calls, no AI.
Takes a stock's current price and advisor flag, returns technical levels.

Level calculation logic:
  Entry zone:  current_price ± 0.5% (narrow zone for swing entry)
  Stop loss:   based on advisor flag volatility profile
  Target:      entry + (entry - stop) × R/R multiplier (default 3.0)

Stop loss percentages by flag:
  BREAKOUT_WATCH     → 3.5%  (tight — breakouts either work or fail fast)
  UNUSUAL_ACTIVITY   → 5.0%  (wider — more uncertainty)
  MOMENTUM_BUILDING  → 4.0%  (moderate — trend is forming)
  ACCUMULATION_ZONE  → 5.0%  (wider — accumulation takes time)
  CONSOLIDATION      → 4.5%  (moderate — waiting for direction)
  default            → 5.0%  (safe fallback)

All output values are Decimal with 2 decimal places.
"""

from __future__ import annotations

import logging
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

logger = logging.getLogger("swing_advisor.level_calculator")

TWO_PLACES = Decimal("0.01")

# ─────────────────────────────────────────────────────────────
# Stop Loss Percentages by Advisor Flag
# ─────────────────────────────────────────────────────────────

STOP_LOSS_PCT: dict[str, Decimal] = {
    "BREAKOUT_WATCH": Decimal("3.5"),
    "UNUSUAL_ACTIVITY": Decimal("5.0"),
    "MOMENTUM_BUILDING": Decimal("4.0"),
    "ACCUMULATION_ZONE": Decimal("5.0"),
    "CONSOLIDATION": Decimal("4.5"),
}

DEFAULT_STOP_LOSS_PCT = Decimal("5.0")

# Entry zone half-width (±0.5% from current price)
ENTRY_ZONE_HALF_PCT = Decimal("0.5")

# Default risk/reward multiplier
DEFAULT_RR_MULTIPLIER = Decimal("3.0")


class TechnicalLevels:
    """Calculated technical levels for a trade setup."""

    def __init__(
        self,
        entry_zone_low: Decimal,
        entry_zone_high: Decimal,
        stop_loss: Decimal,
        target_price: Decimal,
        risk_per_share: Decimal,
        reward_per_share: Decimal,
        risk_reward_ratio: str,
    ):
        self.entry_zone_low = entry_zone_low
        self.entry_zone_high = entry_zone_high
        self.stop_loss = stop_loss
        self.target_price = target_price
        self.risk_per_share = risk_per_share
        self.reward_per_share = reward_per_share
        self.risk_reward_ratio = risk_reward_ratio


class TechnicalLevelCalculator:
    """Calculate entry, stop loss, and target levels for a stock.

    All calculations use Decimal for precision.
    No rounding surprises — everything is ROUND_HALF_UP to 2 places.

    Usage:
        calc = TechnicalLevelCalculator()
        levels = calc.calculate(
            current_price=769.55,
            advisor_flag="ACCUMULATION_ZONE",
        )
        print(levels.entry_zone_low)   # Decimal("765.70")
        print(levels.stop_loss)        # Decimal("731.07")
        print(levels.target_price)     # Decimal("885.0x")
    """

    def calculate(
        self,
        current_price: float,
        advisor_flag: Optional[str] = None,
        rr_multiplier: Optional[Decimal] = None,
        custom_stop_pct: Optional[Decimal] = None,
    ) -> TechnicalLevels:
        """Calculate technical levels for a stock.

        Args:
            current_price: Current market price (float from M1).
            advisor_flag: M1 advisor flag string (e.g. "ACCUMULATION_ZONE").
            rr_multiplier: Risk/reward multiplier override (default 3.0).
            custom_stop_pct: Custom stop loss percentage override.

        Returns:
            TechnicalLevels with all calculated values.
        """
        price = Decimal(str(current_price))
        multiplier = rr_multiplier or DEFAULT_RR_MULTIPLIER

        # ── Entry zone: price ± 0.5% ──
        half_width = (price * ENTRY_ZONE_HALF_PCT / Decimal("100")).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )
        entry_low = (price - half_width).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        entry_high = (price + half_width).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

        # ── Stop loss: entry_low - stop_pct% ──
        stop_pct = custom_stop_pct or self._get_stop_pct(advisor_flag)
        stop_distance = (entry_low * stop_pct / Decimal("100")).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )
        stop_loss = (entry_low - stop_distance).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )

        # ── Risk per share ──
        risk_per_share = (entry_low - stop_loss).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )

        # ── Target: entry_high + risk × multiplier ──
        reward_per_share = (risk_per_share * multiplier).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )
        target_price = (entry_high + reward_per_share).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )

        # ── Risk/Reward ratio string ──
        if risk_per_share > 0:
            rr_value = (reward_per_share / risk_per_share).quantize(
                TWO_PLACES, rounding=ROUND_HALF_UP
            )
            risk_reward_ratio = f"1:{rr_value}"
        else:
            risk_reward_ratio = "1:0.00"

        levels = TechnicalLevels(
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            risk_reward_ratio=risk_reward_ratio,
        )

        logger.info(
            f"[LevelCalc] price={price} flag={advisor_flag} "
            f"entry=[{entry_low}-{entry_high}] "
            f"stop={stop_loss} target={target_price} "
            f"R/R={risk_reward_ratio}"
        )

        return levels

    def _get_stop_pct(self, advisor_flag: Optional[str]) -> Decimal:
        """Get stop loss percentage for an advisor flag."""
        if advisor_flag is None:
            return DEFAULT_STOP_LOSS_PCT

        flag_upper = advisor_flag.upper()
        # Handle enum values like "accumulation_zone" or "ACCUMULATION_ZONE"
        if hasattr(advisor_flag, "value"):
            flag_upper = advisor_flag.value.upper()

        return STOP_LOSS_PCT.get(flag_upper, DEFAULT_STOP_LOSS_PCT)


# Module-level singleton
level_calculator = TechnicalLevelCalculator()
