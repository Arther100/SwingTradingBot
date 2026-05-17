"""
SwingAdvisorBot — Module 3: Risk Management Engine
calculators/risk_calculator.py — Risk/reward analysis

Calculates risk/reward ratio and validates against
the minimum threshold (1:2 for all tolerance levels).

This calculator answers ONE question:
  "Is the potential reward worth the risk?"

If R/R < 2.0 → the trade is REJECTED with a
suggested target price that would meet minimum R/R.

All math uses Decimal. Never float for money.
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP

from module3_risk_engine.rules import RiskRules

logger = logging.getLogger("swing_advisor.risk_calculator")


class RiskCalculator:
    """Calculate risk/reward metrics for a trade proposal.

    Core formula:
      risk_per_share  = entry_price - stop_loss
      gain_per_share  = target_price - entry_price
      total_risk      = shares × risk_per_share
      total_gain      = shares × gain_per_share
      rr_ratio        = total_gain / total_risk

    Minimum R/R = 2.0 (from RiskRules.MIN_RISK_REWARD)
    Ideal R/R   = 3.0 (from RiskRules.IDEAL_RISK_REWARD)

    Usage:
        calc = RiskCalculator()
        result = calc.calculate(
            entry_price=Decimal("1623.00"),
            target_price=Decimal("1900.00"),
            stop_loss=Decimal("1548.00"),
            shares=13,
        )
        # result = {
        #     "risk_per_share": Decimal("75.00"),
        #     "gain_per_share": Decimal("277.00"),
        #     "total_risk": Decimal("975.00"),
        #     "total_gain": Decimal("3601.00"),
        #     "rr_ratio": Decimal("3.69"),
        #     "rr_string": "1:3.69",
        #     "meets_minimum": True,
        #     "meets_ideal": True,
        #     "suggested_target": None,
        # }
    """

    def calculate(
        self,
        entry_price: Decimal,
        target_price: Decimal,
        stop_loss: Decimal,
        shares: int = 1,
    ) -> dict:
        """Calculate risk/reward metrics.

        Args:
            entry_price: Proposed entry price in INR.
            target_price: Proposed target price in INR.
            stop_loss: Proposed stop loss price in INR.
            shares: Number of shares (for total amounts).

        Returns:
            Dict with risk_per_share, gain_per_share,
            total_risk, total_gain, rr_ratio, rr_string,
            meets_minimum, meets_ideal, suggested_target.
        """
        risk_per_share = entry_price - stop_loss
        gain_per_share = target_price - entry_price
        shares_dec = Decimal(str(shares))

        total_risk = (shares_dec * risk_per_share).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        total_gain = (shares_dec * gain_per_share).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Calculate R/R ratio
        rr_ratio = Decimal("0.00")
        if risk_per_share > Decimal("0") and total_risk > Decimal("0"):
            rr_ratio = (gain_per_share / risk_per_share).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        rr_string = f"1:{rr_ratio}"
        meets_minimum = rr_ratio >= RiskRules.MIN_RISK_REWARD
        meets_ideal = rr_ratio >= RiskRules.IDEAL_RISK_REWARD

        # Calculate suggested target if below minimum
        suggested_target = None
        if not meets_minimum and risk_per_share > Decimal("0"):
            suggested_target = RiskRules.get_suggested_target(
                entry_price=entry_price,
                stop_loss=stop_loss,
            )

        logger.info(
            f"[RiskCalc] Entry={entry_price}, Target={target_price}, "
            f"Stop={stop_loss}, Shares={shares}. "
            f"R/R={rr_string}, Minimum={meets_minimum}, "
            f"Ideal={meets_ideal}"
        )

        return {
            "risk_per_share": risk_per_share,
            "gain_per_share": gain_per_share,
            "total_risk": total_risk,
            "total_gain": total_gain,
            "rr_ratio": rr_ratio,
            "rr_string": rr_string,
            "meets_minimum": meets_minimum,
            "meets_ideal": meets_ideal,
            "suggested_target": suggested_target,
        }

    def validate_stop_loss(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
    ) -> dict:
        """Validate that stop loss is correctly placed.

        For long trades: stop must be below entry.
        Risk per share must be positive.

        Args:
            entry_price: Proposed entry price.
            stop_loss: Proposed stop loss price.

        Returns:
            Dict with is_valid, risk_per_share, reason.
        """
        risk_per_share = entry_price - stop_loss

        if stop_loss >= entry_price:
            logger.warning(
                f"[RiskCalc] Invalid stop loss: "
                f"stop={stop_loss} >= entry={entry_price}"
            )
            return {
                "is_valid": False,
                "risk_per_share": risk_per_share,
                "reason": (
                    f"Stop loss (₹{stop_loss}) must be below "
                    f"entry price (₹{entry_price}) for long trades."
                ),
            }

        if risk_per_share <= Decimal("0"):
            return {
                "is_valid": False,
                "risk_per_share": risk_per_share,
                "reason": "Risk per share must be positive.",
            }

        return {
            "is_valid": True,
            "risk_per_share": risk_per_share,
            "reason": None,
        }

    def validate_target(
        self,
        entry_price: Decimal,
        target_price: Decimal,
    ) -> dict:
        """Validate that target is correctly placed.

        For long trades: target must be above entry.

        Args:
            entry_price: Proposed entry price.
            target_price: Proposed target price.

        Returns:
            Dict with is_valid, gain_per_share, reason.
        """
        gain_per_share = target_price - entry_price

        if target_price <= entry_price:
            logger.warning(
                f"[RiskCalc] Invalid target: "
                f"target={target_price} <= entry={entry_price}"
            )
            return {
                "is_valid": False,
                "gain_per_share": gain_per_share,
                "reason": (
                    f"Target (₹{target_price}) must be above "
                    f"entry price (₹{entry_price}) for long trades."
                ),
            }

        return {
            "is_valid": True,
            "gain_per_share": gain_per_share,
            "reason": None,
        }


# Module-level singleton
risk_calculator = RiskCalculator()
