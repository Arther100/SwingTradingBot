"""
SwingAdvisorBot — Module 3: Risk Management Engine
calculators/position_calculator.py — Position sizing logic

The 2% Rule implemented with Decimal precision:
  max_risk = capital × risk_pct_limit
  shares = floor(max_risk / risk_per_share)
  position_value = shares × entry_price

This calculator answers ONE question:
  "How many shares can I buy without risking
   more than X% of my capital?"

Position size limit (20% of capital) is also enforced here.
If the 2% rule allows 20 shares but that exceeds 20% of
capital, reduce to fit the position limit.

All math uses Decimal. Never float for money.
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

from module3_risk_engine.rules import RiskRules

logger = logging.getLogger("swing_advisor.position_calculator")


class PositionCalculator:
    """Calculate optimal position size based on risk rules.

    Implements the 2% rule (or 1%/3% based on tolerance):
      Step 1: Calculate max risk in rupees
      Step 2: Calculate shares from risk budget
      Step 3: Check position size limit (20%)
      Step 4: Reduce if position exceeds limit
      Step 5: Return final shares and amounts

    All inputs and outputs are Decimal.

    Usage:
        calc = PositionCalculator()
        result = calc.calculate(
            capital=Decimal("50000.00"),
            entry_price=Decimal("1623.00"),
            stop_loss=Decimal("1548.00"),
            risk_tolerance="moderate",
        )
        # result = {
        #     "shares": 13,
        #     "position_rupees": Decimal("21099.00"),
        #     "position_pct": Decimal("42.20"),
        #     "max_risk_rupees": Decimal("1000.00"),
        #     "total_risk_rupees": Decimal("975.00"),
        #     "risk_pct": Decimal("1.95"),
        #     "risk_per_share": Decimal("75.00"),
        #     "capped_by": None,
        # }
    """

    def calculate(
        self,
        capital: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        risk_tolerance: str = "moderate",
    ) -> dict:
        """Calculate optimal position size.

        Args:
            capital: Total trading capital in INR.
            entry_price: Proposed entry price.
            stop_loss: Proposed stop loss price.
            risk_tolerance: 'conservative', 'moderate', 'aggressive'.

        Returns:
            Dict with shares, position_rupees, position_pct,
            max_risk_rupees, total_risk_rupees, risk_pct,
            risk_per_share, capped_by (None or 'position_limit').
        """
        risk_pct_limit = RiskRules.get_risk_pct(risk_tolerance)
        risk_per_share = entry_price - stop_loss

        logger.info(
            f"[PositionCalc] Entry={entry_price}, "
            f"Stop={stop_loss}, Risk/share={risk_per_share}, "
            f"Capital={capital}, Tolerance={risk_tolerance}"
        )

        # Step 1: Max risk in rupees
        max_risk_rupees = (capital * risk_pct_limit).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Step 2: Shares from risk budget (floor division)
        if risk_per_share <= Decimal("0"):
            logger.warning(
                f"[PositionCalc] Risk per share is {risk_per_share} "
                f"(stop >= entry). Returning 0 shares."
            )
            return {
                "shares": 0,
                "position_rupees": Decimal("0.00"),
                "position_pct": Decimal("0.00"),
                "max_risk_rupees": max_risk_rupees,
                "total_risk_rupees": Decimal("0.00"),
                "risk_pct": Decimal("0.00"),
                "risk_per_share": risk_per_share,
                "capped_by": "invalid_stop_loss",
            }

        shares = int(max_risk_rupees / risk_per_share)
        capped_by = None

        logger.info(
            f"[PositionCalc] Max risk={max_risk_rupees}, "
            f"Shares from risk budget=floor({max_risk_rupees}/{risk_per_share})={shares}"
        )

        # Step 3: Check position size limit (20% of capital)
        max_position_rupees = (
            capital * RiskRules.MAX_POSITION_PCT
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        position_rupees = (
            Decimal(str(shares)) * entry_price
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Step 4: Reduce if position exceeds 20% limit
        if position_rupees > max_position_rupees and shares > 0:
            shares = int(max_position_rupees / entry_price)
            position_rupees = (
                Decimal(str(shares)) * entry_price
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            capped_by = "position_limit"
            logger.info(
                f"[PositionCalc] Position capped by 20% rule. "
                f"Reduced to {shares} shares ({position_rupees} INR)"
            )

        # Step 5: Final calculations
        total_risk_rupees = (
            Decimal(str(shares)) * risk_per_share
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        position_pct = Decimal("0.00")
        risk_pct = Decimal("0.00")
        if capital > Decimal("0"):
            position_pct = (
                (position_rupees / capital) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            risk_pct = (
                (total_risk_rupees / capital) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        logger.info(
            f"[PositionCalc] Final: {shares} shares, "
            f"Position={position_rupees} ({position_pct}%), "
            f"Risk={total_risk_rupees} ({risk_pct}%)"
        )

        return {
            "shares": shares,
            "position_rupees": position_rupees,
            "position_pct": position_pct,
            "max_risk_rupees": max_risk_rupees,
            "total_risk_rupees": total_risk_rupees,
            "risk_pct": risk_pct,
            "risk_per_share": risk_per_share,
            "capped_by": capped_by,
        }

    def validate_requested_shares(
        self,
        requested_shares: int,
        capital: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        risk_tolerance: str = "moderate",
    ) -> dict:
        """Validate a user-requested share count against risk rules.

        Used for REDUCE_SIZE verdicts: if user wants 30 shares
        but only 13 are allowed, return the comparison.

        Args:
            requested_shares: Number of shares user wants.
            capital: Total trading capital.
            entry_price: Proposed entry price.
            stop_loss: Proposed stop loss price.
            risk_tolerance: Risk tolerance level.

        Returns:
            Dict with optimal calculation plus requested vs approved
            comparison fields.
        """
        optimal = self.calculate(
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_tolerance=risk_tolerance,
        )

        risk_per_share = entry_price - stop_loss

        requested_risk = (
            Decimal(str(requested_shares)) * risk_per_share
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        requested_risk_pct = Decimal("0.00")
        if capital > Decimal("0"):
            requested_risk_pct = (
                (requested_risk / capital) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        optimal["requested_shares"] = requested_shares
        optimal["approved_shares"] = optimal["shares"]
        optimal["requested_risk_rupees"] = requested_risk
        optimal["approved_risk_rupees"] = optimal["total_risk_rupees"]
        optimal["risk_pct_at_requested"] = requested_risk_pct
        optimal["risk_pct_at_approved"] = optimal["risk_pct"]
        optimal["needs_reduction"] = requested_shares > optimal["shares"]

        return optimal


# Module-level singleton
position_calculator = PositionCalculator()
