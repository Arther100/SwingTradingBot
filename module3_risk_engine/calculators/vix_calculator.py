"""
SwingAdvisorBot — Module 3: Risk Management Engine
calculators/vix_calculator.py — VIX gate logic

The VIX gate is the FIRST check in the 10-step CoT.
If VIX is too high → REJECT immediately.
No further analysis needed.

VIX thresholds by risk tolerance:
  conservative → VIX < 15
  moderate     → VIX < 20
  aggressive   → VIX < 25

VIX signal classification (matches M1 VIXSignal):
  < 14  → low_fear
  14-20 → moderate_fear
  20-30 → high_fear
  ≥ 30  → extreme_fear

When gate is closed, the advisor note explains:
  - Current VIX value
  - Why it's dangerous
  - When to re-evaluate
  - "Stay in cash. Protect capital."
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP

from module3_risk_engine.models import VixGateResult, VixGateStatus
from module3_risk_engine.rules import RiskRules

logger = logging.getLogger("swing_advisor.vix_calculator")


class VixCalculator:
    """VIX gate check — first line of defense.

    Call this BEFORE any position sizing or R/R analysis.
    If gate is closed, skip everything else.

    Usage:
        calc = VixCalculator()
        status = calc.check_gate(
            vix_value=Decimal("14.2"),
            tolerance="moderate",
        )
        if status.gate == VixGateResult.CLOSED:
            # No new trades allowed
    """

    def check_gate(
        self,
        vix_value: Decimal,
        tolerance: str = "moderate",
    ) -> VixGateStatus:
        """Check if VIX gate allows new swing trades.

        Args:
            vix_value: Current India VIX as Decimal.
            tolerance: 'conservative', 'moderate', 'aggressive'.

        Returns:
            VixGateStatus with gate open/closed and advisor note.
        """
        vix_limit = RiskRules.get_vix_gate(tolerance)
        vix_signal = RiskRules.classify_vix(vix_value)
        gate = (
            VixGateResult.OPEN
            if vix_value < vix_limit
            else VixGateResult.CLOSED
        )

        advisor_note = self._build_advisor_note(
            vix_value=vix_value,
            vix_limit=vix_limit,
            vix_signal=vix_signal,
            gate=gate,
            tolerance=tolerance,
        )

        logger.info(
            f"[VixCalc] VIX={vix_value}, Limit={vix_limit} "
            f"({tolerance}), Signal={vix_signal}, Gate={gate.value}"
        )

        return VixGateStatus(
            vix_value=vix_value,
            vix_limit=vix_limit,
            tolerance=tolerance,
            gate=gate,
            vix_signal=vix_signal,
            advisor_note=advisor_note,
        )

    def _build_advisor_note(
        self,
        vix_value: Decimal,
        vix_limit: Decimal,
        vix_signal: str,
        gate: VixGateResult,
        tolerance: str,
    ) -> str:
        """Build plain English VIX assessment.

        Gate open → brief confirmation.
        Gate closed → detailed warning with resume condition.
        """
        if gate == VixGateResult.OPEN:
            if vix_signal == "low_fear":
                return (
                    f"VIX at {vix_value} — low fear. "
                    f"Market conditions are calm. "
                    f"Safe for new swing trades."
                )
            return (
                f"VIX at {vix_value} — {vix_signal.replace('_', ' ')}. "
                f"Within {tolerance} tolerance (limit {vix_limit}). "
                f"New swing trades allowed."
            )

        # Gate closed
        if vix_signal == "extreme_fear":
            return (
                f"India VIX at {vix_value} — extreme fear. "
                f"Stop losses get triggered randomly in this "
                f"environment. No new swing trades until VIX "
                f"drops below {vix_limit}. Stay in cash. "
                f"Protect capital."
            )

        return (
            f"India VIX at {vix_value} — {vix_signal.replace('_', ' ')}. "
            f"Above {tolerance} tolerance limit of {vix_limit}. "
            f"Market volatility is too high for new swing positions. "
            f"Re-evaluate when VIX drops below {vix_limit}."
        )


# Module-level singleton
vix_calculator = VixCalculator()
