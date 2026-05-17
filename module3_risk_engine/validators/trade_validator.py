"""
SwingAdvisorBot — Module 3: Risk Management Engine
validators/trade_validator.py — 10-step Chain-of-Thought trade validation

This is the CORE of Module 3 — the gatekeeper.
Every trade proposal goes through this 10-step CoT:

  Step 1:  Validate inputs (stop < entry, target > entry)
  Step 2:  VIX gate check (first kill switch)
  Step 3:  Calculate risk per share
  Step 4:  Calculate risk/reward ratio
  Step 5:  Position sizing (2% rule + 20% cap)
  Step 6:  Check requested shares (if user specified)
  Step 7:  Capital adequacy check
  Step 8:  Sector exposure check
  Step 9:  Open trade count check
  Step 10: Final verdict + advisor note

Each step logs its reasoning. The full CoT trail
is stored in RiskReport.cot_reasoning.

Borderline cases → REDUCE_SIZE (never hard reject).
Only clear violations get REJECTED.

All math uses Decimal. Never float for money.
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from module3_risk_engine.calculators.position_calculator import position_calculator
from module3_risk_engine.calculators.portfolio_calculator import portfolio_calculator
from module3_risk_engine.calculators.risk_calculator import risk_calculator
from module3_risk_engine.calculators.vix_calculator import vix_calculator
from module3_risk_engine.models import (
    OpenPosition,
    RiskReport,
    RiskVerdict,
    TradeProposal,
    VixGateResult,
)
from module3_risk_engine.rules import RiskRules

logger = logging.getLogger("swing_advisor.trade_validator")


class TradeValidator:
    """10-step Chain-of-Thought trade validator.

    Orchestrates all calculators to produce a RiskReport.
    Never raises exceptions — always returns a verdict.

    Usage:
        validator = TradeValidator()
        report = validator.validate(
            proposal=TradeProposal(...),
            capital=Decimal("50000.00"),
            tolerance="moderate",
            vix_value=Decimal("14.2"),
            positions=[...],
            display_name="Vijay",
        )
    """

    def validate(
        self,
        proposal: TradeProposal,
        capital: Decimal,
        tolerance: str = "moderate",
        vix_value: Decimal = Decimal("14.00"),
        positions: Optional[list[OpenPosition]] = None,
        display_name: str = "Trader",
    ) -> RiskReport:
        """Execute the 10-step CoT validation.

        Args:
            proposal: What the user wants to trade.
            capital: Total trading capital (INR).
            tolerance: Risk tolerance level.
            vix_value: Current India VIX value.
            positions: List of existing open positions.
            display_name: User's name for advisor note.

        Returns:
            Complete RiskReport with verdict and reasoning.
        """
        if positions is None:
            positions = []

        cot_steps: list[str] = []
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        ticker = proposal.ticker
        entry = proposal.entry_price
        target = proposal.target_price
        stop = proposal.stop_loss

        logger.info(
            f"[TradeValidator] Starting 10-step CoT for {ticker}. "
            f"Entry={entry}, Target={target}, Stop={stop}, "
            f"Capital={capital}, Tolerance={tolerance}, VIX={vix_value}"
        )

        # ══════════════════════════════════════════════════
        # STEP 1: Validate inputs
        # ══════════════════════════════════════════════════
        cot_steps.append(f"Step 1: Validate inputs for {ticker}")

        stop_check = risk_calculator.validate_stop_loss(entry, stop)
        if not stop_check["is_valid"]:
            cot_steps.append(f"  FAIL: {stop_check['reason']}")
            return self._build_rejection(
                ticker=ticker,
                reason="invalid_stop_loss",
                cot_steps=cot_steps,
                checks_passed=checks_passed,
                checks_failed=["invalid_stop_loss: " + stop_check["reason"]],
                vix_value=vix_value,
                advisor_note=(
                    f"{display_name}, your stop loss (₹{stop}) must be "
                    f"below entry price (₹{entry}) for long trades. "
                    f"Please fix and re-submit."
                ),
            )

        target_check = risk_calculator.validate_target(entry, target)
        if not target_check["is_valid"]:
            cot_steps.append(f"  FAIL: {target_check['reason']}")
            return self._build_rejection(
                ticker=ticker,
                reason="invalid_target",
                cot_steps=cot_steps,
                checks_passed=checks_passed,
                checks_failed=["invalid_target: " + target_check["reason"]],
                vix_value=vix_value,
                advisor_note=(
                    f"{display_name}, your target (₹{target}) must be "
                    f"above entry price (₹{entry}) for long trades. "
                    f"Please fix and re-submit."
                ),
            )

        cot_steps.append(
            f"  PASS: Entry=₹{entry}, Stop=₹{stop}, Target=₹{target}. "
            f"Risk/share=₹{stop_check['risk_per_share']}"
        )

        # ══════════════════════════════════════════════════
        # STEP 2: VIX gate check
        # ══════════════════════════════════════════════════
        cot_steps.append(f"Step 2: VIX gate check (VIX={vix_value})")

        vix_status = vix_calculator.check_gate(vix_value, tolerance)

        if vix_status.gate == VixGateResult.CLOSED:
            cot_steps.append(
                f"  FAIL: VIX {vix_value} ≥ limit {vix_status.vix_limit} "
                f"({tolerance}). Gate CLOSED."
            )
            return self._build_rejection(
                ticker=ticker,
                reason="vix_gate_failed",
                cot_steps=cot_steps,
                checks_passed=checks_passed,
                checks_failed=[
                    f"vix_gate_failed: VIX {vix_value} >= {vix_status.vix_limit}"
                ],
                vix_value=vix_value,
                vix_limit=vix_status.vix_limit,
                vix_signal=vix_status.vix_signal,
                advisor_note=vix_status.advisor_note,
            )

        cot_steps.append(
            f"  PASS: VIX {vix_value} < {vix_status.vix_limit}. "
            f"Signal: {vix_status.vix_signal}. Gate OPEN."
        )
        checks_passed.append("vix_gate_passed")

        # ══════════════════════════════════════════════════
        # STEP 3: Calculate risk per share
        # ══════════════════════════════════════════════════
        risk_per_share = entry - stop
        cot_steps.append(
            f"Step 3: Risk per share = ₹{entry} - ₹{stop} = ₹{risk_per_share}"
        )

        # ══════════════════════════════════════════════════
        # STEP 4: Risk/reward ratio
        # ══════════════════════════════════════════════════
        cot_steps.append("Step 4: Risk/reward analysis")

        rr_result = risk_calculator.calculate(
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            shares=1,  # per-share first, will recalculate with actual shares
        )

        if not rr_result["meets_minimum"]:
            cot_steps.append(
                f"  FAIL: R/R = {rr_result['rr_string']} < minimum 1:{RiskRules.MIN_RISK_REWARD}. "
                f"Suggested target: ₹{rr_result['suggested_target']}"
            )
            return self._build_rejection(
                ticker=ticker,
                reason="risk_reward_below_minimum",
                cot_steps=cot_steps,
                checks_passed=checks_passed,
                checks_failed=[
                    f"risk_reward_below_minimum: {rr_result['rr_string']} < 1:{RiskRules.MIN_RISK_REWARD}"
                ],
                vix_value=vix_value,
                vix_limit=vix_status.vix_limit,
                vix_signal=vix_status.vix_signal,
                risk_per_share=risk_per_share,
                gain_per_share=rr_result["gain_per_share"],
                risk_reward_ratio=rr_result["rr_string"],
                suggested_target=rr_result["suggested_target"],
                minimum_required=f"1:{RiskRules.MIN_RISK_REWARD}",
                advisor_note=(
                    f"{display_name}, this {ticker} trade has a risk/reward "
                    f"of {rr_result['rr_string']} — below the minimum "
                    f"1:{RiskRules.MIN_RISK_REWARD}. You're risking "
                    f"₹{risk_per_share}/share to gain only "
                    f"₹{rr_result['gain_per_share']}/share. "
                    f"Move your target to ₹{rr_result['suggested_target']} "
                    f"for a 1:{RiskRules.MIN_RISK_REWARD} ratio, or find "
                    f"a tighter entry."
                ),
            )

        rr_note = "meets ideal" if rr_result["meets_ideal"] else "above minimum"
        cot_steps.append(
            f"  PASS: R/R = {rr_result['rr_string']} ({rr_note}). "
            f"Gain/share=₹{rr_result['gain_per_share']}"
        )
        checks_passed.append("risk_reward_above_minimum")

        # ══════════════════════════════════════════════════
        # STEP 5: Position sizing (2% rule + 20% cap)
        # ══════════════════════════════════════════════════
        cot_steps.append("Step 5: Position sizing (2% rule)")

        pos_result = position_calculator.calculate(
            capital=capital,
            entry_price=entry,
            stop_loss=stop,
            risk_tolerance=tolerance,
        )

        optimal_shares = pos_result["shares"]
        if optimal_shares == 0:
            cot_steps.append(
                f"  FAIL: 0 shares affordable. "
                f"Max risk=₹{pos_result['max_risk_rupees']}, "
                f"Risk/share=₹{risk_per_share}"
            )
            return self._build_rejection(
                ticker=ticker,
                reason="insufficient_capital",
                cot_steps=cot_steps,
                checks_passed=checks_passed,
                checks_failed=[
                    f"insufficient_capital: 0 shares at ₹{entry} with ₹{capital} capital"
                ],
                vix_value=vix_value,
                vix_limit=vix_status.vix_limit,
                vix_signal=vix_status.vix_signal,
                advisor_note=(
                    f"{display_name}, your capital of ₹{capital} is not enough "
                    f"to take even 1 share of {ticker} at ₹{entry} within "
                    f"risk limits. You need at least ₹{risk_per_share / RiskRules.get_risk_pct(tolerance):.0f} "
                    f"capital for this trade."
                ),
            )

        cap_note = ""
        if pos_result["capped_by"] == "position_limit":
            cap_note = " (capped by 20% position limit)"

        cot_steps.append(
            f"  Optimal shares: {optimal_shares}{cap_note}. "
            f"Position: ₹{pos_result['position_rupees']} "
            f"({pos_result['position_pct']}% of capital). "
            f"Risk: ₹{pos_result['total_risk_rupees']} "
            f"({pos_result['risk_pct']}%)"
        )
        checks_passed.append("position_size_within_limit")

        # ══════════════════════════════════════════════════
        # STEP 6: Check requested shares (REDUCE_SIZE?)
        # ══════════════════════════════════════════════════
        cot_steps.append("Step 6: Requested shares check")

        final_shares = optimal_shares
        is_reduce_size = False
        reduce_result = None

        if proposal.requested_shares is not None:
            reduce_result = position_calculator.validate_requested_shares(
                requested_shares=proposal.requested_shares,
                capital=capital,
                entry_price=entry,
                stop_loss=stop,
                risk_tolerance=tolerance,
            )

            if reduce_result["needs_reduction"]:
                is_reduce_size = True
                cot_steps.append(
                    f"  REDUCE: Requested {proposal.requested_shares} shares, "
                    f"approved {optimal_shares}. "
                    f"Requested risk: ₹{reduce_result['requested_risk_rupees']} "
                    f"({reduce_result['risk_pct_at_requested']}%), "
                    f"Approved risk: ₹{reduce_result['approved_risk_rupees']} "
                    f"({reduce_result['risk_pct_at_approved']}%)"
                )
            else:
                final_shares = proposal.requested_shares
                cot_steps.append(
                    f"  PASS: Requested {proposal.requested_shares} shares "
                    f"≤ optimal {optimal_shares}. Approved as-is."
                )
        else:
            cot_steps.append(
                f"  No specific share count requested. Using optimal: {optimal_shares}"
            )

        # ══════════════════════════════════════════════════
        # STEP 7: Capital adequacy
        # ══════════════════════════════════════════════════
        cot_steps.append("Step 7: Capital adequacy check")

        if capital < Decimal(str(RiskRules.MIN_CAPITAL)):
            cot_steps.append(
                f"  FAIL: Capital ₹{capital} < minimum ₹{RiskRules.MIN_CAPITAL}"
            )
            return self._build_rejection(
                ticker=ticker,
                reason="insufficient_capital",
                cot_steps=cot_steps,
                checks_passed=checks_passed,
                checks_failed=[
                    f"insufficient_capital: ₹{capital} < minimum ₹{RiskRules.MIN_CAPITAL}"
                ],
                vix_value=vix_value,
                vix_limit=vix_status.vix_limit,
                vix_signal=vix_status.vix_signal,
                advisor_note=(
                    f"{display_name}, you need at least ₹{RiskRules.MIN_CAPITAL} "
                    f"to swing trade safely. Your capital is ₹{capital}."
                ),
            )

        cot_steps.append(f"  PASS: Capital ₹{capital} ≥ minimum ₹{RiskRules.MIN_CAPITAL}")

        # ══════════════════════════════════════════════════
        # STEP 8: Sector exposure check
        # ══════════════════════════════════════════════════
        cot_steps.append("Step 8: Sector exposure check")

        new_position_value = Decimal(str(final_shares)) * entry
        sector_result = portfolio_calculator.get_sector_exposure(
            capital=capital,
            positions=positions,
            new_ticker=ticker,
            new_position_value=new_position_value,
        )

        if not sector_result["within_limit"]:
            cot_steps.append(
                f"  FAIL: {sector_result['sector']} sector at "
                f"{sector_result['after_pct']}% after trade "
                f"(limit {sector_result['limit_pct']}%). "
                f"Current: {sector_result['current_pct']}%"
            )
            alternatives = portfolio_calculator.get_suggested_alternatives(
                positions=positions,
                exclude_sector=sector_result["sector"],
            )
            return self._build_rejection(
                ticker=ticker,
                reason="sector_overexposure",
                cot_steps=cot_steps,
                checks_passed=checks_passed,
                checks_failed=[
                    f"sector_overexposure: {sector_result['sector']} "
                    f"at {sector_result['after_pct']}% > {sector_result['limit_pct']}%"
                ],
                vix_value=vix_value,
                vix_limit=vix_status.vix_limit,
                vix_signal=vix_status.vix_signal,
                sector=sector_result["sector"],
                current_exposure_pct=sector_result["current_pct"],
                max_exposure_pct=sector_result["limit_pct"],
                suggested_alternatives=alternatives if alternatives else None,
                advisor_note=(
                    f"{display_name}, adding {ticker} would push your "
                    f"{sector_result['sector']} exposure to "
                    f"{sector_result['after_pct']}% — well above the "
                    f"{sector_result['limit_pct']}% limit. "
                    f"You already have {sector_result['current_pct']}% "
                    f"in {sector_result['sector']}. "
                    + (
                        f"Consider diversifying with: "
                        f"{', '.join(alternatives)}."
                        if alternatives
                        else "Consider a different sector."
                    )
                ),
            )

        cot_steps.append(
            f"  PASS: {sector_result['sector']} sector at "
            f"{sector_result['after_pct']}% after trade "
            f"(limit {sector_result['limit_pct']}%)"
        )
        checks_passed.append("sector_exposure_within_limit")

        # ══════════════════════════════════════════════════
        # STEP 9: Open trade count check
        # ══════════════════════════════════════════════════
        cot_steps.append("Step 9: Open trade count check")

        open_count = len(positions)
        if open_count >= RiskRules.MAX_OPEN_TRADES:
            cot_steps.append(
                f"  FAIL: {open_count} open trades "
                f"≥ max {RiskRules.MAX_OPEN_TRADES}"
            )
            return self._build_rejection(
                ticker=ticker,
                reason="max_trades_reached",
                cot_steps=cot_steps,
                checks_passed=checks_passed,
                checks_failed=[
                    f"max_trades_reached: {open_count} >= {RiskRules.MAX_OPEN_TRADES}"
                ],
                vix_value=vix_value,
                vix_limit=vix_status.vix_limit,
                vix_signal=vix_status.vix_signal,
                advisor_note=(
                    f"{display_name}, you already have {open_count} open "
                    f"swing trades (max {RiskRules.MAX_OPEN_TRADES}). "
                    f"Close an existing position before adding {ticker}."
                ),
            )

        cot_steps.append(
            f"  PASS: {open_count} open trades < max {RiskRules.MAX_OPEN_TRADES}"
        )
        checks_passed.append("open_trades_within_limit")

        # ══════════════════════════════════════════════════
        # STEP 10: Final verdict + advisor note
        # ══════════════════════════════════════════════════
        cot_steps.append("Step 10: Final verdict")

        # Recalculate R/R with final shares
        rr_final = risk_calculator.calculate(
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            shares=final_shares,
        )

        position_rupees = (
            Decimal(str(final_shares)) * entry
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        position_pct = Decimal("0.00")
        risk_pct = Decimal("0.00")
        if capital > Decimal("0"):
            position_pct = (
                (position_rupees / capital) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            risk_pct = (
                (rr_final["total_risk"] / capital) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        checks_passed.append("risk_pct_within_limit")

        # Build verdict
        if is_reduce_size:
            verdict = RiskVerdict.REDUCE_SIZE
            cot_steps.append(
                f"  REDUCE_SIZE: {proposal.requested_shares} → {final_shares} shares. "
                f"Risk: ₹{rr_final['total_risk']} ({risk_pct}%)"
            )
            advisor_note = (
                f"{display_name}, {ticker} is a solid setup with "
                f"{rr_final['rr_string']} risk/reward, but {proposal.requested_shares} "
                f"shares would risk ₹{reduce_result['requested_risk_rupees']} "
                f"({reduce_result['risk_pct_at_requested']}% of capital). "
                f"Reducing to {final_shares} shares keeps your risk at "
                f"₹{rr_final['total_risk']} ({risk_pct}%). "
                f"Same trade, safer size."
            )

            return RiskReport(
                ticker=ticker,
                verdict=verdict,
                rejection_reason="position_size_exceeded",
                position_size_shares=final_shares,
                position_size_rupees=position_rupees,
                position_pct_of_capital=position_pct,
                risk_per_share=risk_per_share,
                total_risk_rupees=rr_final["total_risk"],
                risk_pct_of_capital=risk_pct,
                potential_gain_rupees=rr_final["total_gain"],
                risk_reward_ratio=rr_final["rr_string"],
                vix_value=vix_value,
                vix_limit=vix_status.vix_limit,
                vix_signal=vix_status.vix_signal,
                checks_passed=checks_passed,
                checks_failed=[
                    f"position_size_exceeded: requested {proposal.requested_shares} > optimal {final_shares}"
                ],
                requested_shares=proposal.requested_shares,
                approved_shares=final_shares,
                requested_risk_rupees=reduce_result["requested_risk_rupees"],
                approved_risk_rupees=rr_final["total_risk"],
                risk_pct_at_requested=reduce_result["risk_pct_at_requested"],
                risk_pct_at_approved=risk_pct,
                cot_reasoning="\n".join(cot_steps),
                advisor_note=advisor_note,
            )

        # APPROVED
        verdict = RiskVerdict.APPROVED
        rr_quality = "excellent" if rr_final["meets_ideal"] else "acceptable"
        cot_steps.append(
            f"  APPROVED: {final_shares} shares of {ticker}. "
            f"Risk ₹{rr_final['total_risk']} ({risk_pct}%), "
            f"R/R {rr_final['rr_string']} ({rr_quality})"
        )

        advisor_note = (
            f"{display_name}, {ticker} passes all risk checks. "
            f"Buy {final_shares} shares at ₹{entry}, "
            f"stop loss ₹{stop}, target ₹{target}. "
            f"You're risking ₹{rr_final['total_risk']} "
            f"({risk_pct}% of capital) to gain "
            f"₹{rr_final['total_gain']} — a {rr_final['rr_string']} "
            f"risk/reward. Solid risk management."
        )

        logger.info(
            f"[TradeValidator] {ticker}: APPROVED. "
            f"{final_shares} shares, R/R {rr_final['rr_string']}"
        )

        return RiskReport(
            ticker=ticker,
            verdict=verdict,
            position_size_shares=final_shares,
            position_size_rupees=position_rupees,
            position_pct_of_capital=position_pct,
            risk_per_share=risk_per_share,
            total_risk_rupees=rr_final["total_risk"],
            risk_pct_of_capital=risk_pct,
            potential_gain_rupees=rr_final["total_gain"],
            risk_reward_ratio=rr_final["rr_string"],
            vix_value=vix_value,
            vix_limit=vix_status.vix_limit,
            vix_signal=vix_status.vix_signal,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            cot_reasoning="\n".join(cot_steps),
            advisor_note=advisor_note,
        )

    # ──────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────

    def _build_rejection(
        self,
        ticker: str,
        reason: str,
        cot_steps: list[str],
        checks_passed: list[str],
        checks_failed: list[str],
        advisor_note: str,
        vix_value: Optional[Decimal] = None,
        vix_limit: Optional[Decimal] = None,
        vix_signal: Optional[str] = None,
        risk_per_share: Optional[Decimal] = None,
        gain_per_share: Optional[Decimal] = None,
        risk_reward_ratio: Optional[str] = None,
        suggested_target: Optional[Decimal] = None,
        minimum_required: Optional[str] = None,
        sector: Optional[str] = None,
        current_exposure_pct: Optional[Decimal] = None,
        max_exposure_pct: Optional[Decimal] = None,
        suggested_alternatives: Optional[list[str]] = None,
    ) -> RiskReport:
        """Build a REJECTED RiskReport with consistent structure."""

        logger.info(
            f"[TradeValidator] {ticker}: REJECTED — {reason}"
        )

        return RiskReport(
            ticker=ticker,
            verdict=RiskVerdict.REJECTED,
            rejection_reason=reason,
            vix_value=vix_value,
            vix_limit=vix_limit,
            vix_signal=vix_signal,
            risk_per_share=risk_per_share,
            gain_per_share=gain_per_share,
            risk_reward_ratio=risk_reward_ratio,
            suggested_target=suggested_target,
            minimum_required=minimum_required,
            sector=sector,
            current_exposure_pct=current_exposure_pct,
            max_exposure_pct=max_exposure_pct,
            suggested_alternatives=suggested_alternatives,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            cot_reasoning="\n".join(cot_steps),
            advisor_note=advisor_note,
        )


# Module-level singleton
trade_validator = TradeValidator()
