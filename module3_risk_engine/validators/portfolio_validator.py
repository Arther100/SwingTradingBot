"""
SwingAdvisorBot — Module 3: Risk Management Engine
validators/portfolio_validator.py — Portfolio-level validation

Standalone portfolio health check — independent of any trade proposal.
Used by the check_portfolio_risk MCP tool and the agent's
portfolio monitoring flow.

What it checks:
  1. Total capital at risk (across all positions)
  2. Sector concentration warnings
  3. Open trade count vs limit
  4. Available capital for new trades
  5. Overall portfolio health grade

This is a READ-ONLY assessment. It does not approve or reject
any specific trade — that's trade_validator's job.

All math uses Decimal. Never float for money.
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from module3_risk_engine.calculators.portfolio_calculator import portfolio_calculator
from module3_risk_engine.models import OpenPosition, PortfolioRiskReport
from module3_risk_engine.rules import RiskRules

logger = logging.getLogger("swing_advisor.portfolio_validator")


class PortfolioValidator:
    """Portfolio-level risk assessment.

    Takes a snapshot of the current portfolio and returns
    a health assessment with warnings and recommendations.

    Usage:
        validator = PortfolioValidator()
        result = validator.assess(
            capital=Decimal("50000.00"),
            positions=[pos1, pos2],
            tolerance="moderate",
            display_name="Vijay",
        )
    """

    def assess(
        self,
        capital: Decimal,
        positions: Optional[list[OpenPosition]] = None,
        tolerance: str = "moderate",
        display_name: str = "Trader",
    ) -> dict:
        """Full portfolio health assessment.

        Args:
            capital: Total trading capital (INR).
            positions: List of current open positions.
            tolerance: Risk tolerance level.
            display_name: User's name for advisor note.

        Returns:
            Dict with portfolio_report, warnings, health_grade,
            can_add_trade, advisor_note.
        """
        if positions is None:
            positions = []

        # Get portfolio snapshot
        report = portfolio_calculator.calculate(
            capital=capital,
            positions=positions,
        )

        warnings: list[str] = []
        recommendations: list[str] = []

        # Check 1: Total risk level
        risk_pct = report.total_risk_pct
        risk_limit_pct = (
            RiskRules.get_risk_pct(tolerance) * Decimal("100")
            * Decimal(str(report.open_trade_count))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        max_total_risk_pct = (
            RiskRules.get_risk_pct(tolerance) * Decimal("100")
            * Decimal(str(RiskRules.MAX_OPEN_TRADES))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if risk_pct > max_total_risk_pct:
            warnings.append(
                f"Total risk at {risk_pct}% exceeds maximum "
                f"{max_total_risk_pct}% for {tolerance} tolerance"
            )

        # Check 2: Sector concentration
        for sector, pct in report.sector_exposures.items():
            limit_pct = (RiskRules.MAX_SECTOR_PCT * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if pct > limit_pct:
                warnings.append(
                    f"{sector} sector at {pct}% — exceeds "
                    f"{limit_pct}% limit"
                )
            elif pct > limit_pct * Decimal("0.8"):
                recommendations.append(
                    f"{sector} sector at {pct}% — approaching "
                    f"{limit_pct}% limit. Consider diversifying."
                )

        # Check 3: Open trade count
        can_add_trade = report.open_trade_count < report.max_trades
        trades_remaining = report.max_trades - report.open_trade_count

        if not can_add_trade:
            warnings.append(
                f"At maximum {report.max_trades} open trades. "
                f"Close a position before adding new trades."
            )
        elif trades_remaining == 1:
            recommendations.append(
                f"Only 1 trade slot remaining "
                f"({report.open_trade_count}/{report.max_trades})."
            )

        # Check 4: Available capital
        min_useful_capital = Decimal("5000.00")
        if report.available_capital < min_useful_capital and can_add_trade:
            warnings.append(
                f"Available capital ₹{report.available_capital} may be "
                f"too low for meaningful new positions."
            )

        # Determine health grade
        health_grade = self._calculate_health_grade(
            warnings=warnings,
            risk_pct=risk_pct,
            open_count=report.open_trade_count,
            max_trades=report.max_trades,
        )

        # Build advisor note
        advisor_note = self._build_advisor_note(
            display_name=display_name,
            report=report,
            health_grade=health_grade,
            warnings=warnings,
            recommendations=recommendations,
            can_add_trade=can_add_trade,
            trades_remaining=trades_remaining,
        )

        logger.info(
            f"[PortfolioValidator] Health={health_grade}, "
            f"Warnings={len(warnings)}, CanAdd={can_add_trade}, "
            f"Risk={risk_pct}%"
        )

        return {
            "portfolio_report": report,
            "warnings": warnings,
            "recommendations": recommendations,
            "health_grade": health_grade,
            "can_add_trade": can_add_trade,
            "trades_remaining": trades_remaining,
            "advisor_note": advisor_note,
        }

    def _calculate_health_grade(
        self,
        warnings: list[str],
        risk_pct: Decimal,
        open_count: int,
        max_trades: int,
    ) -> str:
        """Calculate portfolio health grade.

        Returns:
            'EXCELLENT' — no warnings, low risk
            'GOOD'      — minor concerns
            'CAUTION'   — approaching limits
            'AT_RISK'   — warnings present, needs attention
        """
        if len(warnings) >= 2:
            return "AT_RISK"
        if len(warnings) == 1:
            return "CAUTION"
        if risk_pct > Decimal("5.00") or open_count >= max_trades - 1:
            return "GOOD"
        return "EXCELLENT"

    def _build_advisor_note(
        self,
        display_name: str,
        report: PortfolioRiskReport,
        health_grade: str,
        warnings: list[str],
        recommendations: list[str],
        can_add_trade: bool,
        trades_remaining: int,
    ) -> str:
        """Build plain English portfolio assessment."""
        parts = [
            f"{display_name}, your portfolio health is {health_grade}."
        ]

        parts.append(
            f"You have {report.open_trade_count} open trades "
            f"with ₹{report.total_invested} invested "
            f"({report.available_capital} available)."
        )

        parts.append(
            f"Total risk: ₹{report.total_risk_rupees} "
            f"({report.total_risk_pct}% of capital)."
        )

        if warnings:
            parts.append("Warnings: " + "; ".join(warnings) + ".")

        if recommendations:
            parts.append("Notes: " + "; ".join(recommendations) + ".")

        if can_add_trade:
            parts.append(
                f"You can add {trades_remaining} more "
                f"{'trade' if trades_remaining == 1 else 'trades'}."
            )

        return " ".join(parts)


# Module-level singleton
portfolio_validator = PortfolioValidator()
