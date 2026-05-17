"""
SwingAdvisorBot — Module 3: Risk Management Engine
agents/risk_assessment_agent.py — CrewAI agent wrapping the risk engine

This agent is the public face of Module 3 in the CrewAI crew.
It extends SwingAdvisorBaseAgent (from M1) and orchestrates:
  → trade_validator  (10-step CoT for individual trades)
  → portfolio_validator (portfolio health assessment)
  → vix_calculator   (standalone VIX gate check)

The agent does NOT call Claude API — all calculations are
pure Python + Decimal. Fast, deterministic, zero hallucination.

CoT steps logged by this agent:
  Step 1: Load user context (capital, tolerance, positions)
  Step 2: Validate trade proposal via trade_validator
  Step 3: Validate output quality
  Step 4: Return RiskReport

All math uses Decimal. Never float for money.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Optional

from pydantic import Field

from module1_data_layer.agents.base_agent import SwingAdvisorBaseAgent
from module3_risk_engine.models import (
    OpenPosition,
    RiskReport,
    TradeProposal,
)
from module5_memory.engine import memory_engine
from module3_risk_engine.validators.portfolio_validator import portfolio_validator
from module3_risk_engine.validators.trade_validator import trade_validator

logger = logging.getLogger("swing_advisor.risk_assessment_agent")


class RiskAssessmentAgent(SwingAdvisorBaseAgent):
    """Risk assessment CrewAI agent — the gatekeeper.

    No trade passes without this agent's approval.
    Pure Decimal math, no LLM calls, deterministic output.

    Usage:
        agent = RiskAssessmentAgent()
        report = await agent.execute(
            proposal=TradeProposal(
                ticker="HDFCBANK",
                entry_price=Decimal("1623.00"),
                target_price=Decimal("1900.00"),
                stop_loss=Decimal("1548.00"),
            ),
            vix_value=Decimal("14.2"),
        )
    """

    # ── Agent Identity ──
    agent_name: str = Field(
        default="RiskAssessmentAgent",
        description="Unique agent name for logging and crew identification.",
    )
    role: str = Field(
        default="Risk Management Specialist",
        description="CrewAI role — risk gatekeeper.",
    )
    goal: str = Field(
        default=(
            "Evaluate every trade proposal against strict risk rules. "
            "Protect capital first. Only approve trades that meet all "
            "risk criteria: VIX gate, R/R ratio, position sizing, "
            "sector exposure, and open trade limits."
        ),
        description="CrewAI goal — what this agent achieves.",
    )
    backstory: str = Field(
        default=(
            "You are the risk management arm of a 20+ year senior finance "
            "advisor. Your job is to protect the trader's capital above all "
            "else. You use the 2% rule, enforce VIX gates, check sector "
            "concentration, and never let emotions override math. "
            "A trader who loses 50% needs 100% gain to recover — "
            "you exist to prevent that scenario."
        ),
        description="CrewAI backstory — the agent's background.",
    )

    async def execute(
        self,
        proposal: Optional[TradeProposal] = None,
        vix_value: Optional[Decimal] = None,
        capital: Optional[Decimal] = None,
        tolerance: Optional[str] = None,
        positions: Optional[list[OpenPosition]] = None,
        display_name: Optional[str] = None,
        **kwargs: Any,
    ) -> RiskReport:
        """Execute risk assessment for a trade proposal.

        Args:
            proposal: Trade proposal to assess.
            vix_value: Current India VIX (defaults to stub).
            capital: Trading capital (defaults to stub).
            tolerance: Risk tolerance (defaults to stub).
            positions: Open positions (defaults to stub).
            display_name: User's name (defaults to stub).

        Returns:
            RiskReport with verdict and full reasoning.
        """
        self.reset_reasoning()

        # ── Step 1: Load user context ──
        self.log_reasoning(1, "Loading user context")

        if capital is None:
            capital = memory_engine.get_capital()
        if tolerance is None:
            tolerance = memory_engine.get_risk_tolerance()
        if positions is None:
            positions = memory_engine.get_open_positions()
        if display_name is None:
            display_name = memory_engine.get_display_name()
        if vix_value is None:
            vix_value = Decimal("14.00")  # Default safe VIX

        self.log_reasoning(
            1,
            f"Context loaded: capital=₹{capital}, "
            f"tolerance={tolerance}, "
            f"positions={len(positions)}, "
            f"vix={vix_value}, "
            f"user={display_name}",
        )

        # ── Step 2: Validate trade ──
        if proposal is None:
            self.log_reasoning(2, "No proposal provided — returning empty report")
            return RiskReport(
                ticker="NONE",
                verdict="REJECTED",
                rejection_reason="no_proposal",
                advisor_note=(
                    f"{display_name}, no trade proposal was provided. "
                    f"Please specify ticker, entry, target, and stop loss."
                ),
            )

        self.log_reasoning(
            2,
            f"Validating {proposal.ticker}: "
            f"entry=₹{proposal.entry_price}, "
            f"target=₹{proposal.target_price}, "
            f"stop=₹{proposal.stop_loss}",
        )

        report = trade_validator.validate(
            proposal=proposal,
            capital=capital,
            tolerance=tolerance,
            vix_value=vix_value,
            positions=positions,
            display_name=display_name,
        )

        self.log_reasoning(
            2,
            f"Verdict: {report.verdict.value}. "
            f"Shares: {report.position_size_shares}. "
            + (
                f"R/R: {report.risk_reward_ratio}. "
                if report.risk_reward_ratio
                else ""
            )
            + (
                f"Reason: {report.rejection_reason}"
                if report.rejection_reason
                else "All checks passed."
            ),
        )

        # ── Step 3: Validate output quality ──
        self.log_reasoning(3, "Validating output quality")

        is_valid, issues = self.validate_output(report)
        if not is_valid:
            self.log_reasoning(
                3,
                f"Output validation warnings: {'; '.join(issues)}. "
                f"Proceeding with report (never block).",
            )
        else:
            self.log_reasoning(3, "Output validation passed")

        # ── Step 4: Return ──
        self.log_reasoning(
            4,
            f"Returning {report.verdict.value} report for {report.ticker}",
        )

        return report

    async def assess_portfolio(
        self,
        capital: Optional[Decimal] = None,
        positions: Optional[list[OpenPosition]] = None,
        tolerance: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> dict:
        """Assess portfolio health (no specific trade).

        Args:
            capital: Trading capital (defaults to stub).
            positions: Open positions (defaults to stub).
            tolerance: Risk tolerance (defaults to stub).
            display_name: User's name (defaults to stub).

        Returns:
            Dict from portfolio_validator.assess().
        """
        if capital is None:
            capital = memory_engine.get_capital()
        if tolerance is None:
            tolerance = memory_engine.get_risk_tolerance()
        if positions is None:
            positions = memory_engine.get_open_positions()
        if display_name is None:
            display_name = memory_engine.get_display_name()

        return portfolio_validator.assess(
            capital=capital,
            positions=positions,
            tolerance=tolerance,
            display_name=display_name,
        )

    def validate_output(
        self, output: Any
    ) -> tuple[bool, list[str]]:
        """Validate RiskReport output quality.

        Extends base validation with risk-specific checks.
        """
        is_valid, issues = super().validate_output(output)

        if isinstance(output, RiskReport):
            if not output.advisor_note:
                issues.append(
                    "RiskReport missing advisor_note — "
                    "every verdict needs a plain English explanation."
                )
            if not output.cot_reasoning and output.verdict.value != "REJECTED":
                issues.append(
                    "RiskReport missing cot_reasoning — "
                    "approved trades need full reasoning trail."
                )

        return len(issues) == 0, issues


# Module-level singleton
risk_assessment_agent = RiskAssessmentAgent()
