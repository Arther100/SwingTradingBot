"""
SwingAdvisorBot — Module 3: Risk Management Engine
engine.py — Public API entry point

This is the single import other modules need:
  from module3_risk_engine.engine import risk_engine

The engine exposes 4 public methods matching the 4 MCP tools:
  1. calculate_risk()       → Full 10-step CoT validation
  2. get_position_size()    → Quick 2% rule position sizing
  3. check_portfolio_risk() → Portfolio health assessment
  4. get_vix_gate_status()  → VIX gate check

These methods are the Python API. The MCP tools in mcp_tools.py
are the HTTP API — both call the same underlying logic.

All math uses Decimal. Never float for money.
No Claude API calls. Pure Python, deterministic.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from module3_risk_engine.calculators.position_calculator import position_calculator
from module3_risk_engine.calculators.vix_calculator import vix_calculator
from module3_risk_engine.models import (
    OpenPosition,
    RiskReport,
    TradeProposal,
    VixGateStatus,
)
from module5_memory.engine import memory_engine
from module3_risk_engine.validators.portfolio_validator import portfolio_validator
from module3_risk_engine.validators.trade_validator import trade_validator

logger = logging.getLogger("swing_advisor.risk_engine")


class RiskEngine:
    """Public API for Module 3 — Risk Management Engine.

    Usage:
        from module3_risk_engine.engine import risk_engine

        # Full risk assessment
        report = risk_engine.calculate_risk(
            ticker="HDFCBANK",
            entry_price=Decimal("1623.00"),
            target_price=Decimal("1900.00"),
            stop_loss=Decimal("1548.00"),
            vix_value=Decimal("14.2"),
        )

        # Quick position size
        result = risk_engine.get_position_size(
            entry_price=Decimal("1623.00"),
            stop_loss=Decimal("1548.00"),
        )

        # Portfolio health
        result = risk_engine.check_portfolio_risk()

        # VIX gate
        status = risk_engine.get_vix_gate_status(Decimal("14.2"))
    """

    def calculate_risk(
        self,
        ticker: str,
        entry_price: Decimal,
        target_price: Decimal,
        stop_loss: Decimal,
        vix_value: Decimal = Decimal("14.00"),
        requested_shares: Optional[int] = None,
        capital: Optional[Decimal] = None,
        tolerance: Optional[str] = None,
        positions: Optional[list[OpenPosition]] = None,
        display_name: Optional[str] = None,
    ) -> RiskReport:
        """Full 10-step risk assessment for a trade proposal.

        This is the primary entry point for M4 (Trade Setup Generator).
        Every trade setup card requires a RiskReport.

        Args:
            ticker: NSE ticker symbol.
            entry_price: Proposed entry price (Decimal).
            target_price: Proposed target price (Decimal).
            stop_loss: Proposed stop loss price (Decimal).
            vix_value: Current India VIX (Decimal).
            requested_shares: Specific share count (optional).
            capital: Trading capital (defaults to stub).
            tolerance: Risk tolerance (defaults to stub).
            positions: Open positions (defaults to stub).
            display_name: User's name (defaults to stub).

        Returns:
            RiskReport with verdict, position sizing, and reasoning.
        """
        proposal = TradeProposal(
            ticker=ticker,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            requested_shares=requested_shares,
        )

        cap = capital if capital is not None else memory_engine.get_capital()
        tol = tolerance or memory_engine.get_risk_tolerance()
        pos = positions if positions is not None else memory_engine.get_open_positions()
        name = display_name or memory_engine.get_display_name()

        report = trade_validator.validate(
            proposal=proposal,
            capital=cap,
            tolerance=tol,
            vix_value=vix_value,
            positions=pos,
            display_name=name,
        )

        logger.info(
            f"[RiskEngine] calculate_risk({ticker}): "
            f"{report.verdict.value}"
        )

        return report

    def get_position_size(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
        capital: Optional[Decimal] = None,
        tolerance: Optional[str] = None,
    ) -> dict:
        """Quick position size calculation (2% rule).

        Lightweight — no VIX check, no R/R, no portfolio check.
        Just: "How many shares can I buy?"

        Args:
            entry_price: Proposed entry price.
            stop_loss: Proposed stop loss price.
            capital: Trading capital (defaults to stub).
            tolerance: Risk tolerance (defaults to stub).

        Returns:
            Dict with shares, position_rupees, risk amounts, etc.
        """
        cap = capital if capital is not None else memory_engine.get_capital()
        tol = tolerance or memory_engine.get_risk_tolerance()

        result = position_calculator.calculate(
            capital=cap,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_tolerance=tol,
        )

        logger.info(
            f"[RiskEngine] get_position_size: "
            f"{result['shares']} shares"
        )

        return result

    def check_portfolio_risk(
        self,
        capital: Optional[Decimal] = None,
        positions: Optional[list[OpenPosition]] = None,
        tolerance: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> dict:
        """Portfolio health assessment — read-only snapshot.

        Args:
            capital: Trading capital (defaults to stub).
            positions: Open positions (defaults to stub).
            tolerance: Risk tolerance (defaults to stub).
            display_name: User's name (defaults to stub).

        Returns:
            Dict with portfolio_report, warnings, health_grade,
            can_add_trade, trades_remaining, advisor_note.
        """
        cap = capital if capital is not None else memory_engine.get_capital()
        tol = tolerance or memory_engine.get_risk_tolerance()
        pos = positions if positions is not None else memory_engine.get_open_positions()
        name = display_name or memory_engine.get_display_name()

        result = portfolio_validator.assess(
            capital=cap,
            positions=pos,
            tolerance=tol,
            display_name=name,
        )

        logger.info(
            f"[RiskEngine] check_portfolio_risk: "
            f"health={result['health_grade']}"
        )

        return result

    def get_vix_gate_status(
        self,
        vix_value: Decimal = Decimal("14.00"),
        tolerance: Optional[str] = None,
    ) -> VixGateStatus:
        """Quick VIX gate check.

        Args:
            vix_value: Current India VIX.
            tolerance: Risk tolerance (defaults to stub).

        Returns:
            VixGateStatus with gate open/closed and advisor note.
        """
        tol = tolerance or memory_engine.get_risk_tolerance()

        status = vix_calculator.check_gate(
            vix_value=vix_value,
            tolerance=tol,
        )

        logger.info(
            f"[RiskEngine] get_vix_gate_status: "
            f"VIX={vix_value}, gate={status.gate.value}"
        )

        return status


# Module-level singleton
risk_engine = RiskEngine()
