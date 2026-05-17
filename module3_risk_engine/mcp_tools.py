"""
SwingAdvisorBot — Module 3: Risk Management Engine
mcp_tools.py — FastAPI MCP tool endpoints for the risk engine

This file exposes Module 3's risk capabilities as MCP tools
that other modules (M4, M6, M8) can call via HTTP.

From the MCP Tool Registry:
  | calculate_risk       | M3 | POST /tools/calculate_risk       | M4, M6 |
  | get_position_size    | M3 | POST /tools/get_position_size    | M4     |
  | check_portfolio_risk | M3 | GET  /tools/check_portfolio_risk | M4, M8 |
  | get_vix_gate_status  | M3 | GET  /tools/get_vix_gate_status  | M4     |

All tools return a ToolResponse envelope consistent with M1/M2.
All financial values are strings (serialized Decimals) for precision.

No Claude API calls — pure Decimal math, deterministic results.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from module3_risk_engine.calculators.position_calculator import position_calculator
from module3_risk_engine.calculators.vix_calculator import vix_calculator
from module3_risk_engine.models import OpenPosition, TradeProposal
from module5_memory.engine import memory_engine
from module3_risk_engine.validators.portfolio_validator import portfolio_validator
from module3_risk_engine.validators.trade_validator import trade_validator

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.mcp_tools_m3")


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/tools", tags=["M3 Risk Engine"])


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────


class ToolResponse(BaseModel):
    """Standard MCP tool response envelope."""

    tool: str = Field(..., description="Tool name that produced this response")
    status: str = Field(default="success", description="success or error")
    data: Optional[dict] = Field(default=None, description="Tool output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(IST).isoformat(),
        description="IST timestamp of response",
    )


class CalculateRiskRequest(BaseModel):
    """Request body for calculate_risk tool."""

    ticker: str = Field(..., description="NSE ticker symbol, e.g. HDFCBANK")
    entry_price: str = Field(..., description="Entry price as string Decimal")
    target_price: str = Field(..., description="Target price as string Decimal")
    stop_loss: str = Field(..., description="Stop loss price as string Decimal")
    vix_value: str = Field(default="14.00", description="India VIX as string Decimal")
    requested_shares: Optional[int] = Field(
        default=None,
        ge=1,
        description="Specific share count requested (optional)",
    )
    user_id: str = Field(default="XCU700", description="User identifier")
    display_name: str = Field(default="Vijay", description="User display name")
    capital: Optional[str] = Field(
        default=None,
        description="Trading capital as string Decimal (defaults to stub)",
    )
    tolerance: Optional[str] = Field(
        default=None,
        description="Risk tolerance: conservative, moderate, aggressive",
    )


class PositionSizeRequest(BaseModel):
    """Request body for get_position_size tool."""

    entry_price: str = Field(..., description="Entry price as string Decimal")
    stop_loss: str = Field(..., description="Stop loss price as string Decimal")
    capital: Optional[str] = Field(
        default=None,
        description="Trading capital as string Decimal (defaults to stub)",
    )
    tolerance: Optional[str] = Field(
        default=None,
        description="Risk tolerance: conservative, moderate, aggressive",
    )


# ─────────────────────────────────────────────────────────────
# Tool 1: calculate_risk (POST)
# ─────────────────────────────────────────────────────────────


@router.post("/calculate_risk", response_model=ToolResponse)
async def calculate_risk(request: CalculateRiskRequest) -> ToolResponse:
    """Full 10-step risk assessment for a trade proposal.

    This is the primary M3 tool called by M4 (Trade Setup Generator)
    before producing any trade setup card.

    Returns RiskReport as dict with all fields.
    """
    try:
        proposal = TradeProposal(
            ticker=request.ticker,
            entry_price=Decimal(request.entry_price),
            target_price=Decimal(request.target_price),
            stop_loss=Decimal(request.stop_loss),
            user_id=request.user_id,
            requested_shares=request.requested_shares,
        )

        capital = (
            Decimal(request.capital)
            if request.capital
            else memory_engine.get_capital()
        )
        tolerance = request.tolerance or memory_engine.get_risk_tolerance()
        positions = memory_engine.get_open_positions()
        display_name = request.display_name

        report = trade_validator.validate(
            proposal=proposal,
            capital=capital,
            tolerance=tolerance,
            vix_value=Decimal(request.vix_value),
            positions=positions,
            display_name=display_name,
        )

        logger.info(
            f"[calculate_risk] {request.ticker}: {report.verdict.value}"
        )

        return ToolResponse(
            tool="calculate_risk",
            status="success",
            data=report.model_dump(
                mode="json",
                exclude_none=True,
            ),
        )

    except Exception as e:
        logger.error(f"[calculate_risk] Error: {e}", exc_info=True)
        return ToolResponse(
            tool="calculate_risk",
            status="error",
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────
# Tool 2: get_position_size (POST)
# ─────────────────────────────────────────────────────────────


@router.post("/get_position_size", response_model=ToolResponse)
async def get_position_size(request: PositionSizeRequest) -> ToolResponse:
    """Quick position size calculation (2% rule).

    Lightweight tool for M4 to get optimal share count
    without running the full 10-step validation.
    """
    try:
        capital = (
            Decimal(request.capital)
            if request.capital
            else memory_engine.get_capital()
        )
        tolerance = request.tolerance or memory_engine.get_risk_tolerance()

        result = position_calculator.calculate(
            capital=capital,
            entry_price=Decimal(request.entry_price),
            stop_loss=Decimal(request.stop_loss),
            risk_tolerance=tolerance,
        )

        # Convert Decimals to strings for JSON
        serialized = {
            k: str(v) if isinstance(v, Decimal) else v
            for k, v in result.items()
        }

        logger.info(
            f"[get_position_size] Shares={result['shares']}, "
            f"Risk={result['total_risk_rupees']}"
        )

        return ToolResponse(
            tool="get_position_size",
            status="success",
            data=serialized,
        )

    except Exception as e:
        logger.error(f"[get_position_size] Error: {e}", exc_info=True)
        return ToolResponse(
            tool="get_position_size",
            status="error",
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────
# Tool 3: check_portfolio_risk (GET)
# ─────────────────────────────────────────────────────────────


@router.get("/check_portfolio_risk", response_model=ToolResponse)
async def check_portfolio_risk(
    display_name: str = Query(default="Vijay", description="User display name"),
    tolerance: Optional[str] = Query(
        default=None, description="Risk tolerance override"
    ),
) -> ToolResponse:
    """Portfolio health assessment — read-only snapshot.

    Returns total risk, sector exposures, trade count,
    health grade, and advisor note.
    """
    try:
        capital = memory_engine.get_capital()
        tol = tolerance or memory_engine.get_risk_tolerance()
        positions = memory_engine.get_open_positions()

        result = portfolio_validator.assess(
            capital=capital,
            positions=positions,
            tolerance=tol,
            display_name=display_name,
        )

        # Serialize portfolio report
        report_data = result["portfolio_report"].model_dump(
            mode="json",
            exclude_none=True,
        )

        logger.info(
            f"[check_portfolio_risk] Health={result['health_grade']}, "
            f"CanAdd={result['can_add_trade']}"
        )

        return ToolResponse(
            tool="check_portfolio_risk",
            status="success",
            data={
                "portfolio_report": report_data,
                "warnings": result["warnings"],
                "recommendations": result["recommendations"],
                "health_grade": result["health_grade"],
                "can_add_trade": result["can_add_trade"],
                "trades_remaining": result["trades_remaining"],
                "advisor_note": result["advisor_note"],
            },
        )

    except Exception as e:
        logger.error(f"[check_portfolio_risk] Error: {e}", exc_info=True)
        return ToolResponse(
            tool="check_portfolio_risk",
            status="error",
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────
# Tool 4: get_vix_gate_status (GET)
# ─────────────────────────────────────────────────────────────


@router.get("/get_vix_gate_status", response_model=ToolResponse)
async def get_vix_gate_status(
    vix_value: str = Query(default="14.00", description="India VIX value"),
    tolerance: Optional[str] = Query(
        default=None, description="Risk tolerance override"
    ),
) -> ToolResponse:
    """Quick VIX gate check — is it safe for new swing trades?

    Returns gate status (open/closed), VIX signal classification,
    and advisor note.
    """
    try:
        tol = tolerance or memory_engine.get_risk_tolerance()

        status = vix_calculator.check_gate(
            vix_value=Decimal(vix_value),
            tolerance=tol,
        )

        logger.info(
            f"[get_vix_gate_status] VIX={vix_value}, "
            f"Gate={status.gate.value}"
        )

        return ToolResponse(
            tool="get_vix_gate_status",
            status="success",
            data=status.model_dump(mode="json", exclude_none=True),
        )

    except Exception as e:
        logger.error(f"[get_vix_gate_status] Error: {e}", exc_info=True)
        return ToolResponse(
            tool="get_vix_gate_status",
            status="error",
            error=str(e),
        )
