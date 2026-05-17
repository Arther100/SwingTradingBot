"""
SwingAdvisorBot — Module 4: Trade Setup Generator
mcp_tools.py — FastAPI MCP tool endpoints for setup generation

This file exposes Module 4's capabilities as MCP tools
that other modules (M6 Reports, M8 Frontend) can call via HTTP.

From the MCP Tool Registry:
  | generate_setups      | M4 | POST /tools/generate_setups      | M6, M8 |
  | get_setup_for_stock  | M4 | POST /tools/get_setup_for_stock  | M8     |
  | get_daily_watchlist  | M4 | GET  /tools/get_daily_watchlist  | M6, M8 |

All tools return a ToolResponse envelope consistent with M1/M2/M3.
All prices are string-serialized Decimals for precision.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.mcp_tools_m4")


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/tools", tags=["M4 Trade Setup Generator"])


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


class GenerateSetupsRequest(BaseModel):
    """Request body for generate_setups tool."""

    user_id: str = Field(default="XCU700", description="User identifier")
    display_name: str = Field(default="Vijay", description="User display name")
    capital: float = Field(default=50000.0, ge=10000, description="Trading capital in INR")
    risk_tolerance: str = Field(
        default="moderate",
        description="Risk tolerance: conservative, moderate, aggressive",
    )
    max_setups: int = Field(default=5, ge=1, le=10, description="Maximum setups to generate")
    min_confidence: float = Field(
        default=6.0, ge=1.0, le=10.0,
        description="Minimum confidence score threshold",
    )
    tickers: Optional[list[str]] = Field(
        default=None,
        description="Specific tickers to evaluate (None = all qualified)",
    )
    skip_claude: bool = Field(
        default=False,
        description="Skip Claude reasoning (faster, no API cost)",
    )


class GetSetupForStockRequest(BaseModel):
    """Request body for get_setup_for_stock tool."""

    ticker: str = Field(..., description="NSE ticker symbol, e.g. HDFCBANK")
    user_id: str = Field(default="XCU700", description="User identifier")
    display_name: str = Field(default="Vijay", description="User display name")
    capital: float = Field(default=50000.0, ge=10000, description="Trading capital in INR")
    risk_tolerance: str = Field(
        default="moderate",
        description="Risk tolerance: conservative, moderate, aggressive",
    )
    skip_claude: bool = Field(
        default=False,
        description="Skip Claude reasoning",
    )


# ─────────────────────────────────────────────────────────────
# Tool 1: generate_setups
# ─────────────────────────────────────────────────────────────


@router.post("/generate_setups", response_model=ToolResponse)
async def generate_setups(request: GenerateSetupsRequest) -> ToolResponse:
    """Generate 3-5 swing trade setups from current market data.

    Pipeline: M1 data → screen → levels → M3 risk → score → Claude → package.

    This is the primary tool for M6 morning reports and M8 dashboard.
    """
    try:
        from module4_setup_generator.engine import setup_engine

        package = await setup_engine.generate_setups(
            user_id=request.user_id,
            display_name=request.display_name,
            capital=request.capital,
            risk_tolerance=request.risk_tolerance,
            max_setups=request.max_setups,
            min_confidence=request.min_confidence,
            tickers=request.tickers,
            skip_claude=request.skip_claude,
        )

        return ToolResponse(
            tool="generate_setups",
            status="success",
            data=package.model_dump(mode="json", exclude_none=True),
        )

    except Exception as e:
        logger.error(f"[MCP] generate_setups failed: {e}", exc_info=True)
        return ToolResponse(
            tool="generate_setups",
            status="error",
            error=str(e)[:300],
        )


# ─────────────────────────────────────────────────────────────
# Tool 2: get_setup_for_stock
# ─────────────────────────────────────────────────────────────


@router.post("/get_setup_for_stock", response_model=ToolResponse)
async def get_setup_for_stock(request: GetSetupForStockRequest) -> ToolResponse:
    """Generate a setup for a single specific stock.

    User asks: "Should I buy HDFCBANK?" → This tool answers.
    Runs full pipeline but only for the requested ticker.
    """
    try:
        from module4_setup_generator.engine import setup_engine

        package = await setup_engine.generate_setups(
            user_id=request.user_id,
            display_name=request.display_name,
            capital=request.capital,
            risk_tolerance=request.risk_tolerance,
            max_setups=1,
            min_confidence=1.0,  # Don't filter — show the user what we found
            tickers=[request.ticker],
            skip_claude=request.skip_claude,
        )

        # Return setup or skip reason
        if package.setups:
            setup_data = package.setups[0].model_dump(
                mode="json", exclude_none=True
            )
            return ToolResponse(
                tool="get_setup_for_stock",
                status="success",
                data={
                    "setup": setup_data,
                    "market_mood": package.market_mood,
                    "india_vix": package.india_vix,
                },
            )
        elif package.skipped_setups:
            skip = package.skipped_setups[0]
            return ToolResponse(
                tool="get_setup_for_stock",
                status="success",
                data={
                    "setup": None,
                    "skipped": skip.model_dump(mode="json", exclude_none=True),
                    "market_mood": package.market_mood,
                    "advisor_note": package.advisor_note,
                },
            )
        else:
            return ToolResponse(
                tool="get_setup_for_stock",
                status="success",
                data={
                    "setup": None,
                    "advisor_note": f"No data available for {request.ticker}.",
                },
            )

    except Exception as e:
        logger.error(
            f"[MCP] get_setup_for_stock({request.ticker}) failed: {e}",
            exc_info=True,
        )
        return ToolResponse(
            tool="get_setup_for_stock",
            status="error",
            error=str(e)[:300],
        )


# ─────────────────────────────────────────────────────────────
# Tool 3: get_daily_watchlist
# ─────────────────────────────────────────────────────────────


@router.get("/get_daily_watchlist", response_model=ToolResponse)
async def get_daily_watchlist(
    user_id: str = Query(default="XCU700", description="User identifier"),
    display_name: str = Query(default="Vijay", description="User display name"),
    capital: float = Query(default=50000.0, ge=10000, description="Trading capital"),
    risk_tolerance: str = Query(default="moderate", description="Risk tolerance"),
) -> ToolResponse:
    """Get daily watchlist — top setups for the day.

    Same as generate_setups but with conservative defaults:
    max 5 setups, min confidence 6.0, skip_claude=True for speed.

    Used by M6 for morning brief generation.
    """
    try:
        from module4_setup_generator.engine import setup_engine

        package = await setup_engine.generate_setups(
            user_id=user_id,
            display_name=display_name,
            capital=capital,
            risk_tolerance=risk_tolerance,
            max_setups=5,
            min_confidence=6.0,
            skip_claude=True,  # Fast watchlist — no Claude
        )

        # Build watchlist summary
        watchlist = []
        for s in package.setups:
            watchlist.append({
                "ticker": s.ticker,
                "company_name": s.company_name,
                "sector": s.sector,
                "current_price": str(s.current_price),
                "entry_zone": f"₹{s.entry_zone_low} - ₹{s.entry_zone_high}",
                "target": str(s.target_price),
                "stop_loss": str(s.stop_loss),
                "confidence": s.confidence_score,
                "risk_reward": s.risk_reward_ratio,
                "advisor_flag": s.advisor_flag,
            })

        return ToolResponse(
            tool="get_daily_watchlist",
            status="success",
            data={
                "watchlist": watchlist,
                "count": len(watchlist),
                "market_mood": package.market_mood,
                "india_vix": package.india_vix,
                "advisor_note": package.advisor_note,
                "freshness": package.freshness.value,
            },
        )

    except Exception as e:
        logger.error(f"[MCP] get_daily_watchlist failed: {e}", exc_info=True)
        return ToolResponse(
            tool="get_daily_watchlist",
            status="error",
            error=str(e)[:300],
        )
