"""
SwingAdvisorBot — Module 2: AI Analysis Engine
mcp_tools.py — FastAPI MCP tool endpoints for the analysis engine

This file exposes Module 2's analysis capabilities as MCP tools
that other modules (M4, M6, M8) can call via HTTP.

From the MCP Tool Registry (Section 10):
  | analyse_market   | M2 | POST /tools/analyse_market | M4, M6 |
  | get_market_mood  | M2 | GET  /tools/get_market_mood | M3, M4 |

These tools run on the same MCP server infrastructure (port 8001).
In the final integration, M1 and M2 tools are composed into a
single FastAPI app. During development, this file provides the
FastAPI router that can be mounted onto the M1 server.

Tool contracts:
  1. analyse_market (POST)
     → Input: AnalyseMarketRequest (tickers, user_id, depth)
     → Action: Fetches market data from M1 MCP, runs AnalysisCrew
     → Output: ToolResponse with AnalysisResult as data

  2. get_market_mood (GET)
     → Input: None (uses cached or fresh quick analysis)
     → Action: Runs quick mood check via AnalysisCrew
     → Output: ToolResponse with mood + confidence + situation

Both tools use the same ToolResponse envelope as Module 1 for
uniform handling across the system.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from module1_data_layer.models import MarketData
from module2_analysis_engine.analysis_crew import analysis_crew
from module2_analysis_engine.config import get_claude_settings
from module2_analysis_engine.models import (
    AnalysisDepth,
    FinalAnalysisError,
    InsufficientDataError,
    UserContext,
)

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.mcp_tools_m2")


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────


class AnalyseMarketRequest(BaseModel):
    """Request body for the analyse_market tool.

    This is the primary entry point for downstream modules
    (M4 Trade Setups, M6 Reports, M8 Frontend) to request
    a full or quick market analysis.
    """

    user_id: str = Field(
        default="XCU700",
        description="User identifier for personalisation context.",
    )
    display_name: str = Field(
        default="Vijay",
        description="User display name for personalised advice.",
    )
    total_capital: float = Field(
        default=0.0,
        ge=0.0,
        description="User's total trading capital in INR.",
    )
    risk_tolerance: str = Field(
        default="moderate",
        description="User's risk tolerance: conservative, moderate, aggressive.",
    )
    analysis_depth: str = Field(
        default="full",
        description="Analysis depth: 'full' or 'quick'.",
    )
    market_data_json: str | None = Field(
        default=None,
        description=(
            "Pre-fetched MarketData JSON. If None, the tool fetches "
            "fresh data from M1 MCP server automatically."
        ),
    )
    message: str | None = Field(
        default=None,
        description="User's chat message / question for conversational analysis.",
    )
    conversation_history: list[dict] | None = Field(
        default=None,
        description="Recent conversation history [{role, content}, ...].",
    )


class MoodResponse(BaseModel):
    """Lightweight response for quick mood checks."""

    market_mood: str = Field(
        description="Market mood: bullish, cautious_bullish, neutral, etc.",
    )
    mood_confidence: float = Field(
        description="Confidence in the mood assessment (0.0-1.0).",
    )
    situation: str = Field(
        description="Brief market situation summary.",
    )
    top_opportunities: list[str] = Field(
        default_factory=list,
        description="Top ticker opportunities.",
    )
    top_risks: list[str] = Field(
        default_factory=list,
        description="Top risk tickers.",
    )


class ToolResponse(BaseModel):
    """Standard MCP tool response envelope.

    Matches Module 1's ToolResponse exactly for uniform
    handling across the system.
    """

    success: bool = Field(description="Whether the tool call succeeded.")
    tool_name: str = Field(description="Name of the tool that was called.")
    data: dict | list | None = Field(
        default=None,
        description="Tool output data. Structure varies by tool.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if success is False.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(IST).isoformat(),
        description="When this response was generated (IST).",
    )


# ─────────────────────────────────────────────────────────────
# FastAPI Router — Mounted onto the MCP server
# ─────────────────────────────────────────────────────────────

router = APIRouter(tags=["analysis-tools"])


@router.post(
    "/tools/analyse_market",
    response_model=ToolResponse,
    summary="Run full or quick market analysis",
    description=(
        "Fetches market data from M1 (or uses provided data), "
        "runs the AnalysisCrew pipeline, and returns advisor-quality "
        "MarketAnalysis with quality report and metadata."
    ),
)
async def analyse_market(request: AnalyseMarketRequest) -> ToolResponse:
    """MCP Tool: analyse_market

    Called by: M4 (Trade Setups), M6 (Reports), M8 (Frontend)

    Flow:
      1. Get MarketData (from request or fetch from M1)
      2. Build UserContext from request
      3. Run AnalysisCrew.run()
      4. Return AnalysisResult as ToolResponse

    Args:
        request: AnalyseMarketRequest with user context and options.

    Returns:
        ToolResponse with AnalysisResult data.
    """
    try:
        # ── Get MarketData ──
        if request.market_data_json:
            market_data = MarketData.model_validate_json(request.market_data_json)
        else:
            market_data = await _fetch_market_data_from_m1()

        # ── Build UserContext ──
        user_context = UserContext(
            user_id=request.user_id,
            display_name=request.display_name,
            total_capital=request.total_capital,
            risk_tolerance=request.risk_tolerance,
        )

        # ── Determine depth ──
        depth = (
            AnalysisDepth.QUICK
            if request.analysis_depth == "quick"
            else AnalysisDepth.FULL
        )

        # ── Run analysis ──
        result = await analysis_crew.run(
            market_data=market_data,
            user_context=user_context,
            analysis_depth=depth,
            user_message=request.message,
            conversation_history=request.conversation_history,
        )

        # Serialize result
        result_data = result.model_dump(mode="json")

        logger.info(
            f"analyse_market tool: {result.analysis.market_mood.value} "
            f"(depth={depth.value}, tokens={result.total_tokens}, "
            f"latency={result.total_latency_ms}ms)."
        )

        return ToolResponse(
            success=True,
            tool_name="analyse_market",
            data=result_data,
        )

    except InsufficientDataError as e:
        logger.error(f"analyse_market: Insufficient data — {e}")
        raise HTTPException(
            status_code=503,
            detail=ToolResponse(
                success=False,
                tool_name="analyse_market",
                error=str(e),
            ).model_dump(),
        )

    except FinalAnalysisError as e:
        logger.error(f"analyse_market: Analysis failed — {e}")
        raise HTTPException(
            status_code=502,
            detail=ToolResponse(
                success=False,
                tool_name="analyse_market",
                error=str(e),
            ).model_dump(),
        )

    except Exception as e:
        logger.error(f"analyse_market: Unexpected error — {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ToolResponse(
                success=False,
                tool_name="analyse_market",
                error=f"Unexpected error: {type(e).__name__}: {str(e)[:200]}",
            ).model_dump(),
        )


@router.get(
    "/tools/get_market_mood",
    response_model=ToolResponse,
    summary="Quick market mood assessment",
    description=(
        "Returns a lightweight market mood check: mood label, "
        "confidence, situation summary, and top opportunities/risks. "
        "Faster and cheaper than full analyse_market."
    ),
)
async def get_market_mood() -> ToolResponse:
    """MCP Tool: get_market_mood

    Called by: M3 (Risk Engine), M4 (Trade Setups)

    Flow:
      1. Fetch MarketData from M1
      2. Run AnalysisCrew.run_quick_mood()
      3. Return mood summary as ToolResponse

    Returns:
        ToolResponse with MoodResponse data.
    """
    try:
        market_data = await _fetch_market_data_from_m1()

        result = await analysis_crew.run_quick_mood(market_data=market_data)

        mood_data = MoodResponse(
            market_mood=result.analysis.market_mood.value,
            mood_confidence=result.analysis.mood_confidence,
            situation=result.analysis.situation,
            top_opportunities=result.analysis.top_opportunities,
            top_risks=result.analysis.top_risks,
        )

        logger.info(
            f"get_market_mood tool: {mood_data.market_mood} "
            f"(confidence={mood_data.mood_confidence:.2f})."
        )

        return ToolResponse(
            success=True,
            tool_name="get_market_mood",
            data=mood_data.model_dump(),
        )

    except InsufficientDataError as e:
        logger.error(f"get_market_mood: Insufficient data — {e}")
        raise HTTPException(
            status_code=503,
            detail=ToolResponse(
                success=False,
                tool_name="get_market_mood",
                error=str(e),
            ).model_dump(),
        )

    except Exception as e:
        logger.error(f"get_market_mood: Unexpected error — {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ToolResponse(
                success=False,
                tool_name="get_market_mood",
                error=f"Unexpected error: {type(e).__name__}: {str(e)[:200]}",
            ).model_dump(),
        )


# ─────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────


async def _fetch_market_data_from_m1() -> MarketData:
    """Fetch MarketData from Module 1's MCP server.

    Calls M1's fetch_market_data tool at POST /tools/fetch_market_data
    on the local MCP server and deserializes the response.

    Returns:
        MarketData from Module 1 pipeline.

    Raises:
        HTTPException: If M1 server is unreachable or returns error.
    """
    settings = get_claude_settings()
    m1_url = f"{settings.mcp_base_url}/tools/fetch_market_data"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(
                m1_url,
                json={"max_stocks": 15, "max_news": 5, "token_budget": 5000},
            )
            response.raise_for_status()

        response_data = response.json()

        if not response_data.get("success"):
            error_msg = response_data.get("error", "Unknown M1 error")
            raise HTTPException(
                status_code=502,
                detail=ToolResponse(
                    success=False,
                    tool_name="fetch_market_data_proxy",
                    error=f"M1 data fetch failed: {error_msg}",
                ).model_dump(),
            )

        market_data_dict = response_data.get("data", {})
        return MarketData.model_validate(market_data_dict)

    except httpx.ConnectError:
        logger.error(
            f"Cannot connect to M1 MCP server at {m1_url}. "
            f"Ensure Module 1 server is running on port 8001."
        )
        raise HTTPException(
            status_code=503,
            detail=ToolResponse(
                success=False,
                tool_name="fetch_market_data_proxy",
                error=(
                    f"M1 MCP server unreachable at {settings.mcp_base_url}. "
                    f"Start it with: python -m module1_data_layer.mcp_server"
                ),
            ).model_dump(),
        )

    except httpx.TimeoutException:
        logger.error(f"M1 MCP server timeout at {m1_url}.")
        raise HTTPException(
            status_code=504,
            detail=ToolResponse(
                success=False,
                tool_name="fetch_market_data_proxy",
                error="M1 data fetch timed out after 30 seconds.",
            ).model_dump(),
        )
