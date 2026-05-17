"""
SwingAdvisorBot — Module 5: Memory & Personalization
mcp_tools.py — FastAPI MCP tool endpoints for memory operations

Exposes Module 5's memory capabilities as MCP tools
that other modules can call via HTTP.

MCP Tool Registry:
  | get_user_profile       | M5 | GET  /tools/get_user_profile       | M2, M4 |
  | save_trade             | M5 | POST /tools/save_trade             | M4     |
  | get_trade_history      | M5 | GET  /tools/get_trade_history      | M2, M4 |
  | get_memory_context     | M5 | POST /tools/get_memory_context     | M2     |
  | update_learning        | M5 | POST /tools/update_learning        | M7     |

All financial values are strings (serialized Decimals) for precision.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from module5_memory.memory_provider import MemoryProvider
from module5_memory.models import (
    LearningProgress,
    TradeRecord,
    TradeStatus,
)

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.mcp_tools_m5")


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/tools", tags=["M5 Memory"])


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
        description="Response timestamp IST",
    )


class SaveTradeRequest(BaseModel):
    """Request body for save_trade tool."""

    ticker: str = Field(..., description="NSE ticker symbol")
    entry_price: str = Field(..., description="Entry price as string (Decimal)")
    stop_loss: str = Field(..., description="Stop loss price as string")
    target_price: str = Field(..., description="Target price as string")
    shares: int = Field(..., ge=1, description="Number of shares")
    user_id: str = Field(default="XCU700", description="Zerodha client ID")
    sector: str = Field(default="Other", description="Business sector")
    market_mood: Optional[str] = Field(default=None, description="M2 market mood")
    vix_at_entry: Optional[str] = Field(default=None, description="India VIX at entry")
    setup_source: str = Field(default="m4_generated", description="Setup origin")
    notes: Optional[str] = Field(default=None, description="Trade notes")


class MemoryContextRequest(BaseModel):
    """Request body for get_memory_context tool."""

    user_id: str = Field(default="XCU700", description="Zerodha client ID")
    query: str = Field(default="", description="Query text for semantic search")
    agent_name: Optional[str] = Field(default=None, description="Agent name for scoping")


class UpdateLearningRequest(BaseModel):
    """Request body for update_learning tool."""

    user_id: str = Field(default="XCU700", description="Zerodha client ID")
    concept: str = Field(..., description="Concept name")
    quiz_score: Optional[int] = Field(default=None, ge=0, le=100, description="Quiz score")
    times_taught: int = Field(default=1, ge=1, description="Times taught")


# ─────────────────────────────────────────────────────────────
# Module-level provider
# ─────────────────────────────────────────────────────────────

_provider: MemoryProvider | None = None


def _get_provider() -> MemoryProvider:
    """Lazy-init the MemoryProvider singleton."""
    global _provider
    if _provider is None:
        _provider = MemoryProvider()
    return _provider


# ─────────────────────────────────────────────────────────────
# Tool 1: get_user_profile
# ─────────────────────────────────────────────────────────────


@router.get("/get_user_profile")
async def get_user_profile(
    user_id: str = Query(default="XCU700", description="Zerodha client ID"),
) -> ToolResponse:
    """Get user profile from memory.

    Returns profile with capital, risk tolerance, win rate, etc.
    Creates default profile if not found.
    """
    try:
        provider = _get_provider()
        profile = provider.get_or_create_profile(user_id)

        return ToolResponse(
            tool="get_user_profile",
            data={
                "user_id": profile.user_id,
                "name": profile.name,
                "capital": str(profile.capital),
                "risk_tolerance": profile.risk_tolerance,
                "total_trades": profile.total_trades,
                "winning_trades": profile.winning_trades,
                "win_rate": profile.win_rate,
                "total_pnl": str(profile.total_pnl),
                "open_positions_count": profile.open_positions_count,
            },
        )
    except Exception as e:
        logger.error(f"[MCP] get_user_profile failed: {e}")
        return ToolResponse(tool="get_user_profile", status="error", error=str(e))


# ─────────────────────────────────────────────────────────────
# Tool 2: save_trade
# ─────────────────────────────────────────────────────────────


@router.post("/save_trade")
async def save_trade(req: SaveTradeRequest) -> ToolResponse:
    """Save a new trade to memory (SQLite + Pinecone).

    Accepts all trade fields. Returns trade_id.
    """
    try:
        provider = _get_provider()

        trade = TradeRecord(
            user_id=req.user_id,
            ticker=req.ticker,
            sector=req.sector,
            entry_price=Decimal(req.entry_price),
            stop_loss=Decimal(req.stop_loss),
            target_price=Decimal(req.target_price),
            shares=req.shares,
            market_mood=req.market_mood,
            vix_at_entry=Decimal(req.vix_at_entry) if req.vix_at_entry else None,
            notes=req.notes,
        )

        trade_id = provider.save_trade(trade)

        return ToolResponse(
            tool="save_trade",
            data={
                "trade_id": trade_id,
                "ticker": req.ticker,
                "entry_price": req.entry_price,
                "shares": req.shares,
                "status": "open",
            },
        )
    except Exception as e:
        logger.error(f"[MCP] save_trade failed: {e}")
        return ToolResponse(tool="save_trade", status="error", error=str(e))


# ─────────────────────────────────────────────────────────────
# Tool 3: get_trade_history
# ─────────────────────────────────────────────────────────────


@router.get("/get_trade_history")
async def get_trade_history(
    user_id: str = Query(default="XCU700", description="Zerodha client ID"),
    limit: int = Query(default=20, ge=1, le=100, description="Max trades to return"),
    status: Optional[str] = Query(default=None, description="Filter by status: open/closed/stopped_out"),
) -> ToolResponse:
    """Get trade history from memory.

    Returns list of trades with full lifecycle data.
    """
    try:
        provider = _get_provider()

        trade_status = TradeStatus(status) if status else None
        trades = provider._sqlite.get_trades_by_user(
            user_id, status=trade_status, limit=limit
        )

        trade_list = []
        for t in trades:
            trade_data = {
                "trade_id": t.trade_id,
                "ticker": t.ticker,
                "sector": t.sector,
                "entry_price": str(t.entry_price),
                "stop_loss": str(t.stop_loss),
                "target_price": str(t.target_price),
                "shares": t.shares,
                "entry_date": t.entry_date.isoformat(),
                "status": t.status.value,
            }
            if t.exit_price is not None:
                trade_data["exit_price"] = str(t.exit_price)
            if t.pnl_rupees is not None:
                trade_data["pnl_rupees"] = str(t.pnl_rupees)
            if t.exit_reason:
                trade_data["exit_reason"] = t.exit_reason.value
            trade_list.append(trade_data)

        return ToolResponse(
            tool="get_trade_history",
            data={
                "user_id": user_id,
                "count": len(trade_list),
                "trades": trade_list,
            },
        )
    except Exception as e:
        logger.error(f"[MCP] get_trade_history failed: {e}")
        return ToolResponse(tool="get_trade_history", status="error", error=str(e))


# ─────────────────────────────────────────────────────────────
# Tool 4: get_memory_context
# ─────────────────────────────────────────────────────────────


@router.post("/get_memory_context")
async def get_memory_context(req: MemoryContextRequest) -> ToolResponse:
    """Build ≤300 token memory context for Claude prompt injection.

    This is the main tool M2 calls before every Claude request.
    """
    try:
        provider = _get_provider()
        ctx = provider.get_memory_context(
            user_id=req.user_id,
            query=req.query,
            agent_name=req.agent_name,
        )

        return ToolResponse(
            tool="get_memory_context",
            data={
                "text": ctx.text,
                "token_estimate": ctx.token_estimate,
                "chunks_used": ctx.chunks_used,
                "within_budget": ctx.within_budget,
                "is_empty": ctx.is_empty,
            },
        )
    except Exception as e:
        logger.error(f"[MCP] get_memory_context failed: {e}")
        return ToolResponse(tool="get_memory_context", status="error", error=str(e))


# ─────────────────────────────────────────────────────────────
# Tool 5: update_learning_progress
# ─────────────────────────────────────────────────────────────


@router.post("/update_learning")
async def update_learning(req: UpdateLearningRequest) -> ToolResponse:
    """Update learning progress for a concept.

    Used by M7 (Education) to track what user has been taught.
    """
    try:
        provider = _get_provider()

        # Check if concept already exists
        existing = provider.get_learning(req.user_id, req.concept)

        if existing:
            existing.times_taught += 1
            existing.last_taught = datetime.now(IST)
            if req.quiz_score is not None:
                existing.quiz_score = req.quiz_score
            provider.update_learning(existing)
            progress = existing
        else:
            progress = LearningProgress(
                user_id=req.user_id,
                concept=req.concept,
                quiz_score=req.quiz_score,
                times_taught=req.times_taught,
            )
            provider.update_learning(progress)

        return ToolResponse(
            tool="update_learning",
            data={
                "progress_id": progress.progress_id,
                "concept": progress.concept,
                "times_taught": progress.times_taught,
                "quiz_score": progress.quiz_score,
            },
        )
    except Exception as e:
        logger.error(f"[MCP] update_learning failed: {e}")
        return ToolResponse(tool="update_learning", status="error", error=str(e))
