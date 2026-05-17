"""
SwingAdvisorBot — Module 1: Data Layer
mcp_server.py — FastAPI Model Context Protocol server

This server exposes Module 1's data capabilities as structured tool
endpoints that Module 2 (Claude AI) can call. In the MCP pattern,
the LLM agent doesn't fetch data directly — it calls tools that
return structured JSON responses.

5 MCP Tools:
  1. fetch_market_data   → Full pipeline run → complete MarketData JSON
  2. fetch_single_stock  → Single stock quote with signals
  3. get_market_status   → Quick market vitals (VIX + Nifty + status)
  4. get_top_news        → Scored, filtered, high-relevance news
  5. get_pipeline_health → Last health check report

Why FastAPI:
  - async/await natively (all fetchers are async)
  - Auto-generated OpenAPI docs at /docs (useful for debugging)
  - Pydantic integration (our models serialize directly)
  - Lightweight (no ORM, no templates, just JSON endpoints)

Server config:
  - Port: 8001 (avoids conflict with common dev ports)
  - Host: 127.0.0.1 (local only — no external exposure)
  - No auth on endpoints (Module 2 runs locally on same machine)

Usage:
  python -m module1_data_layer.mcp_server
  # Server starts at http://127.0.0.1:8001
  # API docs at http://127.0.0.1:8001/docs

  # From Module 2 (Claude agent):
  # POST /tools/fetch_market_data {"tickers": ["HDFCBANK", "RELIANCE"]}
  # GET  /tools/get_market_status
  # GET  /tools/get_top_news
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from module1_data_layer.config import DEFAULT_WATCHLIST, DataFetchConfig
from module1_data_layer.models import (
    DataFetchError,
    PipelineHealthError,
    TokenBudgetError,
)

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.mcp_server")


# ─────────────────────────────────────────────────────────────
# Request / Response Models — Typed contracts for every tool
# ─────────────────────────────────────────────────────────────


class FetchMarketDataRequest(BaseModel):
    """Request body for the fetch_market_data tool."""

    tickers: list[str] = Field(
        default_factory=lambda: DEFAULT_WATCHLIST.copy(),
        description=(
            "NSE ticker symbols to fetch. "
            "Defaults to the 15-stock core watchlist."
        ),
    )
    max_stocks: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum stocks to include in the response.",
    )
    max_news: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum news items to include.",
    )
    token_budget: int = Field(
        default=2500,
        ge=500,
        le=10000,
        description="Token budget for the serialized MarketData payload.",
    )


class ToolResponse(BaseModel):
    """Standard wrapper for all tool responses.

    Every MCP tool returns this envelope so Module 2 can
    handle success and error cases uniformly.
    """

    success: bool = Field(
        description="Whether the tool call succeeded."
    )
    tool_name: str = Field(
        description="Name of the tool that was called."
    )
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
# Application Lifecycle
# ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle.

    Startup:
      - Configure logging
      - Log server start with timestamp

    Shutdown:
      - Log clean shutdown
      - Clear cache to release memory
    """
    # Startup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        f"SwingAdvisorBot MCP Server starting at "
        f"{datetime.now(IST).strftime('%H:%M:%S IST')}. "
        f"Default watchlist: {len(DEFAULT_WATCHLIST)} stocks."
    )

    # Start Telegram listener for Kite auth URLs (background task)
    _listener_task = None
    _scheduler = None
    try:
        from module1_data_layer.auth.telegram_listener import telegram_listener
        import asyncio

        _listener_task = asyncio.create_task(
            telegram_listener.start_listening()
        )
        logger.info("Telegram listener started — listening for Kite auth URLs.")
    except Exception as e:
        logger.warning(f"Telegram listener not started: {e}")

    # Start report scheduler (5:50 AM token check, 8:00 AM reminder, etc.)
    try:
        from module6_reports.scheduler.report_scheduler import ReportScheduler

        _scheduler = ReportScheduler()
        await _scheduler.start()
        logger.info("Report scheduler started.")
    except Exception as e:
        logger.warning(f"Report scheduler not started: {e}")

    yield

    # Shutdown
    if _listener_task:
        try:
            from module1_data_layer.auth.telegram_listener import telegram_listener as _tl
            _tl.stop()
            _listener_task.cancel()
        except Exception:
            pass

    if _scheduler:
        try:
            await _scheduler.stop()
        except Exception:
            pass

    from module1_data_layer.cache import cache as app_cache

    app_cache.clear()
    logger.info("MCP Server shutting down. Cache cleared.")


app = FastAPI(
    title="SwingAdvisorBot MCP Server",
    description=(
        "Module 1 Data Layer — exposes market data tools for the "
        "AI advisor (Module 2). Fetches real-time NSE data via Kite Connect, "
        "scores news relevance, calculates advisor signals, and enforces "
        "token budgets. All timestamps in IST."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

import os as _os

_cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
_frontend_url = _os.environ.get("FRONTEND_URL", "")
if _frontend_url:
    _cors_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount module routers ────────────────────────────────────
from module2_analysis_engine.mcp_tools import router as m2_router
from module3_risk_engine.mcp_tools import router as m3_router
from module4_setup_generator.mcp_tools import router as m4_router
from module5_memory.mcp_tools import router as m5_router
from module6_reports.mcp_tools import router as m6_router
from module7_education.mcp_tools import router as m7_router

app.include_router(m2_router)
app.include_router(m3_router)
app.include_router(m4_router)
app.include_router(m5_router)
app.include_router(m6_router)
app.include_router(m7_router)


# ─────────────────────────────────────────────────────────────
# Health / Root Endpoints
# ─────────────────────────────────────────────────────────────


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Root endpoint — confirms server is running."""
    return {
        "service": "SwingAdvisorBot MCP Server",
        "module": "Module 1 — Data Layer",
        "status": "running",
        "timestamp": datetime.now(IST).isoformat(),
    }


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Lightweight health check for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(IST).isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# Tool 1: fetch_market_data — Full pipeline run
# ─────────────────────────────────────────────────────────────


@app.post("/tools/fetch_market_data", tags=["tools"])
async def tool_fetch_market_data(
    request: FetchMarketDataRequest,
) -> ToolResponse:
    """Run the full 9-step data pipeline and return complete MarketData.

    This is the primary tool for Module 2. It fetches stocks, news,
    VIX, sectors, and economic data — applies advisor signals,
    runs health checks, and trims to token budget.

    Typical response time: 3-8 seconds (first call), < 1s (cached).

    Returns:
        ToolResponse with data containing the full MarketData JSON.
    """
    from module1_data_layer.pipeline import run_data_pipeline

    config = DataFetchConfig(
        max_stocks=request.max_stocks,
        max_news=request.max_news,
        token_budget=request.token_budget,
    )

    try:
        market_data = await run_data_pipeline(
            tickers=request.tickers,
            config=config,
        )

        return ToolResponse(
            success=True,
            tool_name="fetch_market_data",
            data=market_data.model_dump(mode="json", by_alias=True),
        )

    except DataFetchError as e:
        logger.error(f"fetch_market_data failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=ToolResponse(
                success=False,
                tool_name="fetch_market_data",
                error=str(e),
            ).model_dump(mode="json"),
        )

    except PipelineHealthError as e:
        logger.error(f"fetch_market_data health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=ToolResponse(
                success=False,
                tool_name="fetch_market_data",
                error=str(e),
            ).model_dump(mode="json"),
        )

    except TokenBudgetError as e:
        logger.error(f"fetch_market_data token budget exceeded: {e}")
        raise HTTPException(
            status_code=413,
            detail=ToolResponse(
                success=False,
                tool_name="fetch_market_data",
                error=str(e),
            ).model_dump(mode="json"),
        )


# ─────────────────────────────────────────────────────────────
# Tool 2: fetch_single_stock — Single stock with signals
# ─────────────────────────────────────────────────────────────


@app.get("/tools/fetch_single_stock", tags=["tools"])
async def tool_fetch_single_stock(
    ticker: str = Query(
        ...,
        description="NSE ticker symbol, e.g. HDFCBANK, RELIANCE, TCS",
        min_length=1,
        max_length=20,
    ),
) -> ToolResponse:
    """Fetch complete data for a single NSE stock.

    Returns price, volume analysis, 52-week range, and advisor signals.
    Useful when Module 2 needs to drill into a specific stock
    without running the full pipeline.

    The stock is enriched with 30d average volume, 52w range,
    and advisor_flag + cot_reasoning via the signal calculator.
    """
    from module1_data_layer.fetchers.stock_fetcher import fetch_single_stock
    from module1_data_layer.signals.advisor_signals import calculate_advisor_flag

    ticker_upper = ticker.strip().upper()

    try:
        stock = await fetch_single_stock(ticker_upper)

        # Apply advisor signal if not already set
        if stock.advisor_flag is None:
            flag, reasoning = calculate_advisor_flag(stock)
            stock.advisor_flag = flag
            stock.cot_reasoning = reasoning

        return ToolResponse(
            success=True,
            tool_name="fetch_single_stock",
            data=stock.model_dump(mode="json", by_alias=True),
        )

    except DataFetchError as e:
        logger.error(f"fetch_single_stock failed for {ticker_upper}: {e}")
        raise HTTPException(
            status_code=502,
            detail=ToolResponse(
                success=False,
                tool_name="fetch_single_stock",
                error=str(e),
            ).model_dump(mode="json"),
        )


# ─────────────────────────────────────────────────────────────
# Tool 3: get_market_status — Quick market vitals
# ─────────────────────────────────────────────────────────────


@app.get("/tools/get_market_status", tags=["tools"])
async def tool_get_market_status() -> ToolResponse:
    """Get current market status with VIX and index data.

    Lightweight check — faster than the full pipeline.
    Returns market status (open/closed/pre_market), VIX level
    and signal, Nifty/Sensex values.

    Ideal for Module 2 to check if markets are open before
    deciding whether to run the full pipeline.
    """
    from module1_data_layer.fetchers.vix_fetcher import fetch_market_status_data

    try:
        status_data = await fetch_market_status_data()

        return ToolResponse(
            success=True,
            tool_name="get_market_status",
            data=status_data,
        )

    except Exception as e:
        logger.error(f"get_market_status failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=ToolResponse(
                success=False,
                tool_name="get_market_status",
                error=str(e),
            ).model_dump(mode="json"),
        )


# ─────────────────────────────────────────────────────────────
# Tool 4: get_top_news — Scored, filtered news
# ─────────────────────────────────────────────────────────────


@app.get("/tools/get_top_news", tags=["tools"])
async def tool_get_top_news(
    max_items: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of news items to return.",
    ),
    min_relevance: float = Query(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold.",
    ),
) -> ToolResponse:
    """Fetch top market news scored by relevance to Indian markets.

    Headlines are fetched from NewsAPI, scored through the 6-step
    CoT news scorer, filtered by minimum relevance, and sorted
    by relevance_score descending.

    Each news item includes sentiment, affected_sectors,
    market_impact, and an advisor_note.
    """
    from module1_data_layer.fetchers.news_fetcher import fetch_top_news

    config = DataFetchConfig(
        max_news=max_items,
        min_news_relevance=min_relevance,
    )

    try:
        news_items = await fetch_top_news(config)

        return ToolResponse(
            success=True,
            tool_name="get_top_news",
            data=[item.model_dump(mode="json") for item in news_items],
        )

    except DataFetchError as e:
        logger.error(f"get_top_news failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=ToolResponse(
                success=False,
                tool_name="get_top_news",
                error=str(e),
            ).model_dump(mode="json"),
        )


# ─────────────────────────────────────────────────────────────
# Tool 5: get_pipeline_health — Last health check report
# ─────────────────────────────────────────────────────────────


@app.get("/tools/get_pipeline_health", tags=["tools"])
async def tool_get_pipeline_health() -> ToolResponse:
    """Get the pipeline health report from the last full pipeline run.

    Runs a fresh mini-pipeline to generate a current health report.
    Checks stock fetching, VIX availability, signal completeness,
    timestamp integrity, and token budget compliance.

    If no full pipeline has been run yet in this session, runs
    the full pipeline first to generate the health report.
    """
    from module1_data_layer.cache import cache as app_cache
    from module1_data_layer.pipeline import run_data_pipeline

    # Check if we have a recent pipeline result cached
    # Look for any pipeline cache key
    pipeline_cache_key = f"pipeline:full:{'|'.join(sorted(DEFAULT_WATCHLIST))}"
    cached_data = app_cache.get(pipeline_cache_key)

    if cached_data is not None and cached_data.pipeline_health_report is not None:
        report = cached_data.pipeline_health_report
        return ToolResponse(
            success=True,
            tool_name="get_pipeline_health",
            data=report.model_dump(mode="json"),
        )

    # No cached data — run a fresh pipeline to get a health report
    try:
        config = DataFetchConfig()
        market_data = await run_data_pipeline(
            tickers=DEFAULT_WATCHLIST,
            config=config,
        )

        if market_data.pipeline_health_report is not None:
            return ToolResponse(
                success=True,
                tool_name="get_pipeline_health",
                data=market_data.pipeline_health_report.model_dump(mode="json"),
            )

        return ToolResponse(
            success=True,
            tool_name="get_pipeline_health",
            data={
                "status": market_data.pipeline_status.value,
                "note": "Health report not available. Pipeline ran but report was not generated.",
            },
        )

    except (DataFetchError, PipelineHealthError, TokenBudgetError) as e:
        logger.error(f"get_pipeline_health failed: {e}")
        return ToolResponse(
            success=False,
            tool_name="get_pipeline_health",
            error=str(e),
            data={
                "status": "failed",
                "reason": str(e),
            },
        )


# ─────────────────────────────────────────────────────────────
# Server Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "module1_data_layer.mcp_server:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info",
    )
