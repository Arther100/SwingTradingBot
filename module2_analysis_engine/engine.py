"""
SwingAdvisorBot — Module 2: AI Analysis Engine
engine.py — Top-level public API for the analysis engine

This is the single entry point for ALL analysis operations.
Every other module (M3, M4, M5, M6, M7, M8) calls functions
from this file — never the internal agents, crew, or client
directly.

Public API:
  analyse_market()     → Full advisor-quality market analysis
  get_market_mood()    → Quick mood check (lightweight)
  get_engine_health()  → Health check for monitoring
  get_engine_status()  → Detailed status with crew info

Data flow:
  Caller (M4/M6/M8)
    → engine.analyse_market(market_data, user_context)
    → AnalysisCrew.run()
      → SentimentAnalysisAgent.execute()
      → MarketAnalysisAgent.execute()
        → TokenController.prepare_input()
        → ClaudeClient.call_claude()
        → QualityChecker.check()
        → HallucinationGuard.verify()
    → AnalysisResult
    ← returns to caller

Why engine.py exists (and not just import from analysis_crew):
  1. Stable public API — internal refactoring doesn't break callers
  2. Input validation — validates MarketData before passing to crew
  3. Logging — structured entry/exit logging for every analysis call
  4. Error translation — converts internal exceptions to clean errors
  5. Health monitoring — exposes engine health for pipeline checks
  6. Future: rate limiting, circuit breaker, metrics

Module dependency tree (this file is the ONLY public surface):
  engine.py (you are here)
    └── analysis_crew.py (orchestrator)
        ├── agents/market_analysis_agent.py (core analysis)
        │   ├── claude_client.py (API calls)
        │   ├── token_controller.py (budget)
        │   ├── quality_checker.py (self-reflection)
        │   └── hallucination_guard.py (fact-check)
        └── agents/sentiment_agent.py (news analysis)
            └── claude_client.py (API calls)
    └── mcp_tools.py (FastAPI endpoints — calls this engine)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from zoneinfo import ZoneInfo

from module1_data_layer.models import MarketData, MarketStatus
from module2_analysis_engine.analysis_crew import AnalysisCrew, analysis_crew
from module2_analysis_engine.config import (
    MARKET_DATA_BUDGET,
    get_claude_settings,
)
from module2_analysis_engine.token_controller import token_controller
from module2_analysis_engine.models import (
    AnalysisDepth,
    AnalysisResult,
    FinalAnalysisError,
    InsufficientDataError,
    UserContext,
)

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.engine")


def _get_min_stocks(market_status: MarketStatus) -> int:
    """Return minimum stocks required based on market hours.

    During market hours, Kite returns more stocks reliably.
    During closed/pre-market hours, fewer stocks are available
    and we should still allow analysis with reduced data.

    Args:
        market_status: Current MarketStatus from M1 pipeline.

    Returns:
        Minimum stock count required for analysis.
    """
    if market_status == MarketStatus.OPEN:
        return 5
    elif market_status == MarketStatus.PRE_MARKET:
        return 3
    else:  # CLOSED
        return 3


async def analyse_market(
    market_data: MarketData,
    user_context: UserContext | None = None,
    analysis_depth: AnalysisDepth = AnalysisDepth.FULL,
) -> AnalysisResult:
    """Run full or quick market analysis — the primary public API.

    This is THE function that downstream modules call. It validates
    input, runs the analysis crew, and returns a complete result.

    Args:
        market_data: MarketData from Module 1 pipeline.
            Must have is_real_data=True and sufficient stocks.
        user_context: Optional UserContext for personalisation.
            If None, a default context is used.
        analysis_depth: FULL (2400 tokens, ~5-10s) or
            QUICK (1200 tokens, ~2-5s).

    Returns:
        AnalysisResult containing:
          - analysis: MarketAnalysis with mood, situation, advice
          - quality_report: Self-reflection results
          - Token accounting, timing, cache status

    Raises:
        InsufficientDataError: If MarketData has fewer stocks
            than required for the current market status.
        FinalAnalysisError: If Claude API fails after all retries.
        ValueError: If market_data is None or invalid.

    Example:
        from module2_analysis_engine.engine import analyse_market
        result = await analyse_market(market_data, user_context)
        print(result.analysis.market_mood)
        print(result.analysis.situation)
    """
    start_time = time.monotonic()

    # ── Input validation ──
    if market_data is None:
        raise ValueError(
            "market_data cannot be None. "
            "Fetch real data from Module 1 pipeline first."
        )

    if not market_data.is_real_data:
        raise ValueError(
            "market_data.is_real_data is False. "
            "SwingAdvisorBot only works with real market data. "
            "No mock data. No sample data. Ever."
        )

    stock_count = len(market_data.stocks)
    required = _get_min_stocks(market_data.market_status)

    if stock_count < required:
        raise InsufficientDataError(
            stocks_available=stock_count,
            stocks_required=required,
            reason=(
                f"Market status: {market_data.market_status.value}. "
                f"Tip: Run during market hours for full data."
            ),
        )

    logger.info(
        f"Engine: analyse_market called. "
        f"Depth: {analysis_depth.value}. "
        f"Stocks: {stock_count}/{required} minimum. "
        f"Market: {market_data.market_status.value}. "
        f"News: {len(market_data.news)}. "
        f"Sectors: {len(market_data.sectors)}. "
        f"VIX: {market_data.india_vix:.2f}. "
        f"User: {user_context.user_id if user_context else 'default'}."
    )

    # ── Trim MarketData to token budget ──
    market_data = token_controller.trim_to_budget(market_data)
    trimmed_tokens = token_controller.estimate(
        market_data.model_dump_json(
            by_alias=True, exclude_none=True, exclude_defaults=True
        )
    )
    logger.info(
        f"Engine: MarketData after trim: "
        f"{trimmed_tokens} tokens (budget: {MARKET_DATA_BUDGET})"
    )

    # ── Run the crew ──
    result = await analysis_crew.run(
        market_data=market_data,
        user_context=user_context,
        analysis_depth=analysis_depth,
    )

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    logger.info(
        f"Engine: analyse_market complete. "
        f"Mood: {result.analysis.market_mood.value} "
        f"(confidence: {result.analysis.mood_confidence:.2f}). "
        f"Quality: {result.quality_report.verdict.value if result.quality_report else 'N/A'}. "
        f"Tokens: {result.total_tokens}. "
        f"Latency: {elapsed_ms}ms. "
        f"Retries: {result.retry_count}. "
        f"Cache: {result.from_cache}."
    )

    return result


async def get_market_mood(
    market_data: MarketData,
) -> AnalysisResult:
    """Quick market mood check — lightweight, fast, cheap.

    Returns just the mood assessment without full analysis.
    Use this when you need a quick read on the market before
    deciding whether to run a full analysis.

    Costs ~1200 tokens vs ~2400 for full analysis.
    Latency ~2-5s vs ~5-10s for full analysis.

    Args:
        market_data: MarketData from Module 1 pipeline.

    Returns:
        AnalysisResult with QUICK depth analysis.

    Raises:
        InsufficientDataError: If too few stocks.
        FinalAnalysisError: If Claude API fails.

    Example:
        from module2_analysis_engine.engine import get_market_mood
        result = await get_market_mood(market_data)
        if result.analysis.market_mood.value in ("bearish", "extreme_fear"):
            print("Risk-off day — reduce positions")
    """
    return await analyse_market(
        market_data=market_data,
        analysis_depth=AnalysisDepth.QUICK,
    )


def get_engine_health() -> dict:
    """Health check for the analysis engine.

    Returns a status dict that pipeline health monitoring
    and MCP tools can use to verify the engine is operational.

    Checks:
      → Claude API key is configured
      → Analysis crew is initialised
      → Token limits are set correctly

    Returns:
        Dict with health status and component checks.
    """
    settings = get_claude_settings()

    checks = {
        "api_key_configured": bool(settings.anthropic_api_key),
        "model_configured": bool(settings.claude_model),
        "crew_ready": analysis_crew is not None,
        "token_budget": settings.token_budget,
        "mcp_base_url": settings.mcp_base_url,
    }

    all_healthy = all([
        checks["api_key_configured"],
        checks["model_configured"],
        checks["crew_ready"],
    ])

    return {
        "engine": "module2_analysis_engine",
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "crew": analysis_crew.get_crew_status(),
        "checked_at": datetime.now(IST).isoformat(),
    }


def get_engine_status() -> dict:
    """Detailed engine status for diagnostics.

    More verbose than health check. Includes configuration
    details, crew member info, and token budget breakdown.

    Returns:
        Dict with full engine status.
    """
    settings = get_claude_settings()
    from module2_analysis_engine.config import (
        ANALYSIS_CACHE_TTL,
        HARD_TOKEN_LIMIT,
        INPUT_TOKEN_LIMIT,
        MAX_RETRIES,
        OUTPUT_TOKEN_LIMIT,
        SENTIMENT_CACHE_TTL,
    )

    return {
        "engine": "module2_analysis_engine",
        "version": "1.0.0",
        "model": settings.claude_model,
        "token_budget": {
            "hard_limit": HARD_TOKEN_LIMIT,
            "input_limit": INPUT_TOKEN_LIMIT,
            "output_limit": OUTPUT_TOKEN_LIMIT,
        },
        "retry_config": {
            "max_retries": MAX_RETRIES,
        },
        "cache_config": {
            "analysis_ttl": ANALYSIS_CACHE_TTL,
            "sentiment_ttl": SENTIMENT_CACHE_TTL,
        },
        "min_stocks": {
            "open": _get_min_stocks(MarketStatus.OPEN),
            "pre_market": _get_min_stocks(MarketStatus.PRE_MARKET),
            "closed": _get_min_stocks(MarketStatus.CLOSED),
        },
        "crew": analysis_crew.get_crew_status(),
        "health": get_engine_health(),
        "generated_at": datetime.now(IST).isoformat(),
    }
