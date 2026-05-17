"""
SwingAdvisorBot — Module 2: AI Analysis Engine
analysis_crew.py — CrewAI crew orchestrating the analysis pipeline

From the CrewAI Master Plan (Section 9):
  CREWAI PROCESS TYPE: Sequential with shared memory
  MANAGER: MarketAnalysisAgent coordinates all others

The AnalysisCrew is the orchestrator that runs the Module 2
analysis pipeline in the correct sequence:

  Step 1: SentimentAnalysisAgent → SentimentReport + SectorAnalysis[]
  Step 2: MarketAnalysisAgent → MarketAnalysis (uses sentiment as context)
  Step 3: Package into AnalysisResult with full metadata

The crew handles:
  → Sequencing: sentiment first, then market analysis
  → Data flow: sentiment output feeds into market analysis
  → Error handling: if sentiment fails, market analysis still runs
  → Caching: respects cache TTLs for both agents
  → Timing: tracks total pipeline latency

Why a crew (not just calling agents directly)?
  1. CrewAI compatibility — when we add Module 3/4/6 agents,
     they plug into the same crew architecture
  2. Shared context — sentiment feeds into market analysis
  3. Future parallelisation — sentiment + sector can run in parallel
  4. Single entry point — engine.py calls crew.run(), not individual agents

Current crew members (Module 2 only):
  → SentimentAnalysisAgent (specialist — runs first)
  → MarketAnalysisAgent (coordinator — runs second, uses sentiment)

Future crew members (added in later modules):
  → RiskAssessmentAgent (Module 3 — reads MarketAnalysis)
  → TradeSetupAgent (Module 4 — reads MarketAnalysis + RiskReport)
  → EducationAgent (Module 7 — reads TradeSetups)
  → ReportAgent (Module 6 — reads everything, produces DailyReport)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from zoneinfo import ZoneInfo

from module1_data_layer.models import MarketData
from module2_analysis_engine.agents.market_analysis_agent import (
    MarketAnalysisAgent,
    market_analysis_agent,
)
from module2_analysis_engine.agents.sentiment_agent import (
    SentimentAnalysisAgent,
    sentiment_agent,
)
from module2_analysis_engine.models import (
    AnalysisDepth,
    AnalysisResult,
    InsufficientDataError,
    SentimentReport,
    UserContext,
)

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.analysis_crew")


def _get_min_stocks(market_status: str) -> int:
    """Return minimum stocks required based on market hours.

    During market hours, Kite returns more stocks reliably.
    During closed/pre-market hours, fewer stocks are available
    and we should still allow analysis with reduced data.

    Args:
        market_status: MarketStatus value (open/closed/pre_market).

    Returns:
        Minimum stock count required for analysis.
    """
    thresholds = {
        "open": 5,
        "pre_market": 3,
        "closed": 3,
    }
    return thresholds.get(market_status, 5)


class AnalysisCrew:
    """CrewAI-style crew that orchestrates the Module 2 analysis pipeline.

    The crew runs agents in sequence, passing outputs downstream:
      1. SentimentAnalysisAgent → SentimentReport + list[SectorAnalysis]
      2. MarketAnalysisAgent → AnalysisResult (incorporates sentiment)

    The crew is the single entry point for all analysis requests.
    engine.py calls crew.run() and gets back an AnalysisResult.

    Usage:
        crew = AnalysisCrew()
        result = await crew.run(
            market_data=market_data,
            user_context=user_context,
        )
        # result.analysis has the full MarketAnalysis
        # result.analysis.sentiment_report has the SentimentReport
        # result.analysis.sector_analyses has the SectorAnalysis[]
    """

    def __init__(
        self,
        market_agent: MarketAnalysisAgent | None = None,
        sentiment_agent_instance: SentimentAnalysisAgent | None = None,
    ):
        """Initialize the crew with its agents.

        Args:
            market_agent: MarketAnalysisAgent instance. Defaults to singleton.
            sentiment_agent_instance: SentimentAnalysisAgent instance. Defaults to singleton.
        """
        self._market_agent = market_agent or market_analysis_agent
        self._sentiment_agent = sentiment_agent_instance or sentiment_agent

    async def run(
        self,
        market_data: MarketData,
        user_context: UserContext | None = None,
        analysis_depth: AnalysisDepth = AnalysisDepth.FULL,
        user_message: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> AnalysisResult:
        """Run the complete analysis crew pipeline.

        Sequential execution:
          Step 1: Validate input data
          Step 2: Run sentiment + sector analysis (if full depth)
          Step 3: Run market analysis (uses sentiment context)
          Step 4: Attach sentiment and sector results to analysis
          Step 5: Return AnalysisResult with full metadata

        Args:
            market_data: MarketData from Module 1 pipeline.
            user_context: UserContext for personalisation.
            analysis_depth: FULL or QUICK analysis.

        Returns:
            AnalysisResult with complete analysis + metadata.

        Raises:
            InsufficientDataError: If MarketData has too few stocks.
        """
        start_time = time.monotonic()

        logger.info(
            f"AnalysisCrew starting: depth={analysis_depth.value}, "
            f"stocks={len(market_data.stocks)}, "
            f"news={len(market_data.news)}, "
            f"sectors={len(market_data.sectors)}."
        )

        # ── Step 1: Validate input ──
        min_required = _get_min_stocks(market_data.market_status)
        stock_count = len(market_data.stocks)

        if stock_count < min_required:
            raise InsufficientDataError(
                stocks_available=stock_count,
                stocks_required=min_required,
                reason=(
                    f"Market status: {market_data.market_status}. "
                    f"Tip: Run during market hours (9:15AM-3:30PM IST) "
                    f"for full stock data."
                ),
            )

        logger.info(
            f"[AnalysisCrew] Stock count: {stock_count}/"
            f"{min_required} minimum. "
            f"Market: {market_data.market_status}. "
            f"Proceeding."
        )

        # ── Quick analysis — skip sentiment, go directly to market agent ──
        if analysis_depth == AnalysisDepth.QUICK:
            result = await self._market_agent.execute(
                market_data=market_data,
                user_context=user_context,
                analysis_depth=AnalysisDepth.QUICK,
                user_message=user_message,
                conversation_history=conversation_history,
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            result.total_latency_ms = elapsed_ms
            logger.info(
                f"AnalysisCrew quick analysis complete: "
                f"mood={result.analysis.market_mood.value}, "
                f"latency={elapsed_ms}ms."
            )
            return result

        # ── Full analysis pipeline ──

        # ── Step 2: Sentiment + Sector analysis ──
        sentiment_report, sector_analyses = await self._run_sentiment_phase(
            market_data=market_data,
        )

        # ── Step 3: Market analysis (core) ──
        result = await self._market_agent.execute(
            market_data=market_data,
            user_context=user_context,
            analysis_depth=AnalysisDepth.FULL,
        )

        # ── Step 4: Attach sentiment and sector results ──
        if sentiment_report is not None:
            result.analysis.sentiment_report = sentiment_report
            # Merge risk events from sentiment into analysis
            if sentiment_report.top_risk_events:
                existing_events = set(result.analysis.risk_events)
                for event in sentiment_report.top_risk_events:
                    if event not in existing_events:
                        result.analysis.risk_events.append(event)

        if sector_analyses:
            result.analysis.sector_analyses = sector_analyses

        # ── Step 5: Final metadata ──
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        result.total_latency_ms = elapsed_ms

        logger.info(
            f"AnalysisCrew full analysis complete: "
            f"mood={result.analysis.market_mood.value} "
            f"(confidence={result.analysis.mood_confidence:.2f}), "
            f"sentiment={sentiment_report.overall_sentiment.value if sentiment_report else 'N/A'}, "
            f"sectors={len(sector_analyses)}, "
            f"quality={result.quality_report.verdict.value if result.quality_report else 'N/A'}, "
            f"tokens={result.total_tokens}, "
            f"latency={elapsed_ms}ms."
        )

        return result

    async def _run_sentiment_phase(
        self,
        market_data: MarketData,
    ) -> tuple[SentimentReport | None, list]:
        """Run sentiment and sector analysis.

        Runs sentiment analysis and sector analysis. If either fails,
        the other still returns. Both failures → (None, []).

        In the future, these two calls can run in parallel since
        they are independent. Currently sequential for simplicity.

        Args:
            market_data: MarketData with news and sector data.

        Returns:
            Tuple of (SentimentReport or None, list[SectorAnalysis]).
        """
        sentiment_report = None
        sector_analyses = []

        # ── Sentiment analysis ──
        try:
            sentiment_report = await self._sentiment_agent.execute(
                market_data=market_data,
            )
            logger.info(
                f"Sentiment phase complete: {sentiment_report.overall_sentiment.value} "
                f"(score: {sentiment_report.sentiment_score:+.2f})."
            )
        except Exception as e:
            logger.error(
                f"Sentiment analysis failed (non-fatal): {e}. "
                f"Market analysis will proceed without sentiment context."
            )

        # ── Sector analysis ──
        try:
            sector_analyses = await self._sentiment_agent.analyse_sectors(
                market_data=market_data,
            )
            logger.info(
                f"Sector analysis complete: {len(sector_analyses)} sectors analysed."
            )
        except Exception as e:
            logger.error(
                f"Sector analysis failed (non-fatal): {e}. "
                f"Market analysis will proceed without sector context."
            )

        return sentiment_report, sector_analyses

    async def run_quick_mood(
        self,
        market_data: MarketData,
    ) -> AnalysisResult:
        """Convenience method for quick mood checks.

        Shorthand for run() with QUICK depth.

        Args:
            market_data: MarketData from Module 1.

        Returns:
            AnalysisResult with quick-depth analysis.
        """
        return await self.run(
            market_data=market_data,
            analysis_depth=AnalysisDepth.QUICK,
        )

    def get_crew_status(self) -> dict:
        """Get the current status of crew members.

        Used by MCP health tools and engine diagnostics.

        Returns:
            Dict with crew member names and their status.
        """
        return {
            "crew_name": "AnalysisCrew",
            "process_type": "sequential",
            "agents": [
                {
                    "name": self._sentiment_agent.agent_name,
                    "role": self._sentiment_agent.role,
                    "status": "ready",
                },
                {
                    "name": self._market_agent.agent_name,
                    "role": self._market_agent.role,
                    "status": "ready",
                },
            ],
            "analysis_depths": [d.value for d in AnalysisDepth],
        }


# Module-level singleton — used by engine.py and MCP tools
analysis_crew = AnalysisCrew()
