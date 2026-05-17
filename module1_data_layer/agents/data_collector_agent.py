"""
SwingAdvisorBot — Module 1: Data Layer
agents/data_collector_agent.py — Module 1's primary agent

This is the workhorse of Module 1. DataCollectorAgent orchestrates
all data fetching, signal calculation, health checking, and token
budget trimming into a single execute() call that produces a
complete MarketData object for the advisor.

In the CrewAI crew architecture, this agent will work alongside:
  - MarketAnalysisAgent  (Module 2 — analyzes the data via Claude)
  - RiskAssessmentAgent  (Module 3 — evaluates position risk)
  - TradeSetupAgent      (Module 4 — generates trade setups)
  - EducationAgent       (Module 2 — teaches trading concepts)

DataCollectorAgent is the FIRST agent to run in every crew cycle.
It sets the stage — all other agents depend on its output quality.

Execution flow (9-step CoT):
  Step 1: Check market hours → set market_status
  Step 2: Check cache → return if fresh
  Step 3: Fetch stocks with rate limiter
  Step 4: Fetch news + score relevance
  Step 5: Fetch VIX + macro data
  Step 6: Calculate all advisor signals
  Step 7: Run health check + self-reflection
  Step 8: Trim to token budget
  Step 9: Return MarketData

The full implementation of this flow lives in pipeline.py (File 15).
This agent class provides the structured entry point and delegates
to the pipeline orchestrator for the actual data assembly.
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import Field
from zoneinfo import ZoneInfo

from module1_data_layer.agents.base_agent import SwingAdvisorBaseAgent
from module1_data_layer.config import DEFAULT_WATCHLIST, DataFetchConfig
from module1_data_layer.models import (
    DataFetchError,
    MarketData,
    PipelineHealthError,
    TokenBudgetError,
)

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.agents.data_collector")


class DataCollectorAgent(SwingAdvisorBaseAgent):
    """Module 1's agent — fetches and enriches all market data.

    This agent is responsible for producing the MarketData object
    that feeds every downstream module. The quality of the advisor's
    recommendations is directly proportional to the quality of data
    this agent produces.

    Quality contract:
      → Every stock has an advisor_flag and cot_reasoning.
      → Every news item has a relevance_score >= 0.70.
      → India VIX is always present with vix_signal classification.
      → All timestamps are IST (Asia/Kolkata).
      → is_real_data is always True — never mock, never fake.
      → Token budget is respected (2500 tokens max).
      → Pipeline health check passes before data is returned.

    In Module 2's crew, this agent's output becomes the shared
    context that MarketAnalysisAgent, RiskAssessmentAgent, and
    TradeSetupAgent all reason about. Garbage in → garbage out.

    Usage:
        agent = DataCollectorAgent()
        market_data = await agent.execute(
            tickers=["HDFCBANK", "RELIANCE", "TCS"],
            config=DataFetchConfig()
        )
        # market_data is a complete, validated MarketData object
    """

    # ── Agent Identity ──
    agent_name: str = Field(
        default="DataCollectorAgent",
        description="Unique name for logging and crew identification.",
    )
    role: str = Field(
        default="Senior Market Data Analyst",
        description=(
            "In the crew, this agent is the data specialist — "
            "responsible for real-time NSE data with advisor-quality signals."
        ),
    )
    goal: str = Field(
        default=(
            "Fetch, validate, and enrich real-time NSE market data so that "
            "the senior finance advisor has everything needed to make informed "
            "swing trading recommendations. Every data point must carry context "
            "and signals — bare numbers are useless to the advisor."
        ),
        description="What this agent is trying to achieve in every execution cycle.",
    )
    backstory: str = Field(
        default=(
            "You are the eyes and ears of a 20+ year senior finance advisor. "
            "Every morning, you prepare a comprehensive market briefing covering "
            "stock prices, volume anomalies, India VIX fear levels, sector rotation, "
            "and high-impact news. Your data feeds the advisor's AI brain — if your "
            "data is shallow, the advisor sounds shallow. You have direct access to "
            "Kite Connect (Zerodha), NewsAPI, and FRED economic data. You never "
            "fabricate data. You never return stale prices as live. You always "
            "explain what the data means through Chain of Thought reasoning."
        ),
        description="Background context for CrewAI LLM integration in Module 2.",
    )

    async def execute(
        self,
        tickers: list[str] | None = None,
        config: DataFetchConfig | None = None,
    ) -> MarketData:
        """Execute the complete data collection pipeline.

        This is the single entry point for all market data fetching.
        It orchestrates a 9-step Chain of Thought process that produces
        a validated, signal-rich MarketData object within token budget.

        The actual pipeline implementation is in pipeline.py (File 15).
        This method provides the agent-level wrapper with:
          - Default parameter handling
          - CoT logging integration
          - Output validation via base class
          - Error wrapping with advisor-quality messages

        CoT Steps:
          Step 1: Determine market status from current IST time.
          Step 2: Check cache for fresh MarketData (return early if valid).
          Step 3: Fetch stock data via Kite Connect with rate limiting.
          Step 4: Fetch and score news via NewsAPI.
          Step 5: Fetch India VIX and macro data (FRED).
          Step 6: Calculate advisor signals for all stocks.
          Step 7: Run 7-step pipeline health check (self-reflection).
          Step 8: Trim MarketData to 2500 token budget.
          Step 9: Return validated MarketData to caller.

        Args:
            tickers: NSE ticker symbols to fetch. Defaults to DEFAULT_WATCHLIST
                     (15 core large-caps: HDFCBANK, RELIANCE, TCS, etc.)
            config: DataFetchConfig controlling limits, TTLs, and budget.
                    Defaults to standard config (15 stocks, 5 news, 2500 tokens).

        Returns:
            MarketData: Complete, validated, signal-rich market data object
                        ready for Module 2 (Claude AI analysis).

        Raises:
            DataFetchError: When a critical data source fails and cannot recover.
                           Includes source name, reason, and suggested action.
            PipelineHealthError: When the 7-step health check finds critical issues.
                                Includes the failed step number and reason.
            TokenBudgetError: When data cannot be trimmed within 2500 token budget
                             after all 5 trimming steps are exhausted.
        """
        self.reset_reasoning()

        effective_tickers = tickers if tickers is not None else DEFAULT_WATCHLIST
        effective_config = config if config is not None else DataFetchConfig()

        self.log_reasoning(
            step=1,
            thought=(
                f"Starting data collection pipeline at "
                f"{datetime.now(IST).strftime('%H:%M:%S IST')}. "
                f"Fetching {len(effective_tickers)} tickers: "
                f"{', '.join(effective_tickers[:5])}{'...' if len(effective_tickers) > 5 else ''}. "
                f"Token budget: {effective_config.token_budget}."
            ),
        )

        # Import here to avoid circular import — pipeline.py imports fetchers
        # which import models, but this agent is imported by pipeline too.
        from module1_data_layer.pipeline import run_data_pipeline

        try:
            market_data = await run_data_pipeline(
                tickers=effective_tickers,
                config=effective_config,
                agent=self,
            )

            # Self-reflection: validate output via base class
            is_valid, issues = self.validate_output(market_data)
            if not is_valid:
                logger.warning(
                    f"DataCollectorAgent output validation raised {len(issues)} issue(s). "
                    f"Data is being returned but advisor should be aware of quality concerns: "
                    f"{'; '.join(issues)}"
                )

            self.log_reasoning(
                step=9,
                thought=(
                    f"Pipeline complete. MarketData assembled with "
                    f"{len(market_data.stocks)} stocks, {len(market_data.news)} news items, "
                    f"VIX={market_data.india_vix}, status={market_data.market_status.value}. "
                    f"Estimated tokens: {market_data.estimate_tokens()}. "
                    f"Pipeline status: {market_data.pipeline_status.value}."
                ),
            )

            return market_data

        except (DataFetchError, PipelineHealthError, TokenBudgetError):
            # These are expected, typed errors — let them propagate
            # with their advisor-quality messages intact.
            raise

        except Exception as e:
            # Unexpected error — wrap in DataFetchError with context
            logger.error(
                f"DataCollectorAgent encountered unexpected error: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise DataFetchError(
                source="DataCollectorAgent",
                reason=f"Unexpected error during data collection: {type(e).__name__}: {e}",
                suggestion=(
                    "Check all API credentials in .env, verify network connectivity, "
                    "and review logs for the root cause. This is not a normal failure path."
                ),
            ) from e

    def validate_output(self, output: MarketData) -> tuple[bool, list[str]]:
        """Extended validation specific to DataCollectorAgent output.

        Adds domain-specific checks on top of the base class validation:
          - MarketData.is_real_data must be True.
          - At least one stock must have an advisor_flag set.
          - India VIX must be present (> 0) during market hours.
          - advisor_morning_signal must not be empty.

        Args:
            output: The MarketData object produced by the pipeline.

        Returns:
            Tuple of (is_valid, issues). is_valid is True only if
            all checks pass (base + domain-specific).
        """
        is_valid, issues = super().validate_output(output)

        if not isinstance(output, MarketData):
            issues.append(
                "DataCollectorAgent must return a MarketData object, "
                f"got {type(output).__name__} instead."
            )
            return False, issues

        if not output.is_real_data:
            issues.append(
                "MarketData.is_real_data is False — the advisor NEVER operates on fake data. "
                "This is a critical violation of the data integrity contract."
            )

        stocks_with_signals = sum(
            1 for s in output.stocks if s.advisor_flag is not None
        )
        if output.stocks and stocks_with_signals == 0:
            issues.append(
                f"None of the {len(output.stocks)} stocks have advisor_flag set. "
                f"Bare price data without signals is useless to the advisor."
            )

        if output.market_status.value == "open" and output.india_vix <= 0:
            issues.append(
                "India VIX is 0 or negative during market hours. "
                "VIX is the market's fear gauge — the advisor needs it to assess risk."
            )

        if not output.advisor_morning_signal:
            issues.append(
                "advisor_morning_signal is empty. The advisor needs a 2-3 sentence "
                "market summary to start the conversation with the user."
            )

        is_valid = len(issues) == 0
        return is_valid, issues
