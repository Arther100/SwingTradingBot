"""
SwingAdvisorBot — Module 4: Trade Setup Generator
engine.py — Public API entry point

This is the single import other modules need:
  from module4_setup_generator.engine import setup_engine

The engine exposes 1 primary method:
  generate_setups() → Full pipeline: M1 data → screen → levels
                      → M3 risk → score → Claude → SetupPackage

Data flow:
  Caller (M6/M8 via MCP or direct)
    → engine.generate_setups(user_id, capital, ...)
    → Fetch MarketData from M1 (via MCP HTTP)
    → Fetch MarketAnalysis from M2 (via MCP HTTP)
    → TradeSetupAgent.execute()
      → StockScreener → LevelCalculator → RiskEngine
      → ConfidenceScorer → ClaudeSetup (optional)
    → SetupPackage
    ← returns to caller

Why engine.py exists:
  1. Stable public API — MCP tools and tests call this
  2. Data fetching — gets M1 + M2 data before running agent
  3. Filter construction — builds SetupFilter from params
  4. Error handling — clean errors for MCP layer
  5. Logging — structured entry/exit for every generation
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from module1_data_layer.models import MarketData
from module2_analysis_engine.models import MarketAnalysis, MarketMood
from module4_setup_generator.agents.trade_setup_agent import trade_setup_agent
from module4_setup_generator.claude_setup import get_setup_reasoning
from module4_setup_generator.models import SetupFilter, SetupPackage

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.setup_engine")

# MCP server URLs from env
M1_BASE_URL = os.getenv("M1_MCP_URL", "http://127.0.0.1:8001")
M2_BASE_URL = os.getenv("M2_MCP_URL", "http://127.0.0.1:8001")


class SetupEngine:
    """Public API for Module 4 — Trade Setup Generator.

    Usage:
        from module4_setup_generator.engine import setup_engine

        package = setup_engine.generate_setups(
            user_id="XCU700",
            display_name="Vijay",
            capital=50000.0,
        )

        # Or with MarketData/Analysis already available:
        package = setup_engine.generate_setups_from_data(
            market_data=market_data,
            analysis=analysis,
            setup_filter=setup_filter,
        )
    """

    async def generate_setups(
        self,
        user_id: str = "XCU700",
        display_name: str = "Vijay",
        capital: float = 50000.0,
        risk_tolerance: str = "moderate",
        max_setups: int = 5,
        min_confidence: float = 6.0,
        tickers: Optional[list[str]] = None,
        skip_claude: bool = False,
    ) -> SetupPackage:
        """Generate trade setups — full pipeline.

        Fetches M1 data and M2 analysis via MCP, then runs
        the TradeSetupAgent pipeline.

        Args:
            user_id: User identifier.
            display_name: User's display name.
            capital: Trading capital in INR.
            risk_tolerance: conservative / moderate / aggressive.
            max_setups: Maximum setups to generate.
            min_confidence: Minimum confidence score.
            tickers: Specific tickers to evaluate.
            skip_claude: Skip Claude reasoning calls.

        Returns:
            SetupPackage with setups and market context.

        Raises:
            RuntimeError: If M1 or M2 data cannot be fetched.
        """
        start = time.monotonic()

        logger.info(
            f"[SetupEngine] generate_setups: user={display_name}, "
            f"capital=₹{capital:,.0f}, tolerance={risk_tolerance}, "
            f"max={max_setups}, min_conf={min_confidence}, "
            f"tickers={tickers}, skip_claude={skip_claude}"
        )

        # Build setup filter
        setup_filter = SetupFilter(
            user_id=user_id,
            display_name=display_name,
            capital=capital,
            risk_tolerance=risk_tolerance,
            max_setups=max_setups,
            min_confidence=min_confidence,
            tickers=tickers,
            skip_claude=skip_claude,
        )

        # Fetch M1 data
        market_data = await self._fetch_market_data()

        # Fetch M2 analysis
        analysis = await self._fetch_analysis()

        # Run pipeline
        return self.generate_setups_from_data(
            market_data=market_data,
            analysis=analysis,
            setup_filter=setup_filter,
        )

    def generate_setups_from_data(
        self,
        market_data: MarketData,
        analysis: MarketAnalysis,
        setup_filter: Optional[SetupFilter] = None,
    ) -> SetupPackage:
        """Generate setups from pre-fetched data.

        Use this when M1/M2 data is already available
        (e.g. in tests or when called from M6 morning brief).

        Args:
            market_data: M1 MarketData with stocks.
            analysis: M2 MarketAnalysis with mood and sectors.
            setup_filter: User preferences (defaults to standard).

        Returns:
            SetupPackage with setups and market context.
        """
        start = time.monotonic()
        sf = setup_filter or SetupFilter()

        # Determine Claude reasoning function
        claude_fn = None
        if not sf.skip_claude:
            claude_fn = get_setup_reasoning

        # Run the agent
        package = trade_setup_agent.execute(
            market_data=market_data,
            analysis=analysis,
            setup_filter=sf,
            claude_reasoning_fn=claude_fn,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            f"[SetupEngine] Complete: {len(package.setups)} setups, "
            f"{len(package.skipped_setups)} skipped, "
            f"{elapsed_ms}ms"
        )

        return package

    async def _fetch_market_data(self) -> MarketData:
        """Fetch MarketData from M1 MCP server.

        Calls POST /tools/fetch_market_data on M1.

        Returns:
            MarketData from Module 1.

        Raises:
            RuntimeError: If M1 is unreachable or returns error.
        """
        url = f"{M1_BASE_URL}/tools/fetch_market_data"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.post(
                    url,
                    json={"max_stocks": 15, "max_news": 5, "token_budget": 5000},
                )
                response.raise_for_status()

            data = response.json()

            if not data.get("success"):
                raise RuntimeError(
                    f"M1 fetch failed: {data.get('error', 'Unknown error')}"
                )

            market_data_dict = data.get("data", {})
            return MarketData.model_validate(market_data_dict)

        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to M1 at {M1_BASE_URL}. "
                f"Start M1 server: python -m module1_data_layer.mcp_server"
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"M1 server timed out at {M1_BASE_URL}"
            )

    async def _fetch_analysis(self) -> MarketAnalysis:
        """Fetch MarketAnalysis from M2 MCP server.

        Calls POST /tools/analyse_market on M2.

        Returns:
            MarketAnalysis from Module 2.

        Raises:
            RuntimeError: If M2 is unreachable or returns error.
        """
        url = f"{M2_BASE_URL}/tools/analyse_market"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                response = await client.post(
                    url,
                    json={"analysis_depth": "quick"},
                )
                response.raise_for_status()

            data = response.json()

            if not data.get("success"):
                raise RuntimeError(
                    f"M2 analysis failed: {data.get('error', 'Unknown error')}"
                )

            analysis_dict = data.get("data", {}).get("analysis", {})
            return MarketAnalysis.model_validate(analysis_dict)

        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to M2 at {M2_BASE_URL}. "
                f"Start M2 server: python -m module2_analysis_engine.mcp_server"
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"M2 server timed out at {M2_BASE_URL}"
            )


# Module-level singleton
setup_engine = SetupEngine()
