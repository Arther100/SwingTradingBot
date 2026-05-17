"""
SwingAdvisorBot — Module 2: AI Analysis Engine
agents/sentiment_agent.py — News sentiment analysis via Claude

This is the SentimentAnalysisAgent — a specialised CrewAI agent
that analyses news items and sector data to determine market
sentiment. It feeds directly into MarketAnalysisAgent, providing
the "why" behind market moves.

From the CrewAI Master Plan (Section 9):
  The SentimentAnalysisAgent does NOT appear as a standalone
  crew member — it is a sub-component called by MarketAnalysisAgent
  during the full analysis pipeline. Think of it as a specialist
  consultant the main advisor calls when needed.

Data flow:
  Module 1 NewsItem[] + SectorPerformance[]
    → Claude API (via ClaudeClient)
    → JSON parsing → SentimentReport model
    → Returned to MarketAnalysisAgent for synthesis

What it produces:
  → overall_sentiment: positive/negative/mixed/neutral
  → sentiment_score: -1.0 to +1.0 (granular numeric)
  → sector_sentiments: per-sector sentiment map
  → top_risk_events: top 3 risk events for today
  → risk_level: low/normal/elevated/high
  → news_summary: 2-3 sentence plain English summary
  → cot_reasoning: reasoning trail for audit

Why a separate agent (not inline in MarketAnalysisAgent)?
  1. Token efficiency — sentiment analysis uses a separate,
     smaller prompt that fits within budget
  2. Caching — sentiment changes less frequently than prices.
     We cache it for 10 minutes (SENTIMENT_CACHE_TTL)
  3. Separation of concerns — sentiment is a distinct skill
  4. Future: can run in parallel with price analysis
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

from pydantic import Field
from zoneinfo import ZoneInfo

from module1_data_layer.agents.base_agent import SwingAdvisorBaseAgent
from module1_data_layer.models import MarketData
from module2_analysis_engine.claude_client import claude_client, ClaudeClient
from module2_analysis_engine.config import SENTIMENT_CACHE_TTL
from module2_analysis_engine.models import (
    SectorAnalysis,
    SentimentDirection,
    SentimentReport,
    MarketMood,
)
from module2_analysis_engine.prompts import (
    MASTER_SYSTEM_PROMPT,
    build_sector_analysis_prompt,
    build_sentiment_prompt,
)
from module2_analysis_engine.token_controller import token_controller

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.sentiment_agent")


class SentimentAnalysisAgent(SwingAdvisorBaseAgent):
    """Specialist agent for news sentiment and sector rotation analysis.

    Called by MarketAnalysisAgent to provide the news context layer.
    Transforms Module 1's scored news items and sector performance
    data into an advisor-quality SentimentReport.

    The agent makes two Claude API calls:
      1. Sentiment analysis — overall news sentiment + risk events
      2. Sector analysis — per-sector mood + advisor context

    Both calls are cached separately (SENTIMENT_CACHE_TTL = 10 min).

    Usage:
        agent = SentimentAnalysisAgent()
        sentiment = await agent.execute(market_data=market_data)
        # sentiment is a SentimentReport
        sectors = await agent.analyse_sectors(market_data=market_data)
        # sectors is a list[SectorAnalysis]
    """

    # ── Agent Identity ──
    agent_name: str = Field(
        default="SentimentAnalysisAgent",
        description="Specialist sentiment sub-agent within Module 2.",
    )

    # ── CrewAI Personality Fields ──
    role: str = Field(
        default="News Sentiment Analyst",
        description="CrewAI role — specialist in news interpretation.",
    )
    goal: str = Field(
        default=(
            "Analyse news items and sector data to determine market "
            "sentiment for Indian NSE markets. Identify risk events, "
            "sector rotation patterns, and provide the news context "
            "that the senior advisor needs to make informed calls."
        ),
        description="CrewAI goal — what this agent achieves.",
    )
    backstory: str = Field(
        default=(
            "You are a specialist news analyst supporting a senior "
            "finance advisor. You read every news item critically — "
            "not for what it says, but for what it means for the "
            "market. You identify risk events early, detect sector "
            "rotation from news flow, and provide the context that "
            "turns raw headlines into trading intelligence. You are "
            "sceptical of sensational headlines and focus on facts "
            "that move markets."
        ),
        description="CrewAI backstory — the agent's specialist personality.",
    )

    # ── Injected Dependencies ──
    _claude: ClaudeClient = claude_client

    model_config = {"arbitrary_types_allowed": True}

    async def execute(
        self,
        market_data: MarketData,
    ) -> SentimentReport:
        """Analyse news sentiment from market data.

        Takes the news items and sector data from MarketData and
        produces a SentimentReport via Claude API call.

        Steps:
          1. Extract and serialize news items
          2. Extract and serialize sector data
          3. Call Claude for sentiment analysis
          4. Parse response into SentimentReport
          5. Log reasoning and return

        Args:
            market_data: MarketData with news and sector data.

        Returns:
            SentimentReport with sentiment assessment.
        """
        self.reset_reasoning()
        start_time = time.monotonic()

        # ── Step 1: Extract news items ──
        news_items = market_data.news
        self.log_reasoning(1, (
            f"Extracting news: {len(news_items)} items available. "
            f"Serializing for Claude API call."
        ))

        if not news_items:
            self.log_reasoning(1, "No news items available. Returning neutral sentiment.")
            return SentimentReport(
                overall_sentiment=SentimentDirection.NEUTRAL,
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                news_summary="No news items available for analysis.",
                cot_reasoning="No news data provided — defaulting to neutral sentiment.",
                news_items_analysed=0,
            )

        # Serialize news items (top items by relevance, up to 10)
        sorted_news = sorted(news_items, key=lambda n: n.relevance_score, reverse=True)
        top_news = sorted_news[:10]
        news_json = json.dumps(
            [
                {
                    "headline": item.headline,
                    "source": item.source,
                    "sentiment": item.sentiment.value if item.sentiment else "neutral",
                    "market_impact": item.market_impact.value if item.market_impact else "low",
                    "affected_sectors": item.affected_sectors,
                    "relevance_score": item.relevance_score,
                }
                for item in top_news
            ],
            indent=2,
        )

        # ── Step 2: Extract sector data ──
        sectors = market_data.sectors
        self.log_reasoning(2, f"Extracting sectors: {len(sectors)} sectors available.")

        sector_json = json.dumps(
            [
                {
                    "sector_name": s.sector_name,
                    "change_pct": s.change_pct,
                    "top_gainer": s.top_gainer,
                    "top_gainer_change_pct": s.top_gainer_change_pct,
                    "top_loser": s.top_loser,
                    "top_loser_change_pct": s.top_loser_change_pct,
                    "sector_signal": s.sector_signal.value if s.sector_signal else "neutral",
                }
                for s in sectors
            ],
            indent=2,
        )

        # ── Step 3: Call Claude for sentiment ──
        self.log_reasoning(3, "Calling Claude API for sentiment analysis.")

        user_message = build_sentiment_prompt(
            news_items_json=news_json,
            sector_data_json=sector_json,
        )

        cache_key = ClaudeClient.generate_cache_key(
            market_data_timestamp=market_data.timestamp.isoformat(),
            user_id="sentiment",
            analysis_type="sentiment",
        )

        response_dict = await self._claude.call_claude(
            system_prompt=MASTER_SYSTEM_PROMPT,
            user_message=user_message,
            cache_key=cache_key,
            cache_ttl=SENTIMENT_CACHE_TTL,
        )

        # ── Step 4: Parse into SentimentReport ──
        self.log_reasoning(4, "Parsing Claude response into SentimentReport.")
        report = self._parse_sentiment_report(response_dict, len(top_news))

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        self.log_reasoning(5, (
            f"Sentiment analysis complete: {report.overall_sentiment.value} "
            f"(score: {report.sentiment_score:+.2f}, "
            f"confidence: {report.sentiment_confidence:.2f}). "
            f"Risk level: {report.risk_level}. "
            f"Risk events: {len(report.top_risk_events)}. "
            f"Latency: {elapsed_ms}ms."
        ))

        return report

    async def analyse_sectors(
        self,
        market_data: MarketData,
    ) -> list[SectorAnalysis]:
        """Analyse sector rotation and produce per-sector advisor context.

        Takes sector performance data from MarketData and produces
        a list of SectorAnalysis models with advisor interpretation.

        Args:
            market_data: MarketData with sector and VIX data.

        Returns:
            List of SectorAnalysis with advisor context per sector.
        """
        sectors = market_data.sectors
        if not sectors:
            logger.info("No sector data available. Skipping sector analysis.")
            return []

        self.log_reasoning(1, (
            f"Sector analysis: {len(sectors)} sectors. "
            f"VIX: {market_data.india_vix:.2f} ({market_data.vix_signal}). "
            f"Nifty: {market_data.nifty50_change_pct:+.2f}%."
        ))

        sector_json = json.dumps(
            [
                {
                    "sector_name": s.sector_name,
                    "change_pct": s.change_pct,
                    "top_gainer": s.top_gainer,
                    "top_gainer_change_pct": s.top_gainer_change_pct,
                    "top_loser": s.top_loser,
                    "top_loser_change_pct": s.top_loser_change_pct,
                    "sector_signal": s.sector_signal.value if s.sector_signal else "neutral",
                    "advisor_note": s.advisor_note,
                }
                for s in sectors
            ],
            indent=2,
        )

        user_message = build_sector_analysis_prompt(
            sector_data_json=sector_json,
            vix_value=market_data.india_vix,
            vix_signal=market_data.vix_signal,
            nifty_change_pct=market_data.nifty50_change_pct,
        )

        cache_key = ClaudeClient.generate_cache_key(
            market_data_timestamp=market_data.timestamp.isoformat(),
            user_id="sectors",
            analysis_type="sector",
        )

        response_dict = await self._claude.call_claude(
            system_prompt=MASTER_SYSTEM_PROMPT,
            user_message=user_message,
            cache_key=cache_key,
            cache_ttl=SENTIMENT_CACHE_TTL,
        )

        self.log_reasoning(2, "Parsing sector analysis response.")

        # Response could be a list (array) or dict with a sectors key
        sector_list = self._extract_sector_list(response_dict)
        analyses = [
            self._parse_sector_analysis(entry)
            for entry in sector_list
        ]

        self.log_reasoning(3, (
            f"Sector analysis complete: {len(analyses)} sectors analysed. "
            f"Moods: {[a.sector_mood.value for a in analyses]}."
        ))

        return analyses

    def _parse_sentiment_report(
        self,
        response_dict: dict,
        items_analysed: int,
    ) -> SentimentReport:
        """Parse Claude's sentiment response into SentimentReport.

        Maps Claude's JSON fields to the SentimentReport Pydantic model.
        Handles missing fields gracefully with neutral defaults.

        Args:
            response_dict: Parsed JSON dict from Claude.
            items_analysed: Number of news items that were sent.

        Returns:
            SentimentReport model instance.
        """
        # Map sentiment direction
        sentiment_str = response_dict.get("overall_sentiment", "neutral")
        try:
            overall_sentiment = SentimentDirection(sentiment_str)
        except ValueError:
            logger.warning(
                f"Unknown sentiment '{sentiment_str}' from Claude. Defaulting to neutral."
            )
            overall_sentiment = SentimentDirection.NEUTRAL

        # Clamp score to valid range
        raw_score = float(response_dict.get("sentiment_score", 0.0))
        sentiment_score = max(-1.0, min(1.0, raw_score))

        raw_confidence = float(response_dict.get("sentiment_confidence", 0.5))
        sentiment_confidence = max(0.0, min(1.0, raw_confidence))

        # Sector sentiments — dict of sector name to score
        sector_sentiments = {}
        raw_sectors = response_dict.get("sector_sentiments", {})
        if isinstance(raw_sectors, dict):
            for sector_name, score in raw_sectors.items():
                try:
                    sector_sentiments[sector_name] = max(-1.0, min(1.0, float(score)))
                except (ValueError, TypeError):
                    sector_sentiments[sector_name] = 0.0

        # Risk events
        top_risk_events = response_dict.get("top_risk_events", [])
        if not isinstance(top_risk_events, list):
            top_risk_events = []
        top_risk_events = [str(e) for e in top_risk_events[:5]]

        # Risk level
        risk_level = response_dict.get("risk_level", "normal")
        if risk_level not in ("low", "normal", "elevated", "high"):
            risk_level = "normal"

        return SentimentReport(
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            sector_sentiments=sector_sentiments,
            top_risk_events=top_risk_events,
            risk_level=risk_level,
            news_summary=response_dict.get("news_summary", ""),
            cot_reasoning=response_dict.get("cot_reasoning", ""),
            news_items_analysed=items_analysed,
        )

    def _extract_sector_list(self, response_dict: dict) -> list[dict]:
        """Extract sector analysis entries from Claude's response.

        Claude may return:
          → A JSON array directly (if it follows instructions)
          → A dict with a "sectors" or "sector_analyses" key
          → A dict where each key is a sector name

        This method normalises all formats into a list of dicts.

        Args:
            response_dict: Parsed JSON from Claude.

        Returns:
            List of sector entry dicts.
        """
        # If response itself is a list (shouldn't happen — claude_client
        # forces dict, but handle defensively)
        if isinstance(response_dict, list):
            return [e for e in response_dict if isinstance(e, dict)]

        # Check for common wrapper keys
        for key in ("sectors", "sector_analyses", "data", "results"):
            if key in response_dict and isinstance(response_dict[key], list):
                return [e for e in response_dict[key] if isinstance(e, dict)]

        # If response is a flat dict with sector data, wrap it
        if "sector_name" in response_dict:
            return [response_dict]

        # Try to find any list value in the dict
        for value in response_dict.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value

        logger.warning(
            "Could not extract sector list from Claude response. "
            f"Keys: {list(response_dict.keys())}. Returning empty list."
        )
        return []

    def _parse_sector_analysis(self, entry: dict) -> SectorAnalysis:
        """Parse a single sector entry into SectorAnalysis model.

        Args:
            entry: Dict with sector analysis fields from Claude.

        Returns:
            SectorAnalysis model instance.
        """
        # Map sector mood
        mood_str = entry.get("sector_mood", "neutral")
        try:
            sector_mood = MarketMood(mood_str)
        except ValueError:
            sector_mood = MarketMood.NEUTRAL

        return SectorAnalysis(
            sector_name=entry.get("sector_name", "Unknown"),
            sector_mood=sector_mood,
            change_pct=float(entry.get("change_pct", 0.0)),
            situation=entry.get("situation", ""),
            reasoning=entry.get("reasoning", ""),
            advisor_action=entry.get("advisor_action", ""),
            top_opportunity=entry.get("top_opportunity", ""),
            top_risk=entry.get("top_risk", ""),
        )

    def validate_output(self, output: SentimentReport) -> tuple[bool, list[str]]:
        """Validate the SentimentReport before returning.

        Extends base agent validation with sentiment-specific checks:
          → Sentiment direction is set
          → News summary is non-empty if items were analysed
          → Score is within valid range

        Args:
            output: SentimentReport to validate.

        Returns:
            Tuple of (is_valid, issues).
        """
        is_valid, issues = super().validate_output(output)

        if not isinstance(output, SentimentReport):
            issues.append(
                "SentimentAnalysisAgent: Output is not a SentimentReport."
            )
            return False, issues

        if output.news_items_analysed > 0 and not output.news_summary:
            issues.append(
                "SentimentAnalysisAgent: Analysed news items but "
                "news_summary is empty. The advisor needs a summary."
            )

        if output.sentiment_score < -1.0 or output.sentiment_score > 1.0:
            issues.append(
                f"SentimentAnalysisAgent: sentiment_score {output.sentiment_score} "
                f"is out of range [-1.0, 1.0]."
            )

        return len(issues) == 0, issues


# Module-level singleton — used by MarketAnalysisAgent and analysis_crew
sentiment_agent = SentimentAnalysisAgent()
