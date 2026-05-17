"""
SwingAdvisorBot — Module 2: AI Analysis Engine
agents/market_analysis_agent.py — The senior advisor's market analysis brain

This is Agent 2 from the CrewAI Master Plan (Section 9):

  Agent 2: MarketAnalysisAgent (Module 2)
    Role: Analyse market data as senior advisor
    Tools: DataCollectorAgent output, Memory context
    Output: MarketAnalysis with signals and context

MarketAnalysisAgent is the central intelligence of the entire system.
It takes raw MarketData from Module 1 and transforms it into an
advisor-quality MarketAnalysis — complete with situation assessment,
reasoning, personalised advice, risk warnings, and a lesson.

Data flow:
  Module 1 MarketData
    → TokenController (budget enforcement + trimming)
    → Claude API (via ClaudeClient)
    → JSON parsing → MarketAnalysis model
    → QualityChecker (self-reflection)
    → HallucinationGuard (fact verification)
    → AnalysisResult (with metadata)

The agent orchestrates the full analysis pipeline:
  Step 1: Validate input data (enough stocks? real data?)
  Step 2: Prepare token-budgeted input via TokenController
  Step 3: Build prompt via prompt builders
  Step 4: Call Claude API via ClaudeClient
  Step 5: Parse response into MarketAnalysis
  Step 6: Run quality gate — retry if FAILED
  Step 7: Run hallucination guard — retry if should_retry
  Step 8: Package into AnalysisResult with metadata

Retry strategy:
  → Quality failure → retry with QUALITY_REMINDER (up to MAX_RETRIES)
  → Hallucination → retry with grounding feedback (up to MAX_RETRIES)
  → Parse failure → handled by ClaudeClient internally
  → Connection → handled by ClaudeClient internally
  → After all retries exhausted → raise FinalAnalysisError
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime

from pydantic import Field
from zoneinfo import ZoneInfo

from module1_data_layer.agents.base_agent import SwingAdvisorBaseAgent
from module1_data_layer.models import MarketData
from module2_analysis_engine.claude_client import claude_client, ClaudeClient
from module2_analysis_engine.config import (
    ANALYSIS_CACHE_TTL,
    HARD_TOKEN_LIMIT,
    MAX_RETRIES,
    MOOD_CACHE_TTL,
    OUTPUT_TOKEN_LIMIT,
    QUALITY_RETRY_BACKOFF,
)
from module2_analysis_engine.hallucination_guard import (
    hallucination_guard,
    HallucinationGuard,
)
from module2_analysis_engine.models import (
    AnalysisDepth,
    AnalysisQualityError,
    AnalysisResult,
    AnalysisQualityReport,
    FinalAnalysisError,
    InsufficientDataError,
    MarketAnalysis,
    MarketMood,
    QualityVerdict,
    UserContext,
)
from module2_analysis_engine.prompts import (
    COT_INSTRUCTION,
    MASTER_SYSTEM_PROMPT,
    build_market_analysis_prompt,
    build_quality_retry_prompt,
    build_quick_mood_prompt,
)
from module2_analysis_engine.quality_checker import quality_checker, QualityChecker
from module2_analysis_engine.token_controller import token_controller, TokenController

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.market_analysis_agent")

# ── Measured token budgets (from live testing) ──
# System prompt (548) + CoT instruction (353) + overhead (100) = 1001
_FIXED_PROMPT_TOKENS = 1001
_OUTPUT_BUDGET = OUTPUT_TOKEN_LIMIT        # 1500
_TOTAL_LIMIT = HARD_TOKEN_LIMIT            # 3000
# 3000 - 1001 - 1500 = 499 tokens for market data
_DATA_BUDGET = _TOTAL_LIMIT - _FIXED_PROMPT_TOKENS - _OUTPUT_BUDGET

_SIGNAL_PRIORITY = [
    "breakout_watch",
    "unusual_activity",
    "momentum_building",
    "selling_pressure",
    "accumulation_zone",
    "distribution_zone",
    "consolidation",
    "neutral",
]


class MarketAnalysisAgent(SwingAdvisorBaseAgent):
    """CrewAI agent that produces advisor-quality market analysis.

    This is the brain of SwingAdvisorBot. It takes Module 1's
    MarketData and produces a complete MarketAnalysis with:
      → Market mood assessment (bullish → extreme_fear)
      → Situation description (what is happening)
      → Reasoning (why it is happening)
      → User-specific impact (personalised advice)
      → Action recommendation (with price levels)
      → Risk assessment (what could go wrong)
      → Lesson of the day (educational content)
      → CoT reasoning trail (full audit trail)
      → Top opportunities and risks (for Module 3/4)

    Every analysis passes through:
      1. TokenController — budget enforcement
      2. ClaudeClient — API call with retries
      3. QualityChecker — self-reflection gate
      4. HallucinationGuard — fact verification

    Usage:
        agent = MarketAnalysisAgent()
        result = await agent.execute(
            market_data=market_data,
            user_context=user_context,
        )
        # result is AnalysisResult with analysis + metadata
    """

    # ── Agent Identity ──
    agent_name: str = Field(
        default="MarketAnalysisAgent",
        description="Agent 2 from the CrewAI Master Plan.",
    )

    # ── CrewAI Personality Fields ──
    role: str = Field(
        default="Senior Market Analysis Advisor",
        description="CrewAI role — this agent IS the senior advisor.",
    )
    goal: str = Field(
        default=(
            "Analyse NSE market data and produce advisor-quality analysis "
            "with situation assessment, reasoning, personalised advice, "
            "risk warnings, and educational content. Never give just data — "
            "always give context, signal, and actionable advice."
        ),
        description="CrewAI goal — what this agent achieves.",
    )
    backstory: str = Field(
        default=(
            "You are the core intelligence of SwingAdvisorBot — a system "
            "that embodies a senior finance advisor with 20+ years of "
            "experience in Indian capital markets. You transform raw market "
            "data into personalised, advisor-quality analysis. You never "
            "give bare data. You always explain what is happening, why it "
            "is happening, what it means for the user, and what to do about "
            "it. You are conservative with risk, honest about uncertainty, "
            "and you teach while you advise."
        ),
        description="CrewAI backstory — the agent's personality and principles.",
    )

    # ── Injected Dependencies (allow override for testing) ──
    _claude: ClaudeClient = claude_client
    _quality: QualityChecker = quality_checker
    _hallucination: HallucinationGuard = hallucination_guard
    _tokens: TokenController = token_controller

    model_config = {"arbitrary_types_allowed": True}

    async def execute(
        self,
        market_data: MarketData,
        user_context: UserContext | None = None,
        analysis_depth: AnalysisDepth = AnalysisDepth.FULL,
        user_message: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> AnalysisResult:
        """Execute the full market analysis pipeline.

        This is the main entry point. It orchestrates:
          Step 1: Validate input (data quality, minimum stocks)
          Step 2: Prepare token-budgeted input
          Step 3: Build prompt
          Step 4: Call Claude API
          Step 5: Parse response into MarketAnalysis
          Step 6: Quality gate (retry if failed)
          Step 7: Hallucination guard (retry if fabricated)
          Step 8: Package into AnalysisResult

        Args:
            market_data: MarketData from Module 1 pipeline.
            user_context: UserContext for personalisation (None if unavailable).
            analysis_depth: FULL or QUICK analysis.

        Returns:
            AnalysisResult with analysis, quality report, and metadata.

        Raises:
            InsufficientDataError: If MarketData has too few stocks.
            FinalAnalysisError: If all retries are exhausted.
        """
        self.reset_reasoning()
        start_time = time.monotonic()

        # Default user context if none provided
        if user_context is None:
            user_context = UserContext(user_id="default")

        # Route to appropriate analysis method
        if analysis_depth == AnalysisDepth.QUICK:
            return await self._quick_mood_analysis(
                market_data=market_data,
                start_time=start_time,
                user_message=user_message,
                conversation_history=conversation_history,
            )

        return await self._full_analysis(
            market_data=market_data,
            user_context=user_context,
            start_time=start_time,
        )

    async def _full_analysis(
        self,
        market_data: MarketData,
        user_context: UserContext,
        start_time: float,
    ) -> AnalysisResult:
        """Execute full advisor-quality market analysis.

        The complete 8-step pipeline with quality gate and
        hallucination guard. This is the gold standard output.

        Args:
            market_data: Validated MarketData.
            user_context: UserContext for personalisation.
            start_time: Monotonic start time for latency tracking.

        Returns:
            AnalysisResult with full analysis.

        Raises:
            InsufficientDataError: Too few stocks.
            FinalAnalysisError: All retries exhausted.
        """
        # ── Step 1: Validate input ──
        self.log_reasoning(1, (
            f"Validating input: {len(market_data.stocks)} stocks, "
            f"Nifty {market_data.nifty50_change_pct:+.2f}%, "
            f"VIX {market_data.india_vix:.2f}, "
            f"is_real_data={market_data.is_real_data}."
        ))

        if len(market_data.stocks) < 3:
            raise InsufficientDataError(
                stocks_available=len(market_data.stocks),
                stocks_required=3,
            )

        # ── Step 2: Trim and validate token budget ──
        self.log_reasoning(2, "Trimming market data and validating token budget.")

        market_data = self._trim_market_data(market_data)
        market_data_json = market_data.model_dump_json(
            by_alias=True, exclude_none=True, exclude_defaults=True,
        )
        user_context_json = user_context.model_dump_json(exclude_none=True)
        input_tokens = self._validate_budget(
            system=MASTER_SYSTEM_PROMPT,
            cot=COT_INSTRUCTION,
            market_data_json=market_data_json,
            user_context_json=user_context_json,
        )

        self.log_reasoning(2, (
            f"Token budget: {input_tokens} input tokens. "
            f"Market data trimmed to fit budget."
        ))

        # ── Step 3: Build prompt ──
        user_message = build_market_analysis_prompt(
            market_data_json=market_data_json,
            user_context_json=user_context_json,
            include_cot=True,
        )

        self.log_reasoning(3, "Prompt built with market data, user context, and CoT instruction.")

        # ── Step 4-7: Call Claude + Quality + Hallucination (with retries) ──
        cache_key = ClaudeClient.generate_cache_key(
            market_data_timestamp=market_data.timestamp.isoformat(),
            user_id=user_context.user_id,
            analysis_type="full",
        )

        analysis = None
        quality_report = None
        retry_count = 0
        from_cache = False

        for attempt in range(MAX_RETRIES + 1):
            # ── Step 4: Call Claude API ──
            self.log_reasoning(4, f"Calling Claude API (attempt {attempt + 1}).")

            response_dict = await self._claude.call_claude(
                system_prompt=MASTER_SYSTEM_PROMPT,
                user_message=user_message,
                cache_key=cache_key if attempt == 0 else None,
                cache_ttl=ANALYSIS_CACHE_TTL,
            )

            # ── Step 5: Parse into MarketAnalysis ──
            self.log_reasoning(5, "Parsing Claude response into MarketAnalysis model.")
            analysis = self._parse_market_analysis(response_dict)

            # ── Step 6: Quality gate ──
            self.log_reasoning(6, "Running quality gate self-reflection.")
            quality_report = self._quality.check(analysis, user_context)

            if quality_report.verdict == QualityVerdict.FAILED:
                retry_count = attempt + 1
                if attempt < MAX_RETRIES:
                    issues_text = self._quality.format_issues_for_retry(quality_report)
                    user_message = build_quality_retry_prompt(
                        original_prompt=user_message,
                        quality_issues=issues_text,
                    )
                    self.log_reasoning(6, (
                        f"Quality gate FAILED (attempt {attempt + 1}): "
                        f"{quality_report.issues[:2]}. Retrying."
                    ))
                    await asyncio.sleep(QUALITY_RETRY_BACKOFF)
                    continue
                else:
                    self.log_reasoning(6, (
                        f"Quality gate FAILED after all retries. "
                        f"Issues: {quality_report.issues[:3]}. "
                        f"Proceeding with best available analysis."
                    ))
                    logger.warning(
                        f"MarketAnalysisAgent: Quality gate failed after "
                        f"{MAX_RETRIES + 1} attempts. Returning best-effort analysis."
                    )

            # ── Step 7: Hallucination guard ──
            self.log_reasoning(7, "Running hallucination guard against MarketData.")
            hallucination_report = self._hallucination.verify(
                analysis_dict=response_dict,
                market_data=market_data,
            )

            if hallucination_report.should_retry and attempt < MAX_RETRIES:
                retry_count = attempt + 1
                grounding_feedback = self._hallucination.format_grounding_feedback(
                    hallucination_report
                )
                user_message = f"{grounding_feedback}\n\n{user_message}"
                self.log_reasoning(7, (
                    f"Hallucination guard triggered retry: "
                    f"{len(hallucination_report.hallucinated_tickers)} fake tickers, "
                    f"{len(hallucination_report.price_mismatches)} price mismatches."
                ))
                await asyncio.sleep(QUALITY_RETRY_BACKOFF)
                continue

            # All checks passed (or warnings only)
            break

        if analysis is None:
            raise FinalAnalysisError(
                attempts=MAX_RETRIES + 1,
                last_error="Analysis pipeline produced no result after all attempts.",
            )

        # ── Step 8: Package into AnalysisResult ──
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        output_tokens = self._tokens.estimate(analysis.model_dump_json())
        budget_summary = self._tokens.get_budget_summary(input_tokens, output_tokens)

        result = AnalysisResult(
            analysis=analysis,
            quality_report=quality_report,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model_used=self._claude._settings.claude_model,
            total_latency_ms=elapsed_ms,
            from_cache=from_cache,
            retry_count=retry_count,
        )

        self.log_reasoning(8, (
            f"Analysis complete. Mood: {analysis.market_mood.value} "
            f"(confidence: {analysis.mood_confidence:.2f}). "
            f"Quality: {quality_report.verdict.value if quality_report else 'N/A'}. "
            f"Tokens: {input_tokens} in + {output_tokens} out = "
            f"{input_tokens + output_tokens} total. "
            f"Latency: {elapsed_ms}ms. Retries: {retry_count}."
        ))

        # Validate output via base agent
        is_valid, issues = self.validate_output(result)
        if not is_valid:
            logger.warning(
                f"MarketAnalysisAgent output validation issues: {issues}"
            )

        return result

    def _trim_market_data(self, market_data: MarketData) -> MarketData:
        """Full 8-step trim — matches token_controller.py logic.

        Trims MarketData to fit within _DATA_BUDGET.
        Never raises — always returns best-effort data.

        Steps:
          1. Remove cot_reasoning from stocks
          2. Remove advisor_note from stocks (if present)
          3. Keep top 4 stocks by signal priority
          4. Strip OHLC detail (set to defaults)
          5. Keep top 3 news, remove news CoT/notes
          6. Remove economic events
          7. Remove sector performance
          8. Emergency — keep only 2 stocks
        """
        from copy import deepcopy

        data = deepcopy(market_data)
        budget = _DATA_BUDGET

        def _estimate(d: MarketData) -> int:
            return self._tokens.estimate(
                d.model_dump_json(exclude_none=True, exclude_defaults=True)
            )

        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Starting. "
            f"Current: {current} tokens. Budget: {budget} tokens."
        )

        if current <= budget:
            logger.info("[AgentTrim] Within budget. No trim.")
            return data

        # Step 1: Remove cot_reasoning from stocks
        for stock in data.stocks:
            stock.cot_reasoning = None
        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Step 1 — removed stock CoT. Tokens: {current}"
        )
        if current <= budget:
            return data

        # Step 2: Remove advisor_note from stocks (if field exists)
        for stock in data.stocks:
            if hasattr(stock, "advisor_note"):
                stock.advisor_note = None
        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Step 2 — removed stock notes. Tokens: {current}"
        )
        if current <= budget:
            return data

        # Step 3: Keep top 4 stocks by signal priority
        data.stocks = sorted(
            data.stocks,
            key=lambda s: (
                _SIGNAL_PRIORITY.index(s.advisor_flag)
                if s.advisor_flag in _SIGNAL_PRIORITY
                else 99
            ),
        )[:4]
        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Step 3 — top 4 stocks. Tokens: {current}"
        )
        if current <= budget:
            return data

        # Step 4: Strip OHLC detail (set to defaults, excluded by exclude_defaults)
        for stock in data.stocks:
            stock.open = 0.0
            stock.high = 0.0
            stock.low = 0.0
            stock.volume = 0
            stock.avg_volume_30d = 0
        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Step 4 — stripped OHLC. Tokens: {current}"
        )
        if current <= budget:
            return data

        # Step 5: Keep top 3 news by relevance, strip CoT/notes
        if data.news:
            data.news = sorted(
                data.news,
                key=lambda n: getattr(n, "relevance_score", 0),
                reverse=True,
            )[:3]
            for news in data.news:
                if hasattr(news, "cot_reasoning"):
                    news.cot_reasoning = None
                if hasattr(news, "advisor_note"):
                    news.advisor_note = None
        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Step 5 — top 3 news. Tokens: {current}"
        )
        if current <= budget:
            return data

        # Step 6: Remove economic events
        data.economic_events = []
        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Step 6 — removed events. Tokens: {current}"
        )
        if current <= budget:
            return data

        # Step 7: Remove sector performance
        data.sectors = []
        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Step 7 — removed sectors. Tokens: {current}"
        )
        if current <= budget:
            return data

        # Step 8: Emergency — keep only 2 stocks, 2 news
        data.stocks = data.stocks[:2]
        data.news = data.news[:2] if data.news else []
        current = _estimate(data)
        logger.info(
            f"[AgentTrim] Step 8 — emergency 2 stocks. Tokens: {current}"
        )
        if current <= budget:
            return data

        # Never block — warn and proceed
        logger.warning(
            f"[AgentTrim] Could not trim to budget. "
            f"Proceeding with {current} tokens "
            f"(budget: {budget}). "
            f"Overage: {current - budget} tokens."
        )
        return data

    def _validate_budget(
        self,
        system: str,
        cot: str,
        market_data_json: str,
        user_context_json: str,
        task: str = "",
    ) -> int:
        """Validate total input token count.

        Never raises — always proceeds with best effort.
        Returns estimated total input tokens.
        """
        system_tokens = self._tokens.estimate(system)
        cot_tokens = self._tokens.estimate(cot)
        data_tokens = self._tokens.estimate(market_data_json)
        user_tokens = self._tokens.estimate(user_context_json)
        task_tokens = self._tokens.estimate(task) if task else 0
        overhead = 100

        total = (
            system_tokens + cot_tokens + data_tokens
            + user_tokens + task_tokens + overhead
        )

        logger.info(
            f"[AgentBudget] Final token breakdown — "
            f"system={system_tokens}, cot={cot_tokens}, "
            f"data={data_tokens}, user={user_tokens}, "
            f"task={task_tokens}, overhead={overhead}. "
            f"Total={total}/{_TOTAL_LIMIT}"
        )

        if total > _TOTAL_LIMIT:
            logger.warning(
                f"[AgentBudget] Total {total} exceeds "
                f"{_TOTAL_LIMIT}. Proceeding anyway — "
                f"Claude handles up to 200k context."
            )

        return total

    async def _quick_mood_analysis(
        self,
        market_data: MarketData,
        start_time: float,
        user_message: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> AnalysisResult:
        """Execute a quick mood check — lightweight, no full quality gate.

        Used for fast market status checks before deciding whether
        to run a full analysis. Costs ~1200 tokens vs ~2400 for full.

        Args:
            market_data: MarketData from Module 1.
            start_time: Monotonic start time for latency tracking.
            user_message: Optional user chat question.
            conversation_history: Optional recent conversation.

        Returns:
            AnalysisResult with quick-depth analysis.
        """
        self.log_reasoning(1, (
            f"Quick mood check: {len(market_data.stocks)} stocks, "
            f"VIX {market_data.india_vix:.2f}."
        ))

        # Serialize market data (minimal — no token controller needed for quick)
        market_data_json = market_data.model_dump_json(
            by_alias=True,
            exclude_none=True,
            exclude_defaults=True,
        )
        input_tokens = self._tokens.estimate(market_data_json + MASTER_SYSTEM_PROMPT)

        # Build quick mood prompt
        prompt = build_quick_mood_prompt(
            market_data_json,
            user_message=user_message,
            conversation_history=conversation_history,
        )

        # Disable cache when chat message is present (each question is unique)
        cache_key = (
            None if user_message else
            ClaudeClient.generate_cache_key(
                market_data_timestamp=market_data.timestamp.isoformat(),
                user_id="quick",
                analysis_type="quick",
            )
        )

        self.log_reasoning(2, "Calling Claude API for quick mood assessment.")

        response_dict = await self._claude.call_claude(
            system_prompt=MASTER_SYSTEM_PROMPT,
            user_message=prompt,
            cache_key=cache_key,
            cache_ttl=MOOD_CACHE_TTL,
        )

        self.log_reasoning(3, "Parsing quick mood response.")
        analysis = self._parse_market_analysis(response_dict, is_quick=True)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        output_tokens = self._tokens.estimate(analysis.model_dump_json())

        result = AnalysisResult(
            analysis=analysis,
            quality_report=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model_used=self._claude._settings.claude_model,
            total_latency_ms=elapsed_ms,
            from_cache=False,
            retry_count=0,
        )

        self.log_reasoning(4, (
            f"Quick mood: {analysis.market_mood.value} "
            f"(confidence: {analysis.mood_confidence:.2f}). "
            f"Latency: {elapsed_ms}ms."
        ))

        return result

    def _parse_market_analysis(
        self,
        response_dict: dict,
        is_quick: bool = False,
    ) -> MarketAnalysis:
        """Parse Claude's JSON response into a MarketAnalysis model.

        Maps Claude's response fields to the MarketAnalysis Pydantic
        model. Handles missing fields gracefully with defaults.

        Args:
            response_dict: Parsed JSON dict from Claude.
            is_quick: If True, use QUICK analysis depth (relaxed requirements).

        Returns:
            MarketAnalysis model instance.
        """
        # Map market_mood string to enum
        mood_str = response_dict.get("market_mood", "neutral")
        try:
            market_mood = MarketMood(mood_str)
        except ValueError:
            logger.warning(
                f"Unknown market_mood '{mood_str}' from Claude. Defaulting to neutral."
            )
            market_mood = MarketMood.NEUTRAL

        return MarketAnalysis(
            market_mood=market_mood,
            mood_confidence=float(response_dict.get("mood_confidence", 0.5)),
            situation=response_dict.get("situation", ""),
            reasoning=response_dict.get("reasoning", ""),
            user_impact=response_dict.get("user_impact", ""),
            action=response_dict.get("action", ""),
            risk=response_dict.get("risk", ""),
            lesson=response_dict.get("lesson", ""),
            cot_reasoning=response_dict.get("cot_reasoning", ""),
            top_opportunities=response_dict.get("top_opportunities", []),
            top_risks=response_dict.get("top_risks", []),
            risk_events=response_dict.get("risk_events", []),
            analysis_depth=AnalysisDepth.QUICK if is_quick else AnalysisDepth.FULL,
        )

    def validate_output(self, output: AnalysisResult) -> tuple[bool, list[str]]:
        """Validate the AnalysisResult before returning.

        Extends base agent validation with analysis-specific checks:
          → Analysis object is present and populated
          → Market mood is set (not None)
          → Situation and reasoning are non-empty
          → For full depth: lesson is present

        Args:
            output: AnalysisResult to validate.

        Returns:
            Tuple of (is_valid, issues).
        """
        is_valid, issues = super().validate_output(output)

        if not hasattr(output, "analysis") or output.analysis is None:
            issues.append(
                "MarketAnalysisAgent: AnalysisResult has no analysis object."
            )
        else:
            analysis = output.analysis
            if not analysis.situation:
                issues.append(
                    "MarketAnalysisAgent: situation field is empty. "
                    "The advisor must describe what is happening."
                )
            if not analysis.reasoning:
                issues.append(
                    "MarketAnalysisAgent: reasoning field is empty. "
                    "The advisor must explain why."
                )
            if analysis.analysis_depth == AnalysisDepth.FULL and not analysis.lesson:
                issues.append(
                    "MarketAnalysisAgent: lesson field is empty for full analysis. "
                    "Every full analysis must teach something."
                )

        return len(issues) == 0, issues


# Module-level singleton — used by analysis_crew.py
market_analysis_agent = MarketAnalysisAgent()
