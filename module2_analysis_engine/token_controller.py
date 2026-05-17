"""
SwingAdvisorBot — Module 2: AI Analysis Engine
token_controller.py — Token budget enforcement for Claude API calls

Every Claude API call costs money. Every token earns its place.
This controller ensures we never exceed the 3000 token hard limit
per call, and that we trim intelligently when data is too large.

Token budget per call:
  System prompt (personality):  ~380 tokens (NEVER trimmed)
  CoT instruction:              ~180 tokens (NEVER trimmed)
  User memory (M5):             ~300 tokens (keep if possible)
  ─────────────────────────────────────────────────────────
  Fixed prompt tokens:           860 tokens (system + CoT + memory)
  Market data (M1):             ~740 tokens (trimmable — priority order)
  Output budget:                1500 tokens (Claude response)
  ─────────────────────────────────────────────────────────
  Total per call:               3100 tokens (absolute max)

  Budget math: 3100 - 860(fixed) - 1500(output) = 740 for MarketData

Trimming priority (what gets cut first → last):
  Priority 1: system_prompt      → NEVER trim (advisor personality)
  Priority 2: cot_instruction    → NEVER trim (reasoning quality)
  Priority 3: user_memory        → Keep (critical for personalisation)
  Priority 4: market_data stocks → Trim lowest-signal stocks first
  Priority 5: news               → Keep top 3 by relevance only
  Priority 6: economic_events    → Remove entirely if needed

Token estimation formula:
  tokens ≈ (character_count / 4) × 1.2
  The 1.2× multiplier accounts for JSON overhead (keys, quotes,
  brackets, colons). Accuracy is within ±15% of actual tiktoken
  count — sufficient for budget enforcement.

6-step CoT for prepare_input:
  Step 1: Estimate all components
  Step 2: Check if within budget
  Step 3: If over budget → trim in priority order
  Step 4: Re-estimate after trimming
  Step 5: If still over → raise TokenBudgetError
  Step 6: Return prepared input + final token count
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy

from module1_data_layer.models import MarketData, TokenBudgetError
from module2_analysis_engine.config import (
    COT_INSTRUCTION_TOKENS,
    FIXED_PROMPT_TOKENS,
    HARD_TOKEN_LIMIT,
    INPUT_TOKEN_LIMIT,
    MARKET_DATA_BUDGET,
    MARKET_DATA_TOKENS,
    OUTPUT_TOKEN_LIMIT,
    STRUCTURE_OVERHEAD_TOKENS,
    SYSTEM_PROMPT_TOKENS,
    USER_MEMORY_TOKENS,
)
from module2_analysis_engine.models import UserContext

logger = logging.getLogger("swing_advisor.token_controller")


class TokenController:
    """Enforces token budgets for every Claude API call.

    This is the gatekeeper between Module 2 and the Claude API.
    No prompt goes to Claude without passing through this controller.
    If the input is too large, it trims intelligently. If trimming
    cannot bring it within budget, it raises TokenBudgetError.

    The controller follows a 6-step CoT process:
      Step 1: Estimate all component token counts
      Step 2: Check total against INPUT_TOKEN_LIMIT (1600)
      Step 3: Trim data components in priority order
      Step 4: Re-estimate after trimming
      Step 5: Raise TokenBudgetError if still over budget
      Step 6: Return the prepared input with token accounting

    Usage:
        controller = TokenController()
        prepared, token_count = controller.prepare_input(
            market_data=market_data,
            user_context=user_context,
            system_prompt=MASTER_SYSTEM_PROMPT,
            cot_instruction=COT_INSTRUCTION,
        )
        # prepared is a dict with trimmed market_data_json + user_context_json
        # token_count is the estimated total input tokens
    """

    HARD_LIMIT: int = HARD_TOKEN_LIMIT        # 3000 — absolute max per call
    INPUT_LIMIT: int = INPUT_TOKEN_LIMIT      # 2200 — max input tokens
    OUTPUT_LIMIT: int = OUTPUT_TOKEN_LIMIT    # 800 — max output tokens

    def estimate(self, text: str) -> int:
        """Estimate token count for a text string.

        Formula: (character_count / 4) × 1.2
          → 1 token ≈ 4 characters (standard LLM heuristic)
          → 1.2× multiplier for JSON overhead

        Accuracy: ±15% of actual tiktoken count.
        This is sufficient for budget enforcement without
        adding a tokenizer dependency.

        Args:
            text: Any text string (prompt, JSON, etc.).

        Returns:
            Estimated token count (integer).
        """
        if not text:
            return 0
        return int((len(text) / 4) * 1.2)

    def estimate_components(
        self,
        system_prompt: str,
        cot_instruction: str,
        market_data_json: str,
        user_context_json: str,
        task_instruction: str,
    ) -> dict[str, int]:
        """Estimate token count for each prompt component.

        Returns a breakdown so we know exactly what is consuming
        tokens and what to trim first.

        Args:
            system_prompt: MASTER_SYSTEM_PROMPT text.
            cot_instruction: COT_INSTRUCTION text.
            market_data_json: Serialized MarketData JSON.
            user_context_json: Serialized UserContext JSON.
            task_instruction: Task-specific prompt (grounding + format + task).

        Returns:
            Dict mapping component name to estimated token count.
        """
        breakdown = {
            "system_prompt": self.estimate(system_prompt),
            "cot_instruction": self.estimate(cot_instruction),
            "market_data": self.estimate(market_data_json),
            "user_context": self.estimate(user_context_json),
            "task_instruction": self.estimate(task_instruction),
            "structure_overhead": STRUCTURE_OVERHEAD_TOKENS,
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    def prepare_input(
        self,
        market_data: MarketData,
        user_context: UserContext,
        system_prompt: str,
        cot_instruction: str,
        task_instruction: str = "",
    ) -> tuple[dict[str, str], int]:
        """Prepare and budget-check the complete Claude API input.

        6-step CoT process:
          Step 1: Serialize and estimate all components.
          Step 2: Check if total is within INPUT_LIMIT (2200).
          Step 3: If over budget → trim data in priority order.
          Step 4: Re-estimate after each trim step.
          Step 5: If still over after all trimming → raise TokenBudgetError.
          Step 6: Return prepared data + final token count.

        Trimming priority (from spec Section 6):
          Priority 1: Keep system_prompt (NEVER trim)
          Priority 2: Keep cot_instruction (NEVER trim)
          Priority 3: Keep user_context (critical for personalisation)
          Priority 4: Trim market_data stocks (lowest signal first)
          Priority 5: Trim news to top 3 only
          Priority 6: Remove economic_events entirely
          Priority 7: Strip cot_reasoning from stocks (keep signal labels)

        Args:
            market_data: MarketData from Module 1 pipeline.
            user_context: UserContext for personalisation.
            system_prompt: MASTER_SYSTEM_PROMPT (fixed).
            cot_instruction: COT_INSTRUCTION (fixed).
            task_instruction: Task-specific instruction text.

        Returns:
            Tuple of:
              - dict with keys: market_data_json, user_context_json
                (trimmed to fit budget)
              - int: estimated total input token count

        Raises:
            TokenBudgetError: If input cannot be trimmed within budget
                after all trimming steps are exhausted.
        """
        reasoning_steps: list[str] = []

        # ── Step 1: Serialize and estimate ──
        # Work on a deep copy to avoid mutating the original MarketData
        trimmed_data = deepcopy(market_data)
        market_data_json = trimmed_data.model_dump_json(
            by_alias=True, exclude_none=True, exclude_defaults=True
        )
        user_context_json = user_context.model_dump_json(exclude_none=True)

        breakdown = self.estimate_components(
            system_prompt=system_prompt,
            cot_instruction=cot_instruction,
            market_data_json=market_data_json,
            user_context_json=user_context_json,
            task_instruction=task_instruction,
        )

        reasoning_steps.append(
            f"Step 1: Estimated tokens — "
            f"system={breakdown['system_prompt']}, "
            f"cot={breakdown['cot_instruction']}, "
            f"data={breakdown['market_data']}, "
            f"user={breakdown['user_context']}, "
            f"task={breakdown['task_instruction']}, "
            f"overhead={breakdown['structure_overhead']}. "
            f"Total={breakdown['total']}."
        )

        # ── Step 2: Check budget ──
        if breakdown["total"] <= self.INPUT_LIMIT:
            reasoning_steps.append(
                f"Step 2: Within budget ({breakdown['total']}/{self.INPUT_LIMIT}). "
                f"No trimming needed."
            )
            logger.info(
                f"Token budget OK: {breakdown['total']}/{self.INPUT_LIMIT} input tokens. "
                f"No trimming required."
            )
            return (
                {
                    "market_data_json": market_data_json,
                    "user_context_json": user_context_json,
                },
                breakdown["total"],
            )

        reasoning_steps.append(
            f"Step 2: Over budget ({breakdown['total']}/{self.INPUT_LIMIT}). "
            f"Starting priority-based trimming."
        )

        # ── Step 3: Trim in priority order ──
        # Fixed components that are NEVER trimmed:
        fixed_tokens = (
            breakdown["system_prompt"]
            + breakdown["cot_instruction"]
            + breakdown["task_instruction"]
            + breakdown["structure_overhead"]
        )
        # Budget available for variable data:
        data_budget = self.INPUT_LIMIT - fixed_tokens

        reasoning_steps.append(
            f"Step 3: Fixed tokens={fixed_tokens}. "
            f"Data budget={data_budget} tokens for market_data + user_context."
        )

        # Reserve space for user context (Priority 3 — keep if possible)
        user_tokens = breakdown["user_context"]
        market_budget = data_budget - user_tokens

        if market_budget < 200:
            # User context is too large — trim user context to minimum
            user_context_slim = UserContext(
                user_id=user_context.user_id,
                display_name=user_context.display_name,
                total_capital=user_context.total_capital,
                risk_tolerance=user_context.risk_tolerance,
                open_positions=user_context.open_positions[:3],
            )
            user_context_json = user_context_slim.model_dump_json(exclude_none=True)
            user_tokens = self.estimate(user_context_json)
            market_budget = data_budget - user_tokens
            reasoning_steps.append(
                f"Step 3a: User context trimmed to essentials. "
                f"User tokens now={user_tokens}."
            )

        # Trim market data to fit market_budget
        market_data_json = self._trim_market_data(
            trimmed_data, market_budget, reasoning_steps
        )

        # ── Step 4: Re-estimate ──
        final_breakdown = self.estimate_components(
            system_prompt=system_prompt,
            cot_instruction=cot_instruction,
            market_data_json=market_data_json,
            user_context_json=user_context_json,
            task_instruction=task_instruction,
        )

        reasoning_steps.append(
            f"Step 4: After trimming — total={final_breakdown['total']} tokens "
            f"(budget={self.INPUT_LIMIT})."
        )

        # ── Step 5: Final check ──
        if final_breakdown["total"] > self.INPUT_LIMIT:
            reasoning_steps.append(
                f"Step 5: Still over budget after all trimming. "
                f"Raising TokenBudgetError."
            )
            logger.error(
                f"Token budget EXCEEDED after all trimming: "
                f"{final_breakdown['total']}/{self.INPUT_LIMIT}. "
                f"CoT: {' | '.join(reasoning_steps)}"
            )
            raise TokenBudgetError(
                estimated_tokens=final_breakdown["total"],
                budget=self.INPUT_LIMIT,
            )

        # ── Step 6: Return prepared input ──
        reasoning_steps.append(
            f"Step 6: Budget satisfied. Returning prepared input. "
            f"Final tokens: {final_breakdown['total']}/{self.INPUT_LIMIT}."
        )

        logger.info(
            f"Token budget managed: {final_breakdown['total']}/{self.INPUT_LIMIT} input tokens. "
            f"Market data: {final_breakdown['market_data']} tokens. "
            f"User context: {final_breakdown['user_context']} tokens."
        )

        return (
            {
                "market_data_json": market_data_json,
                "user_context_json": user_context_json,
            },
            final_breakdown["total"],
        )

    def _trim_market_data(
        self,
        data: MarketData,
        budget: int,
        reasoning: list[str],
    ) -> str:
        """Trim MarketData to fit within the given token budget.

        Follows aggressive signal-priority-based trimming (8 steps):
          Step A: Strip cot_reasoning from all stocks
          Step B: Keep top 4 stocks by signal priority
          Step C: Strip OHLC detail from stocks (keep price, change_pct)
          Step D: Keep top 3 news by relevance, strip news cot_reasoning
          Step E: Remove economic events entirely
          Step F: Remove all sectors
          Step G: Remove all news (last resort)
          Step H: Emergency trim to 3 stocks

        Stock signal priority (best signals kept first):
          breakout_watch > unusual_activity > strong_momentum >
          selling_pressure > accumulation_zone > value_zone > neutral

        Each step re-serializes and checks if within budget.

        Args:
            data: MarketData object (will be mutated — caller passes deepcopy).
            budget: Maximum tokens for the serialized MarketData.
            reasoning: List of reasoning step strings (appended to).

        Returns:
            Serialized JSON string of the trimmed MarketData.
        """
        signal_priority = [
            "breakout_watch",
            "unusual_activity",
            "strong_momentum",
            "selling_pressure",
            "accumulation_zone",
            "value_zone",
            "neutral",
        ]

        def _serialize_and_check() -> tuple[str, int]:
            json_str = data.model_dump_json(
                by_alias=True, exclude_none=True, exclude_defaults=True
            )
            tokens = self.estimate(json_str)
            return json_str, tokens

        # Check current size
        current_json, current_tokens = _serialize_and_check()
        if current_tokens <= budget:
            return current_json

        # ── Step A: Strip cot_reasoning from all stocks ──
        for stock in data.stocks:
            stock.cot_reasoning = None
        current_json, current_tokens = _serialize_and_check()
        reasoning.append(
            f"Trim A: Stripped cot_reasoning from {len(data.stocks)} stocks. "
            f"Market data now {current_tokens} tokens."
        )
        if current_tokens <= budget:
            return current_json

        # ── Step B: Keep top 4 stocks by signal priority ──
        if len(data.stocks) > 4:
            data.stocks = sorted(
                data.stocks,
                key=lambda s: (
                    signal_priority.index(s.advisor_flag)
                    if s.advisor_flag in signal_priority
                    else 99
                ),
            )[:4]
            current_json, current_tokens = _serialize_and_check()
            reasoning.append(
                f"Trim B: Trimmed to top 4 stocks by signal priority. "
                f"Market data now {current_tokens} tokens."
            )
            if current_tokens <= budget:
                return current_json

        # ── Step C: Strip OHLC detail from stocks ──
        for stock in data.stocks:
            stock.open = 0.0
            stock.high = 0.0
            stock.low = 0.0
            stock.avg_volume_30d = 0
            stock.volume = 0
        current_json, current_tokens = _serialize_and_check()
        reasoning.append(
            f"Trim C: Stripped OHLC detail from stocks. "
            f"Market data now {current_tokens} tokens."
        )
        if current_tokens <= budget:
            return current_json

        # ── Step D: Keep top 3 news by relevance, strip news cot_reasoning ──
        if data.news:
            data.news = sorted(
                data.news,
                key=lambda n: n.relevance_score,
                reverse=True,
            )[:3]
            for news in data.news:
                news.cot_reasoning = None
            current_json, current_tokens = _serialize_and_check()
            reasoning.append(
                f"Trim D: Trimmed to top 3 news, stripped news CoT. "
                f"Market data now {current_tokens} tokens."
            )
            if current_tokens <= budget:
                return current_json

        # ── Step E: Remove economic events entirely ──
        if data.economic_events:
            removed_count = len(data.economic_events)
            data.economic_events = []
            current_json, current_tokens = _serialize_and_check()
            reasoning.append(
                f"Trim E: Removed {removed_count} economic events. "
                f"Market data now {current_tokens} tokens."
            )
            if current_tokens <= budget:
                return current_json

        # ── Step F: Remove all sectors ──
        if data.sectors:
            data.sectors = []
            current_json, current_tokens = _serialize_and_check()
            reasoning.append(
                f"Trim F: Removed all sectors. "
                f"Market data now {current_tokens} tokens."
            )
            if current_tokens <= budget:
                return current_json

        # ── Step G: Remove all news (last resort) ──
        if data.news:
            data.news = []
            current_json, current_tokens = _serialize_and_check()
            reasoning.append(
                f"Trim G: Removed all news. "
                f"Market data now {current_tokens} tokens."
            )
            if current_tokens <= budget:
                return current_json

        # ── Step H: Emergency trim to 3 stocks ──
        if len(data.stocks) > 3:
            data.stocks = data.stocks[:3]
            current_json, current_tokens = _serialize_and_check()
            reasoning.append(
                f"Trim H: Emergency trim to 3 stocks. "
                f"Market data now {current_tokens} tokens."
            )
            if current_tokens <= budget:
                return current_json

        # If we get here, even bare minimum data is over budget
        reasoning.append(
            f"Trim exhausted: Market data still {current_tokens} tokens "
            f"(budget {budget}). Returning best effort."
        )
        logger.warning(
            f"Market data trimming exhausted. "
            f"Still {current_tokens} tokens (budget {budget}). "
            f"Returning minimal data."
        )
        return current_json

    def trim_to_budget(
        self,
        market_data: MarketData,
        fixed_tokens: int = FIXED_PROMPT_TOKENS,
    ) -> MarketData:
        """Trim MarketData to fit within the Claude API token budget.

        This is the public convenience method that downstream callers
        (engine.py, analysis_crew.py) use to ensure MarketData fits
        before building prompts.

        Budget math:
          fixed_tokens = system_prompt(380) + cot_instruction(180)
                       + user_memory(300) = 860 tokens reserved
          output = 1500 tokens reserved
          market_data_budget = 3000 - 860 - 1500 = 640 tokens max

        Trim steps (signal-priority order):
          Step 1: Estimate current MarketData tokens
          Step 2: If under 1340 → return as is
          Step 3: Strip cot_reasoning from all stocks
          Step 4: Remove advisor_note from stocks
          Step 5: Keep only top 4 stocks by signal priority
          Step 6: Strip OHLC detail from stocks
          Step 7: Keep only top 3 news by relevance_score
          Step 8: Remove economic_events entirely
          Step 9: Emergency trim to 3 stocks
          Step 10: Re-estimate → if still over → raise TokenBudgetError

        Args:
            market_data: MarketData from Module 1 pipeline.
            fixed_tokens: Reserved tokens for fixed prompt components.
                Defaults to FIXED_PROMPT_TOKENS (860).

        Returns:
            MarketData trimmed to fit within MARKET_DATA_BUDGET.

        Raises:
            TokenBudgetError: If MarketData cannot be trimmed below budget.
        """
        market_budget = self.HARD_LIMIT - fixed_tokens - self.OUTPUT_LIMIT

        # Verify budget math
        assert fixed_tokens + market_budget + self.OUTPUT_LIMIT <= self.HARD_LIMIT, (
            f"Budget math error: {fixed_tokens} + "
            f"{market_budget} + {self.OUTPUT_LIMIT} > {self.HARD_LIMIT}"
        )

        # Step 1: Estimate current size
        trimmed = deepcopy(market_data)
        current_json = trimmed.model_dump_json(
            by_alias=True, exclude_none=True, exclude_defaults=True
        )
        current = self.estimate(current_json)

        logger.info(
            f"[TokenController] Starting trim. "
            f"Current: {current} tokens. "
            f"Target: {market_budget} tokens."
        )

        # Step 2: Already within budget
        if current <= market_budget:
            logger.info(
                f"[TokenController] Within budget. No trim needed."
            )
            return market_data

        # Steps 3-9: Delegate to _trim_market_data
        reasoning: list[str] = []
        self._trim_market_data(trimmed, market_budget, reasoning)

        # Step 10: Final check
        final_json = trimmed.model_dump_json(
            by_alias=True, exclude_none=True, exclude_defaults=True
        )
        final_tokens = self.estimate(final_json)

        if final_tokens > market_budget:
            raise TokenBudgetError(
                estimated_tokens=final_tokens,
                budget=market_budget,
            )

        # Log result
        saved = current - final_tokens
        logger.info(
            f"[TokenController] Trim complete. "
            f"Final tokens: {final_tokens}/{market_budget}. "
            f"Saved: {saved} tokens. "
            f"Stocks: {len(trimmed.stocks)}, "
            f"News: {len(trimmed.news)}, "
            f"Events: {len(trimmed.economic_events)}."
        )

        return trimmed

    def validate_output(self, response: str) -> bool:
        """Validate that Claude's response is within the output token budget.

        Called after receiving Claude's response. Logs a warning if
        the response exceeds OUTPUT_LIMIT but does not reject it —
        the response is already generated and paid for.

        Args:
            response: Claude's raw response text.

        Returns:
            True if response is within the hard limit.
            False if response exceeds HARD_LIMIT (should never happen
            with proper max_tokens configuration).
        """
        output_tokens = self.estimate(response)

        if output_tokens > self.OUTPUT_LIMIT:
            logger.warning(
                f"Claude output exceeded budget: "
                f"{output_tokens}/{self.OUTPUT_LIMIT} tokens. "
                f"Consider tightening output instructions."
            )

        within_limit = output_tokens <= self.HARD_LIMIT
        if not within_limit:
            logger.error(
                f"Claude output exceeded HARD LIMIT: "
                f"{output_tokens}/{self.HARD_LIMIT} tokens. "
                f"This should never happen with max_tokens set correctly."
            )

        return within_limit

    def get_budget_summary(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> dict[str, int | bool]:
        """Generate a budget summary for logging and metadata.

        Args:
            input_tokens: Estimated input tokens sent to Claude.
            output_tokens: Estimated output tokens received.

        Returns:
            Dict with token accounting and budget status.
        """
        total = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "input_budget": self.INPUT_LIMIT,
            "output_budget": self.OUTPUT_LIMIT,
            "hard_limit": self.HARD_LIMIT,
            "input_within_budget": input_tokens <= self.INPUT_LIMIT,
            "output_within_budget": output_tokens <= self.OUTPUT_LIMIT,
            "total_within_budget": total <= self.HARD_LIMIT,
        }


# Module-level singleton — used across the analysis engine
token_controller = TokenController()
