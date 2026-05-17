"""
SwingAdvisorBot — Module 1: Data Layer
agents/base_agent.py — Base agent class for all SwingAdvisorBot agents

This is the foundation class that every agent in the SwingAdvisorBot
system inherits from. Module 1 defines it, Modules 2–4 extend it.

CrewAI skeleton architecture:
  Module 1 (here):
    → SwingAdvisorBaseAgent  (base class — personality, CoT, validation)
    → DataCollectorAgent     (fetches and enriches market data)

  Module 2 (AI Analysis Engine):
    → MarketAnalysisAgent    (analyzes data via Claude API)
    → EducationAgent         (teaches trading concepts)

  Module 3 (Risk Engine):
    → RiskAssessmentAgent    (evaluates position risk)

  Module 4 (Trade Setup Generator):
    → TradeSetupAgent        (generates actionable setups)

Personality rules (enforced by base class):
  → Every agent reasons before acting (CoT enforced via log_reasoning)
  → Every agent validates its output (self-reflection via validate_output)
  → Every agent speaks like a professional finance advisor assistant
  → No agent returns data without signals or context
  → No agent guesses — if data is unavailable, it says so clearly

Design decisions:
  - Extends CrewAI Agent for framework compatibility and future crew composition.
  - CrewAI Agent is a Pydantic BaseModel — fields are declared as class attributes.
  - abstract execute() pattern enforced via NotImplementedError (Pydantic models
    don't support abc.abstractmethod cleanly — this achieves the same contract).
  - CoT logging is structured: step number + thought string + timestamp.
  - validate_output() returns (is_valid, issues_list) for pipeline health reporting.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from crewai import Agent
from pydantic import Field
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.agents")


class SwingAdvisorBaseAgent(Agent):
    """Base class for all SwingAdvisorBot agents.

    All agents in this system inherit from this class and get:
      1. Advisor personality (role, goal, backstory) — consistent voice.
      2. Chain of Thought logging (log_reasoning) — every decision documented.
      3. Output validation (validate_output) — self-reflection before returning.
      4. Execution contract (execute) — every agent must implement this.

    Personality: 20+ year senior finance advisor assistant.
    The agent is meticulous, accurate, and never guesses.
    If it doesn't have real data — it says so clearly.

    CrewAI integration:
      This class extends crewai.Agent, which is a Pydantic BaseModel.
      CrewAI uses role, goal, and backstory fields to shape agent behaviour
      when integrated with LLMs in Module 2. Here in Module 1, the agent
      operates as a structured data pipeline executor.

    Subclasses must override:
      - agent_name: Unique identifier for logging and health reports.
      - execute(**kwargs): The agent's primary action method.
    """

    # ── Agent Identity ──
    agent_name: str = Field(
        default="BaseAgent",
        description="Unique agent name for logging, health reports, and crew identification.",
    )

    # ── CrewAI Personality Fields ──
    role: str = Field(
        default="Senior Finance Advisor Assistant",
        description=(
            "CrewAI role field. Defines how the agent presents itself. "
            "All SwingAdvisorBot agents are assistants to a 20+ year senior advisor."
        ),
    )
    goal: str = Field(
        default=(
            "Provide accurate, signal-rich market data for advisor analysis. "
            "Every data point must carry enough context for a senior finance "
            "advisor to make an informed swing trading recommendation."
        ),
        description="CrewAI goal field. What this agent is trying to achieve.",
    )
    backstory: str = Field(
        default=(
            "You are an assistant to a 20+ year senior finance advisor who "
            "manages swing trading strategies for NSE (National Stock Exchange) "
            "stocks. Your job is to prepare the morning market briefing with "
            "real-time data, advisor-quality signals, and Chain of Thought "
            "reasoning. You are meticulous, accurate, and never guess. If you "
            "don't have real data — you say so clearly. Every piece of data "
            "you produce must be good enough for the advisor to speak "
            "confidently to a retail investor learning to trade."
        ),
        description="CrewAI backstory field. The agent's background and working principles.",
    )

    # ── CoT Tracking ──
    reasoning_log: list[str] = Field(
        default_factory=list,
        description="Accumulated Chain of Thought reasoning steps from the current execution.",
        exclude=True,
    )

    def log_reasoning(self, step: int, thought: str) -> None:
        """Log a Chain of Thought reasoning step.

        Every significant decision the agent makes is logged as a numbered
        step. This creates an auditable trail that:
          1. The pipeline health check can inspect for completeness.
          2. The advisor can reference when explaining data quality.
          3. Engineers can debug when signals seem wrong.

        Log format:
          [DataCollectorAgent] CoT Step 3: Fetched 15 stocks via Kite API.
          Volume data enriched. 2 stocks returned empty — flagged for retry.

        Args:
            step: Sequential step number (1-based). Steps should follow
                  the CoT pattern defined in the agent's execute() docstring.
            thought: Plain English description of what was done and why.
                     Must be informative enough for a non-technical advisor
                     to understand.
        """
        timestamp = datetime.now(IST).strftime("%H:%M:%S")
        log_entry = f"[{self.agent_name}] CoT Step {step}: {thought}"
        self.reasoning_log.append(log_entry)
        logger.info(f"{log_entry} (at {timestamp} IST)")

    def validate_output(self, output: Any) -> tuple[bool, list[str]]:
        """Self-reflection — validate output quality before returning.

        Every agent must ask: "Would a senior finance advisor be satisfied
        with this output?" This method performs baseline validation that
        subclasses can extend with domain-specific checks.

        Baseline checks:
          1. Output is not None.
          2. Output is not an empty collection (list, dict).
          3. reasoning_log has at least one entry (CoT was followed).

        Subclasses should call super().validate_output(output) first,
        then add their own checks. Return the combined results.

        Args:
            output: The data this agent produced during execute().

        Returns:
            Tuple of (is_valid: bool, issues: list[str]).
            is_valid is True only if issues list is empty.
            Each issue string describes what failed and why.
        """
        issues: list[str] = []

        if output is None:
            issues.append(
                f"{self.agent_name} returned None. "
                f"Every agent must return a concrete result — never None."
            )

        if isinstance(output, (list, dict)) and len(output) == 0:
            issues.append(
                f"{self.agent_name} returned an empty {type(output).__name__}. "
                f"The advisor cannot work with empty data."
            )

        if not self.reasoning_log:
            issues.append(
                f"{self.agent_name} has no reasoning log entries. "
                f"Chain of Thought must be followed — every decision logged."
            )

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning(
                f"{self.agent_name} output validation failed with "
                f"{len(issues)} issue(s): {'; '.join(issues)}"
            )
        else:
            logger.info(
                f"{self.agent_name} output validation passed. "
                f"CoT steps logged: {len(self.reasoning_log)}."
            )

        return is_valid, issues

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the agent's primary action.

        Every subclass MUST override this method with its specific
        data pipeline logic. The base implementation raises
        NotImplementedError to enforce the contract.

        Contract:
          1. Call log_reasoning() for each significant step.
          2. Call validate_output() before returning.
          3. Return a typed result (Pydantic model, not raw dict).
          4. Raise DataFetchError on unrecoverable failures — never return None.
          5. Never return fake/mock data — is_real_data must be True.

        Args:
            **kwargs: Subclass-specific parameters (tickers, config, etc.)

        Returns:
            Subclass-specific typed result (e.g., MarketData for DataCollectorAgent).

        Raises:
            NotImplementedError: If a subclass forgets to implement execute().
            DataFetchError: On unrecoverable data fetch failures.
        """
        raise NotImplementedError(
            f"{self.agent_name} must implement execute(). "
            f"Every SwingAdvisorBot agent has a concrete execution path — "
            f"no abstract agents in production."
        )

    def reset_reasoning(self) -> None:
        """Clear the reasoning log for a fresh execution cycle.

        Called at the start of each execute() run so that reasoning
        steps from previous runs don't pollute the current log.
        """
        self.reasoning_log = []
        logger.debug(f"{self.agent_name} reasoning log cleared for new execution.")

    def get_reasoning_summary(self) -> str:
        """Get a formatted summary of all CoT reasoning steps.

        Used by the pipeline to include reasoning context in the
        MarketData object, and by the health check to verify that
        the agent followed the expected CoT pattern.

        Returns:
            Newline-separated string of all reasoning steps,
            or a message indicating no reasoning was logged.
        """
        if not self.reasoning_log:
            return f"{self.agent_name}: No reasoning steps logged."
        return "\n".join(self.reasoning_log)
