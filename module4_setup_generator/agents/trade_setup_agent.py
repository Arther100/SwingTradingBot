"""
SwingAdvisorBot — Module 4: Trade Setup Generator
agents/trade_setup_agent.py — 10-step CoT CrewAI agent

This agent orchestrates the full setup generation pipeline:
  Step 1:  Load user context (capital, tolerance, tickers)
  Step 2:  Screen stocks via StockScreener (M1 → candidates)
  Step 3:  Calculate technical levels per candidate
  Step 4:  Run M3 risk check per candidate (BEFORE Claude)
  Step 5:  Score confidence per approved candidate
  Step 6:  Filter by min_confidence threshold
  Step 7:  Call Claude for setup reasoning (approved stocks only)
  Step 8:  Assemble TradeSetup objects
  Step 9:  Build SetupPackage with market context
  Step 10: Validate output quality

Key constraint: M3 BEFORE Claude — never spend tokens on rejected trades.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import Field
from zoneinfo import ZoneInfo

from module1_data_layer.agents.base_agent import SwingAdvisorBaseAgent
from module1_data_layer.models import MarketData, StockData
from module2_analysis_engine.models import MarketAnalysis
from module3_risk_engine.engine import risk_engine
from module3_risk_engine.models import RiskReport
from module4_setup_generator.config import (
    get_company_name,
    get_lesson_for_index,
    get_setup_config,
)
from module4_setup_generator.models import (
    SetupFilter,
    SetupFreshness,
    SetupPackage,
    SetupType,
    SkippedSetup,
    TradeSetup,
)
from module4_setup_generator.technical.confidence_scorer import confidence_scorer
from module4_setup_generator.technical.level_calculator import level_calculator
from module4_setup_generator.technical.stock_screener import stock_screener

logger = logging.getLogger("swing_advisor.trade_setup_agent")
IST = ZoneInfo("Asia/Kolkata")


class TradeSetupAgent(SwingAdvisorBaseAgent):
    """Trade setup generation agent — the deal maker.

    Takes raw market data + analysis + risk rules
    and produces actionable trade setups for the user.

    10-step Chain of Thought. No shortcuts.
    M3 risk check BEFORE Claude API call.

    Usage:
        agent = TradeSetupAgent()
        package = agent.execute(
            market_data=market_data,
            analysis=market_analysis,
            setup_filter=setup_filter,
        )
    """

    # ── Agent Identity ──
    agent_name: str = Field(
        default="TradeSetupAgent",
        description="Unique agent name for logging and crew identification.",
    )
    role: str = Field(
        default="Trade Setup Specialist",
        description="CrewAI role — setup generator.",
    )
    goal: str = Field(
        default=(
            "Generate 3-5 actionable swing trade setups with clear entry, "
            "target, stop loss, and reasoning. Every setup must pass M3 risk "
            "validation before receiving Claude reasoning. Quality over quantity."
        ),
        description="CrewAI goal — what this agent achieves.",
    )
    backstory: str = Field(
        default=(
            "You are a trade setup specialist working alongside a senior "
            "finance advisor with 20+ years of NSE swing trading experience. "
            "You screen stocks for actionable signals, calculate precise "
            "technical levels, validate every trade through the risk engine, "
            "and generate data-grounded reasoning. You never recommend "
            "a trade that fails risk rules. You never guess prices."
        ),
        description="CrewAI backstory.",
    )

    def execute(  # type: ignore[override]
        self,
        market_data: MarketData,
        analysis: MarketAnalysis,
        setup_filter: Optional[SetupFilter] = None,
        claude_reasoning_fn: Optional[Any] = None,
    ) -> SetupPackage:
        """Execute the 10-step setup generation pipeline.

        Args:
            market_data: M1 MarketData with stocks.
            analysis: M2 MarketAnalysis with mood and sectors.
            setup_filter: User preferences (capital, tolerance, tickers).
            claude_reasoning_fn: Async function to call Claude for reasoning.
                Signature: (stock, levels, risk_report, analysis, filter) -> dict
                If None, setups are generated without Claude reasoning.

        Returns:
            SetupPackage with setups, skipped stocks, and market context.
        """
        self.reasoning_log = []
        config = get_setup_config()
        sf = setup_filter or SetupFilter()
        setups: list[TradeSetup] = []
        skipped: list[SkippedSetup] = []
        total_input_tokens = 0
        total_output_tokens = 0

        # ── Step 1: Load user context ──
        self.log_reasoning(1, (
            f"User: {sf.display_name}, Capital: ₹{sf.capital}, "
            f"Tolerance: {sf.risk_tolerance}, "
            f"Max setups: {sf.max_setups}, "
            f"Min confidence: {sf.min_confidence}"
        ))

        # ── Step 2: Screen stocks ──
        candidates = stock_screener.screen(
            market_data=market_data,
            max_candidates=config.max_candidates,
            specific_tickers=sf.tickers,
        )

        # Record skipped stocks from screening
        screened_tickers = {c.ticker for c in candidates}
        for stock in market_data.stocks:
            if stock.ticker not in screened_tickers:
                skip_reason = stock_screener.get_skip_reason(stock)
                if skip_reason:
                    skipped.append(SkippedSetup(
                        ticker=stock.ticker,
                        skip_reason=skip_reason,
                    ))

        self.log_reasoning(2, (
            f"Screened {len(market_data.stocks)} stocks → "
            f"{len(candidates)} candidates, "
            f"{len(skipped)} skipped"
        ))

        if not candidates:
            self.log_reasoning(10, "No candidates found. Returning empty package.")
            return self._build_package(
                setups=[], skipped=skipped, analysis=analysis,
                market_data=market_data, sf=sf,
            )

        # ── Step 3–8: Process each candidate ──
        for idx, stock in enumerate(candidates):
            if len(setups) >= sf.max_setups:
                self.log_reasoning(
                    3, f"Reached max setups ({sf.max_setups}). Stopping."
                )
                break

            setup_result = self._process_candidate(
                stock=stock,
                idx=idx,
                analysis=analysis,
                sf=sf,
                market_data=market_data,
                claude_reasoning_fn=claude_reasoning_fn,
            )

            if setup_result is None:
                # Stock was skipped — already added to skipped list internally
                continue

            if isinstance(setup_result, SkippedSetup):
                skipped.append(setup_result)
                continue

            setups.append(setup_result)

        # ── Step 9: Build package ──
        self.log_reasoning(9, (
            f"Built package: {len(setups)} setups, "
            f"{len(skipped)} skipped"
        ))

        # ── Step 10: Validate output ──
        is_valid, issues = self.validate_output(setups)
        self.log_reasoning(10, (
            f"Validation: {'PASS' if is_valid else 'ISSUES'} "
            f"({', '.join(issues) if issues else 'clean'})"
        ))

        return self._build_package(
            setups=setups,
            skipped=skipped,
            analysis=analysis,
            market_data=market_data,
            sf=sf,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

    def _process_candidate(
        self,
        stock: StockData,
        idx: int,
        analysis: MarketAnalysis,
        sf: SetupFilter,
        market_data: MarketData,
        claude_reasoning_fn: Optional[Any] = None,
    ) -> Optional[TradeSetup | SkippedSetup]:
        """Process a single stock candidate through steps 3-8.

        Returns:
            TradeSetup if approved, SkippedSetup if rejected, None on error.
        """
        ticker = stock.ticker
        flag_value = (
            stock.advisor_flag.value if stock.advisor_flag else None
        )

        # ── Step 3: Calculate technical levels ──
        levels = level_calculator.calculate(
            current_price=stock.price,
            advisor_flag=flag_value,
        )

        self.log_reasoning(3, (
            f"{ticker}: entry=[{levels.entry_zone_low}-{levels.entry_zone_high}] "
            f"stop={levels.stop_loss} target={levels.target_price} "
            f"R/R={levels.risk_reward_ratio}"
        ))

        # ── Step 4: M3 risk check (BEFORE Claude) ──
        try:
            risk_report = risk_engine.calculate_risk(
                ticker=ticker,
                entry_price=levels.entry_zone_low,
                target_price=levels.target_price,
                stop_loss=levels.stop_loss,
                vix_value=Decimal(str(market_data.india_vix or 14.0)),
                capital=Decimal(str(sf.capital)),
                tolerance=sf.risk_tolerance,
                display_name=sf.display_name,
            )
        except Exception as e:
            logger.error(f"[TradeSetupAgent] M3 risk check failed for {ticker}: {e}")
            return SkippedSetup(
                ticker=ticker,
                skip_reason=f"Risk check error: {str(e)[:80]}",
            )

        self.log_reasoning(4, (
            f"{ticker}: M3 verdict={risk_report.verdict.value}, "
            f"shares={risk_report.position_size_shares}, "
            f"risk=₹{risk_report.total_risk_rupees}"
        ))

        # Skip if M3 rejects
        if risk_report.verdict.value == "REJECTED":
            return SkippedSetup(
                ticker=ticker,
                skip_reason=risk_report.rejection_reason or "M3 rejected",
                risk_verdict=risk_report.verdict.value,
                advisor_note=risk_report.advisor_note,
            )

        # ── Step 5: Score confidence ──
        sector_mood = self._get_sector_mood(stock.sector, analysis)
        rr_decimal = self._parse_rr_ratio(levels.risk_reward_ratio)

        confidence = confidence_scorer.score(
            stock=stock,
            india_vix=market_data.india_vix or 15.0,
            sector_mood=sector_mood,
            risk_reward_ratio=rr_decimal,
        )

        self.log_reasoning(5, (
            f"{ticker}: confidence={confidence}, "
            f"sector_mood={sector_mood}, "
            f"volume={stock.volume_signal.value}"
        ))

        # ── Step 6: Filter by min_confidence ──
        if confidence < sf.min_confidence:
            self.log_reasoning(6, (
                f"{ticker}: confidence {confidence} < min {sf.min_confidence}. Skipped."
            ))
            return SkippedSetup(
                ticker=ticker,
                skip_reason=f"Confidence {confidence} below minimum {sf.min_confidence}",
                risk_verdict=risk_report.verdict.value,
            )

        self.log_reasoning(6, f"{ticker}: confidence {confidence} >= {sf.min_confidence}. Approved.")

        # ── Step 7: Claude reasoning (if available and not skipped) ──
        claude_fields: dict[str, str] = {}
        if claude_reasoning_fn is not None and not sf.skip_claude:
            try:
                claude_fields = claude_reasoning_fn(
                    stock=stock,
                    levels=levels,
                    risk_report=risk_report,
                    analysis=analysis,
                    setup_filter=sf,
                )
                self.log_reasoning(7, f"{ticker}: Claude reasoning received.")
            except Exception as e:
                logger.warning(
                    f"[TradeSetupAgent] Claude reasoning failed for {ticker}: {e}. "
                    "Using lesson fallback."
                )
                claude_fields = {}

        # Fallback lesson if Claude didn't provide one
        if not claude_fields.get("lesson"):
            lesson = get_lesson_for_index(idx)
            claude_fields.setdefault("lesson", lesson["template"])

        self.log_reasoning(7, f"{ticker}: reasoning fields ready.")

        # ── Step 8: Assemble TradeSetup ──
        setup = TradeSetup(
            ticker=ticker,
            company_name=get_company_name(ticker),
            sector=stock.sector or "Other",
            setup_type=SetupType.SWING_LONG,
            freshness=self._determine_freshness(),
            entry_zone_low=levels.entry_zone_low,
            entry_zone_high=levels.entry_zone_high,
            target_price=levels.target_price,
            stop_loss=levels.stop_loss,
            current_price=Decimal(str(stock.price)),
            confidence_score=confidence,
            risk_reward_ratio=levels.risk_reward_ratio,
            position_size_shares=risk_report.position_size_shares,
            position_size_rupees=Decimal(str(
                risk_report.position_size_shares * float(levels.entry_zone_low)
            )),
            max_risk_rupees=risk_report.total_risk_rupees,
            risk_pct_of_capital=Decimal(str(
                float(risk_report.total_risk_rupees or 0) / sf.capital * 100
            )).quantize(Decimal("0.01")),
            risk_verdict=risk_report.verdict.value,
            setup_reasoning=claude_fields.get("setup_reasoning"),
            entry_trigger=claude_fields.get("entry_trigger"),
            exit_strategy=claude_fields.get("exit_strategy"),
            risk_warning=claude_fields.get("risk_warning"),
            macro_context=claude_fields.get("macro_context"),
            lesson=claude_fields.get("lesson"),
            cot_reasoning="\n".join(self.reasoning_log),
            advisor_flag=flag_value,
            volume_signal=stock.volume_signal.value if stock.volume_signal else None,
        )

        self.log_reasoning(8, (
            f"{ticker}: TradeSetup assembled. "
            f"Entry [{levels.entry_zone_low}-{levels.entry_zone_high}], "
            f"Target {levels.target_price}, Stop {levels.stop_loss}, "
            f"Confidence {confidence}"
        ))

        return setup

    def _get_sector_mood(
        self, sector: str, analysis: MarketAnalysis
    ) -> Optional[str]:
        """Find sector mood from M2 analysis."""
        if not sector or not analysis.sector_analyses:
            return None

        sector_lower = sector.lower()
        for sa in analysis.sector_analyses:
            if sa.sector_name.lower() == sector_lower:
                return sa.sector_mood.value if hasattr(sa.sector_mood, "value") else str(sa.sector_mood)

        return None

    def _parse_rr_ratio(self, rr_string: str) -> Optional[Decimal]:
        """Parse R/R string like '1:3.00' → Decimal('3.00')."""
        try:
            parts = rr_string.split(":")
            if len(parts) == 2:
                return Decimal(parts[1])
        except Exception:
            pass
        return None

    def _determine_freshness(self) -> SetupFreshness:
        """Determine setup freshness based on current time."""
        now = datetime.now(IST)
        hour, minute = now.hour, now.minute
        current_minutes = hour * 60 + minute

        market_open = 9 * 60 + 15   # 9:15 IST
        market_close = 15 * 60 + 30  # 15:30 IST

        if market_open <= current_minutes <= market_close:
            return SetupFreshness.LIVE
        elif current_minutes < market_open:
            return SetupFreshness.PRE_MARKET_PREVIEW
        else:
            return SetupFreshness.NEXT_DAY_WATCHLIST

    def _build_package(
        self,
        setups: list[TradeSetup],
        skipped: list[SkippedSetup],
        analysis: MarketAnalysis,
        market_data: MarketData,
        sf: SetupFilter,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
    ) -> SetupPackage:
        """Build the final SetupPackage."""
        # Determine advisor note
        if not setups:
            advisor_note = (
                f"No setups met the quality threshold today "
                f"(min confidence: {sf.min_confidence}). "
                f"Market mood: {analysis.market_mood.value}. "
                f"Consider reviewing watchlist tomorrow."
            )
        else:
            top = setups[0]
            advisor_note = (
                f"Top pick: {top.ticker} ({top.company_name}) — "
                f"confidence {top.confidence_score}/10. "
                f"{len(setups)} setups generated, "
                f"{len(skipped)} stocks filtered out."
            )

        return SetupPackage(
            setups=setups,
            skipped_setups=skipped,
            market_mood=analysis.market_mood.value,
            india_vix=market_data.india_vix or 0.0,
            advisor_note=advisor_note,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            freshness=self._determine_freshness(),
        )

    def validate_output(
        self, setups: list[TradeSetup]
    ) -> tuple[bool, list[str]]:
        """Validate output quality.

        Returns:
            Tuple of (is_valid, list of issues).
        """
        issues: list[str] = []

        # Check for duplicate tickers
        tickers = [s.ticker for s in setups]
        if len(tickers) != len(set(tickers)):
            issues.append("Duplicate tickers in setups")

        # Check confidence scores are in valid range
        for s in setups:
            if s.confidence_score < 4.0 or s.confidence_score > 9.5:
                issues.append(
                    f"{s.ticker}: confidence {s.confidence_score} out of range"
                )

        # Check all setups have entry < target
        for s in setups:
            if s.entry_zone_high >= s.target_price:
                issues.append(
                    f"{s.ticker}: entry_high >= target_price"
                )

        # Check all setups have stop < entry
        for s in setups:
            if s.stop_loss >= s.entry_zone_low:
                issues.append(
                    f"{s.ticker}: stop_loss >= entry_low"
                )

        return (len(issues) == 0, issues)


# Module-level singleton
trade_setup_agent = TradeSetupAgent()
