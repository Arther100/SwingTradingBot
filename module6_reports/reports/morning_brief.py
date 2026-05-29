"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
reports/morning_brief.py — Morning brief generator

Orchestrates M1→M2→M3→M4→M5 pipeline at 8:50 AM IST,
builds a MorningBrief model, calls Claude for telegram_text,
and returns the complete brief for delivery.

One Claude call — 2630 tokens max (1830 input + 800 output).

Usage:
    from module6_reports.reports.morning_brief import generate_morning_brief

    brief = await generate_morning_brief(user_id="XCU700")
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Optional

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from module6_reports.config import (
    CLAUDE_MODEL,
    DEFAULT_TICKERS,
    IST,
    LESSON_CONCEPTS,
    LESSON_SUMMARIES,
    MAX_SETUPS_IN_BRIEF,
    MIN_SETUP_CONFIDENCE,
    MORNING_BRIEF_SYSTEM_PROMPT,
    MORNING_BRIEF_TOKEN_BUDGET,
    REPORT_MAX_EVENTS,
    REPORT_MAX_NEWS,
    REPORT_MAX_STOCKS,
)
from module6_reports.models import (
    DeliveryStatus,
    LessonOfDay,
    MorningBrief,
    PositionSummary,
    ReportType,
    SetupSummary,
)

logger = logging.getLogger("swing_advisor.morning_brief")


async def generate_morning_brief(
    user_id: str = "XCU700",
    tickers: list[str] | None = None,
    skip_claude: bool = False,
) -> MorningBrief:
    """Generate the complete morning brief.

    Pipeline:
      1. M5 — Load user profile + memory context
      2. M1 — Fetch market data (10 tickers)
      3. M2 — Analyse market mood
      4. M3 — Check VIX gate
      5. M4 — Generate trade setups (if VIX gate open)
      6. Build MorningBrief model from all module outputs
      7. Claude — Generate telegram_text (one call, 2630 tokens)

    Args:
        user_id: User identifier (default: XCU700 for Vijay).
        tickers: Override default ticker list. None = use DEFAULT_TICKERS.
        skip_claude: If True, skip Claude call (use template formatting).

    Returns:
        MorningBrief with all fields populated.

    Raises:
        Never raises — catches all errors and returns a brief with error field set.
    """
    start_ms = time.time()
    tickers = tickers or DEFAULT_TICKERS

    brief = MorningBrief(
        report_type=ReportType.MORNING_BRIEF,
        user_id=user_id,
        delivery_status=DeliveryStatus.PENDING,
    )

    try:
        # ── Step 1: Load user profile + memory context ──
        profile, memory_ctx = await _load_user_context(user_id)
        if profile:
            brief.total_capital = Decimal(str(profile.capital))
            brief.available_capital = Decimal(str(profile.capital))
            logger.info(
                f"[MorningBrief] Profile loaded: {profile.name}, "
                f"capital=₹{profile.capital:,.0f}"
            )

        # ── Step 2: Fetch market data via M1 ──
        # Always reload env before any Kite API call so the fresh token is used
        load_dotenv(override=True)
        fresh_token = os.getenv("KITE_ACCESS_TOKEN")
        logger.info(
            f"[MorningBrief] Using token: "
            f"{fresh_token[:10] if fresh_token else 'NONE'}..."
        )
        market_data = await _fetch_market_data(tickers, access_token=fresh_token)
        if market_data:
            brief.market_status = market_data.market_status.value
            brief.india_vix = market_data.india_vix
            brief.vix_signal = market_data.vix_signal.value
            brief.nifty_value = market_data.nifty50_value
            brief.nifty_change_pct = market_data.nifty50_change_pct
            brief.sensex_value = market_data.sensex_value
            brief.sensex_change_pct = market_data.sensex_change_pct
            # FII/DII and earnings from M1 (Upgrade 1)
            brief.fii_dii = market_data.fii_dii
            brief.earnings_calendar = market_data.earnings_events or []
            logger.info(
                f"[MorningBrief] M1 data: VIX={market_data.india_vix}, "
                f"Nifty={market_data.nifty50_value}, "
                f"FII/DII={'present' if market_data.fii_dii else 'absent'}, "
                f"Earnings={len(market_data.earnings_events or [])}"
            )

        # ── Step 3: Analyse market mood via M2 ──
        analysis = await _analyse_market(market_data)
        if analysis:
            brief.market_mood = analysis.analysis.market_mood.value
            brief.mood_confidence = analysis.analysis.mood_confidence
            brief.situation_summary = analysis.analysis.situation
            logger.info(
                f"[MorningBrief] M2 mood: {brief.market_mood} "
                f"(confidence={brief.mood_confidence:.2f})"
            )

        # ── Step 4: Check VIX gate via M3 ──
        vix_gate_status = _check_vix_gate(
            vix_value=market_data.india_vix if market_data else 14.0,
            tolerance=profile.risk_tolerance if profile else "moderate",
        )
        brief.vix_gate = vix_gate_status.get("gate", "closed")
        brief.vix_limit = vix_gate_status.get("vix_limit")
        logger.info(f"[MorningBrief] M3 VIX gate: {brief.vix_gate}")

        # ── Step 5: Generate setups via M4 (if VIX gate open) ──
        if brief.vix_gate == "open" and market_data and analysis:
            setups = await _generate_setups(market_data, analysis, profile)
            if setups:
                brief.top_setups = _convert_setups(setups.setups)
                brief.no_setup_reason = setups.reason
                brief.advisor_note = setups.advisor_note
                logger.info(
                    f"[MorningBrief] M4 setups: {len(brief.top_setups)} qualifying"
                )
        else:
            brief.no_setup_reason = "vix_gate_closed"
            logger.info("[MorningBrief] VIX gate closed — skipping M4 setups")

        # ── Step 5b: Load open positions from M5 ──
        positions = _load_open_positions(user_id, profile)
        if positions:
            brief.open_positions = positions
            # Adjust available capital for open positions
            position_value = sum(
                p.entry_price * p.shares for p in positions
            )
            if brief.total_capital:
                brief.available_capital = brief.total_capital - position_value

        # ── Step 6: Lesson of the day ──
        brief.lesson_of_day = _get_lesson_of_day()

        # ── Step 7: Claude telegram_text generation ──
        if not skip_claude:
            telegram_text = await _generate_telegram_text(
                brief, memory_ctx, market_data
            )
            if telegram_text:
                brief.telegram_text = telegram_text

    except Exception as e:
        logger.error(f"[MorningBrief] Pipeline error: {e}", exc_info=True)
        brief.error = str(e)

    # Finalize timing
    elapsed_ms = int((time.time() - start_ms) * 1000)
    brief.generation_time_ms = elapsed_ms
    logger.info(f"[MorningBrief] Generated in {elapsed_ms}ms")

    return brief


# ═══════════════════════════════════════════════════════════
# PIPELINE STEPS — each wraps a module call with error handling
# ═══════════════════════════════════════════════════════════


async def _load_user_context(user_id: str):
    """Load user profile and memory context from M5.

    Returns (UserProfile | None, MemoryContext | None).
    """
    try:
        from module5_memory.engine import memory_engine

        profile = memory_engine.get_user_profile(user_id)
        memory_ctx = memory_engine.get_memory_context(
            user_id=user_id,
            query="morning brief trading context",
            agent_name="morning_brief",
        )
        return profile, memory_ctx
    except Exception as e:
        logger.warning(f"[MorningBrief] M5 load failed: {e}")
        return None, None


async def _fetch_market_data(tickers: list[str], access_token: str | None = None):
    """Fetch market data from M1 pipeline.

    Returns MarketData or None on failure.
    """
    try:
        from module1_data_layer.models import DataFetchConfig
        from module1_data_layer.pipeline import run_data_pipeline

        config = DataFetchConfig(
            tickers=tickers,
            max_stocks=REPORT_MAX_STOCKS,
            max_news=REPORT_MAX_NEWS,
            max_events=REPORT_MAX_EVENTS,
        )
        market_data = await run_data_pipeline(
            tickers=tickers,
            config=config,
            access_token=access_token,
        )
        return market_data
    except Exception as e:
        logger.error(f"[MorningBrief] M1 data fetch failed: {e}")
        return None


async def _analyse_market(market_data):
    """Analyse market via M2.

    Returns AnalysisResult or None on failure.
    """
    if not market_data:
        return None

    try:
        from module2_analysis_engine.engine import analyse_market

        result = await analyse_market(market_data)
        return result
    except Exception as e:
        logger.error(f"[MorningBrief] M2 analysis failed: {e}")
        return None


def _check_vix_gate(
    vix_value: float = 14.0,
    tolerance: str = "moderate",
) -> dict:
    """Check VIX gate via M3.

    Returns dict with 'gate' (open/closed) and 'advisor_note'.
    """
    try:
        from module3_risk_engine.engine import risk_engine

        gate_status = risk_engine.get_vix_gate_status(
            vix_value=Decimal(str(vix_value)),
            tolerance=tolerance,
        )
        return {
            "gate": gate_status.gate,
            "advisor_note": gate_status.advisor_note,
            "vix_limit": getattr(gate_status, "vix_limit", None),
        }
    except Exception as e:
        logger.error(f"[MorningBrief] M3 VIX gate check failed: {e}")
        return {"gate": "closed", "advisor_note": f"VIX gate check failed: {e}"}


async def _generate_setups(market_data, analysis, profile):
    """Generate trade setups via M4.

    Returns SetupPackage or None on failure.
    """
    try:
        from module4_setup_generator.engine import setup_engine

        # Build setup filter from profile
        capital = float(profile.capital) if profile else 50000.0
        tolerance = profile.risk_tolerance if profile else "moderate"

        package = setup_engine.generate_setups(
            user_id=profile.user_id if profile else "XCU700",
            display_name=profile.name if profile else "Vijay",
            capital=capital,
            risk_tolerance=tolerance,
            max_setups=MAX_SETUPS_IN_BRIEF,
            min_confidence=MIN_SETUP_CONFIDENCE,
            tickers=[s.ticker for s in market_data.stocks] if market_data.stocks else None,
        )
        return package
    except Exception as e:
        logger.error(f"[MorningBrief] M4 setup generation failed: {e}")
        return None


def _convert_setups(trade_setups: list) -> list[SetupSummary]:
    """Convert M4 TradeSetup objects to M6 SetupSummary objects.

    Maps field names between M4 and M6 models.
    """
    summaries = []
    for ts in trade_setups[:MAX_SETUPS_IN_BRIEF]:
        if ts.confidence_score < MIN_SETUP_CONFIDENCE:
            continue

        summary = SetupSummary(
            ticker=ts.ticker,
            sector=ts.sector,
            entry_low=ts.entry_zone_low,
            entry_high=ts.entry_zone_high,
            target=ts.target_price,
            stop_loss=ts.stop_loss,
            confidence=ts.confidence_score,
            risk_rupees=ts.max_risk_rupees,
            reward_rupees=_calc_reward(ts),
            risk_reward=ts.risk_reward_ratio,
            shares=ts.position_size_shares,
            position_rupees=ts.position_size_rupees,
            risk_pct=ts.risk_pct_of_capital,
            setup_reasoning=ts.setup_reasoning,
            entry_trigger=getattr(ts, "entry_trigger", None),
            exit_strategy=getattr(ts, "exit_strategy", None),
            earnings_risk=getattr(ts, "earnings_risk", None),
        )
        summaries.append(summary)

    return summaries


def _calc_reward(ts) -> Optional[Decimal]:
    """Calculate reward in rupees from TradeSetup."""
    try:
        if ts.position_size_shares and ts.target_price and ts.entry_zone_high:
            reward_per_share = ts.target_price - ts.entry_zone_high
            return reward_per_share * ts.position_size_shares
    except Exception:
        pass
    return None


def _load_open_positions(
    user_id: str,
    profile,
) -> list[PositionSummary]:
    """Load open positions from M5 trade history.

    Returns list of PositionSummary for display in brief.
    """
    try:
        from module5_memory.engine import memory_engine

        trades = memory_engine.get_trade_history(user_id=user_id, limit=50)
        positions = []
        for trade in trades:
            if trade.status == "open":
                positions.append(
                    PositionSummary(
                        ticker=trade.ticker,
                        entry_price=trade.entry_price,
                        shares=trade.shares,
                        stop_loss=getattr(trade, "stop_loss", None),
                        target=getattr(trade, "target_price", None),
                    )
                )
        return positions
    except Exception as e:
        logger.warning(f"[MorningBrief] Failed to load positions: {e}")
        return []


def _get_lesson_of_day() -> LessonOfDay:
    """Get today's lesson based on day-of-year rotation.

    Cycles through LESSON_CONCEPTS sequentially.
    Uses LESSON_SUMMARIES for the summary text.
    """
    day_of_year = datetime.now(IST).timetuple().tm_yday
    index = day_of_year % len(LESSON_CONCEPTS)
    concept = LESSON_CONCEPTS[index]
    summary = LESSON_SUMMARIES.get(concept, f"Learn about {concept} today.")

    return LessonOfDay(
        concept=concept,
        summary=summary,
        difficulty="beginner",
    )


# ═══════════════════════════════════════════════════════════
# CLAUDE TELEGRAM TEXT GENERATION
# ═══════════════════════════════════════════════════════════


async def _generate_telegram_text(
    brief: MorningBrief,
    memory_ctx,
    market_data,
) -> Optional[str]:
    """Call Claude to generate the telegram_text for the brief.

    One call, 2630 tokens max (1830 input + 800 output).
    Returns the generated text or None on failure.
    """
    try:
        import httpx

        api_key = _get_anthropic_key()
        if not api_key:
            logger.warning("[MorningBrief] No ANTHROPIC_API_KEY — skipping Claude")
            return None

        user_prompt = _build_claude_prompt(brief, memory_ctx, market_data)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": MORNING_BRIEF_TOKEN_BUDGET["output_budget"],
                    "system": MORNING_BRIEF_SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=30.0,
            )

            data = response.json()

            if response.status_code != 200:
                error = data.get("error", {}).get("message", "Unknown error")
                logger.error(f"[MorningBrief] Claude API error: {error}")
                return None

            # Extract text from response
            content_blocks = data.get("content", [])
            text_parts = [
                block["text"]
                for block in content_blocks
                if block.get("type") == "text"
            ]
            telegram_text = "\n".join(text_parts)

            # Log token usage
            usage = data.get("usage", {})
            brief.tokens_used = (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            )
            logger.info(
                f"[MorningBrief] Claude: {usage.get('input_tokens', 0)} in + "
                f"{usage.get('output_tokens', 0)} out = {brief.tokens_used} tokens"
            )

            return telegram_text

    except Exception as e:
        logger.error(f"[MorningBrief] Claude call failed: {e}")
        return None


def _build_claude_prompt(
    brief: MorningBrief,
    memory_ctx,
    market_data,
) -> str:
    """Build the user prompt for Claude's morning brief generation.

    Structured sections matching the token budget:
      - Market data (~400 tokens)
      - Analysis summary (~200 tokens)
      - Risk summary (~150 tokens)
      - Setups (~300 tokens)
      - Memory context (~300 tokens)
      - Instruction (~100 tokens)
    """
    sections: list[str] = []

    # Market data section
    sections.append("== MARKET DATA ==")
    sections.append(f"VIX: {brief.india_vix}")
    sections.append(f"VIX Signal: {brief.vix_signal}")
    sections.append(f"Nifty: {brief.nifty_value} ({brief.nifty_change_pct:+.2f}%)")
    sections.append(f"Sensex: {brief.sensex_value} ({brief.sensex_change_pct:+.2f}%)")
    sections.append(f"Market Status: {brief.market_status}")

    # Top stock summaries from M1
    if market_data and market_data.stocks:
        sections.append("\nTop stocks:")
        for stock in market_data.stocks[:5]:
            sections.append(
                f"  {stock.ticker}: ₹{stock.current_price} "
                f"({stock.change_pct:+.2f}%) — {stock.advisor_flag.value}"
            )

    # FII/DII institutional flow section
    if brief.fii_dii:
        f = brief.fii_dii
        sections.append("\n== FII/DII FLOWS ==")
        sections.append(f"FII Net: ₹{f.fii_net:,.0f} Cr → Signal: {f.fii_signal.value}")
        sections.append(f"DII Net: ₹{f.dii_net:,.0f} Cr → Signal: {f.dii_signal.value}")
        sections.append(f"Combined: ₹{f.combined_net:,.0f} Cr → {f.combined_signal.value}")
        if f.consecutive_fii_buying_days and f.consecutive_fii_buying_days > 0:
            sections.append(f"FII buying streak: {f.consecutive_fii_buying_days} consecutive days")
        elif f.consecutive_fii_buying_days and f.consecutive_fii_buying_days < 0:
            streak = abs(f.consecutive_fii_buying_days)
            sections.append(f"FII selling streak: {streak} consecutive days")
        if f.advisor_note:
            sections.append(f"Advisor note: {f.advisor_note}")

    # Analysis section
    sections.append("\n== ANALYSIS ==")
    sections.append(f"Market Mood: {brief.market_mood}")
    sections.append(f"Confidence: {brief.mood_confidence:.2f}")
    if brief.situation_summary:
        # Truncate to ~200 tokens worth
        summary = brief.situation_summary[:600]
        sections.append(f"Situation: {summary}")

    # Risk section
    sections.append("\n== RISK ==")
    sections.append(f"VIX Gate: {brief.vix_gate}")
    if brief.vix_limit:
        sections.append(f"VIX Limit: {brief.vix_limit}")

    # Setups section
    sections.append("\n== SETUPS ==")
    if brief.top_setups:
        for s in brief.top_setups:
            earnings_note = ""
            if s.earnings_risk and s.earnings_risk.has_upcoming_earnings:
                earnings_note = (
                    f" [EARNINGS in {s.earnings_risk.days_to_result}d "
                    f"— {s.earnings_risk.risk_level.value}]"
                )
            sections.append(
                f"  {s.ticker}: Entry ₹{s.entry_low}-{s.entry_high}, "
                f"Target ₹{s.target}, Stop ₹{s.stop_loss}, "
                f"R/R {s.risk_reward}, Confidence {s.confidence:.1f}/10, "
                f"{s.shares} shares{earnings_note}"
            )
    else:
        reason = brief.no_setup_reason or "No qualifying setups"
        sections.append(f"  None — {reason}")

    # Upcoming earnings section
    if brief.earnings_calendar:
        sections.append("\n== UPCOMING EARNINGS ==")
        for ev in sorted(brief.earnings_calendar, key=lambda e: e.days_to_result or 99)[:8]:
            sections.append(
                f"  {ev.ticker} ({ev.company_name or ev.ticker}): "
                f"{ev.result_date} in {ev.days_to_result} days "
                f"[⚠️ {ev.risk_level.value}]"
            )

    # Portfolio section
    sections.append("\n== PORTFOLIO ==")
    sections.append(f"Capital: ₹{brief.total_capital or 50000}")
    sections.append(f"Available: ₹{brief.available_capital or brief.total_capital or 50000}")
    if brief.open_positions:
        for p in brief.open_positions:
            pnl = f", P&L ₹{p.pnl_rupees}" if p.pnl_rupees else ""
            sections.append(
                f"  {p.ticker}: {p.shares} shares @ ₹{p.entry_price}{pnl}"
            )
    else:
        sections.append("  No open positions")

    # Memory context section
    if memory_ctx and hasattr(memory_ctx, "text") and memory_ctx.text:
        sections.append("\n== VIJAY'S CONTEXT ==")
        # Truncate to ~300 tokens worth
        context = memory_ctx.text[:900]
        sections.append(context)

    # Lesson section
    if brief.lesson_of_day:
        concept = brief.lesson_of_day.concept.replace("_", " ").title()
        sections.append(f"\n== TODAY'S LESSON ==")
        sections.append(f"Concept: {concept}")
        sections.append(f"Summary: {brief.lesson_of_day.summary}")

    # Instruction
    sections.append("\n== INSTRUCTION ==")
    sections.append(
        "Write a warm, personal Telegram message for Vijay. "
        "Include all sections: greeting, market mood, portfolio, "
        "setups (or why none), lesson, and closing. "
        "Use HTML formatting (<b>bold</b> for headers). "
        "Keep under 800 tokens. Make it readable on mobile."
    )

    return "\n".join(sections)


def _get_anthropic_key() -> str:
    """Get Anthropic API key from environment."""
    import os
    return os.getenv("ANTHROPIC_API_KEY", "")
