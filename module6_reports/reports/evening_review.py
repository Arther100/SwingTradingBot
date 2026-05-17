"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
reports/evening_review.py — Evening review generator

Runs at 4:30 PM IST after market close.
Fetches closing data, updates position P&L, calls Claude
for a concise evening summary.

One Claude call — 1500 tokens max (800 input + 700 output).

Usage:
    from module6_reports.reports.evening_review import generate_evening_review

    review = await generate_evening_review(user_id="XCU700")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Optional

from zoneinfo import ZoneInfo

from module6_reports.config import (
    CLAUDE_MODEL,
    DEFAULT_TICKERS,
    EVENING_REVIEW_SYSTEM_PROMPT,
    EVENING_REVIEW_TOKEN_BUDGET,
    IST,
    LESSON_CONCEPTS,
    LESSON_SUMMARIES,
    REPORT_MAX_STOCKS,
)
from module6_reports.models import (
    DeliveryStatus,
    EveningReview,
    PositionSummary,
    ReportType,
)

logger = logging.getLogger("swing_advisor.evening_review")


async def generate_evening_review(
    user_id: str = "XCU700",
    tickers: list[str] | None = None,
    skip_claude: bool = False,
) -> EveningReview:
    """Generate the evening review after market close.

    Pipeline:
      1. M5 — Load user profile + open positions
      2. M1 — Fetch closing market data
      3. Build top gainers/losers from stock data
      4. Calculate day P&L for open positions
      5. Get today's lesson recap
      6. Claude — Generate telegram_text (one call, 1500 tokens)

    Args:
        user_id: User identifier (default: XCU700 for Vijay).
        tickers: Override default ticker list. None = use DEFAULT_TICKERS.
        skip_claude: If True, skip Claude call (use template formatting).

    Returns:
        EveningReview with all fields populated.
    """
    start_ms = time.time()
    tickers = tickers or DEFAULT_TICKERS

    review = EveningReview(
        report_type=ReportType.EVENING_REVIEW,
        user_id=user_id,
        delivery_status=DeliveryStatus.PENDING,
    )

    try:
        # ── Step 1: Load user profile + positions ──
        profile, open_positions = _load_user_data(user_id)
        if profile:
            logger.info(
                f"[EveningReview] Profile loaded: {profile.display_name}"
            )

        # ── Step 2: Fetch closing market data ──
        market_data = await _fetch_closing_data(tickers)
        if market_data:
            review.nifty_close = market_data.nifty50_value
            review.nifty_change_pct = market_data.nifty50_change_pct
            review.sensex_close = market_data.sensex_value
            review.sensex_change_pct = market_data.sensex_change_pct
            review.india_vix = Decimal(str(market_data.india_vix))
            review.vix_signal = market_data.vix_signal.value
            logger.info(
                f"[EveningReview] Close: Nifty={market_data.nifty50_value}, "
                f"VIX={market_data.india_vix}"
            )

        # ── Step 3: Top gainers/losers from watchlist ──
        if market_data and market_data.stocks:
            gainers, losers = _get_top_movers(market_data.stocks)
            review.top_gainers = gainers
            review.top_losers = losers

        # ── Step 4: Update positions with closing prices ──
        if open_positions:
            closing_prices = _get_closing_prices(market_data)
            review.open_positions = _update_position_pnl(
                open_positions, closing_prices
            )
            review.day_pnl = _calc_day_pnl(review.open_positions)
            review.total_pnl = _calc_total_pnl(review.open_positions)
            logger.info(
                f"[EveningReview] Positions: {len(review.open_positions)}, "
                f"Day P&L: ₹{review.day_pnl}"
            )

        # ── Step 5: Lesson recap ──
        review.lesson_recap = _get_lesson_recap()

        # ── Step 6: Claude telegram_text ──
        if not skip_claude:
            telegram_text = await _generate_telegram_text(review)
            if telegram_text:
                review.telegram_text = telegram_text

    except Exception as e:
        logger.error(f"[EveningReview] Pipeline error: {e}", exc_info=True)
        review.error = str(e)

    elapsed_ms = int((time.time() - start_ms) * 1000)
    review.generation_time_ms = elapsed_ms
    logger.info(f"[EveningReview] Generated in {elapsed_ms}ms")

    return review


# ═══════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════


def _load_user_data(user_id: str):
    """Load user profile and open positions from M5.

    Returns (UserProfile | None, list[PositionSummary]).
    """
    try:
        from module5_memory.engine import memory_engine

        profile = memory_engine.get_user_profile(user_id)

        # Get open trades
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
        return profile, positions
    except Exception as e:
        logger.warning(f"[EveningReview] M5 load failed: {e}")
        return None, []


async def _fetch_closing_data(tickers: list[str]):
    """Fetch closing market data from M1.

    Returns MarketData or None on failure.
    """
    try:
        from module1_data_layer.models import DataFetchConfig
        from module1_data_layer.pipeline import run_data_pipeline

        config = DataFetchConfig(
            tickers=tickers,
            max_stocks=REPORT_MAX_STOCKS,
            max_news=0,
            max_events=0,
        )
        market_data = await run_data_pipeline(
            tickers=tickers,
            config=config,
        )
        return market_data
    except Exception as e:
        logger.error(f"[EveningReview] M1 closing data failed: {e}")
        return None


def _get_top_movers(stocks: list) -> tuple[list[str], list[str]]:
    """Extract top 3 gainers and top 3 losers from stock data.

    Returns (gainers_list, losers_list) as formatted strings.
    """
    sorted_stocks = sorted(
        stocks,
        key=lambda s: s.change_pct,
        reverse=True,
    )

    gainers = []
    for s in sorted_stocks[:3]:
        if s.change_pct > 0:
            gainers.append(
                f"{s.ticker} +{s.change_pct:.2f}% (₹{s.current_price})"
            )

    losers = []
    for s in reversed(sorted_stocks):
        if s.change_pct < 0 and len(losers) < 3:
            losers.append(
                f"{s.ticker} {s.change_pct:.2f}% (₹{s.current_price})"
            )

    return gainers, losers


def _get_closing_prices(market_data) -> dict[str, Decimal]:
    """Build ticker → closing price map from market data.

    Returns dict of {ticker: Decimal(price)}.
    """
    if not market_data or not market_data.stocks:
        return {}

    return {
        s.ticker: Decimal(str(s.current_price))
        for s in market_data.stocks
    }


def _update_position_pnl(
    positions: list[PositionSummary],
    closing_prices: dict[str, Decimal],
) -> list[PositionSummary]:
    """Update position current_price and P&L from closing prices.

    Returns updated list of PositionSummary.
    """
    updated = []
    for pos in positions:
        closing = closing_prices.get(pos.ticker)
        if closing:
            pos.current_price = closing
            pos.pnl_rupees = (closing - pos.entry_price) * pos.shares
            if pos.entry_price > 0:
                pos.pnl_pct = (
                    (closing - pos.entry_price) / pos.entry_price * 100
                )
        updated.append(pos)
    return updated


def _calc_day_pnl(positions: list[PositionSummary]) -> Optional[Decimal]:
    """Sum P&L across all positions for the day.

    Note: For accurate day P&L we'd need yesterday's close.
    This returns total unrealised P&L from entry as approximation.
    """
    if not positions:
        return None

    total = Decimal("0")
    for pos in positions:
        if pos.pnl_rupees is not None:
            total += pos.pnl_rupees
    return total


def _calc_total_pnl(positions: list[PositionSummary]) -> Optional[Decimal]:
    """Total unrealised P&L across all open positions."""
    return _calc_day_pnl(positions)


def _get_lesson_recap() -> str:
    """Get today's lesson concept for recap.

    Same rotation as morning brief — ensures consistency.
    """
    day_of_year = datetime.now(IST).timetuple().tm_yday
    index = day_of_year % len(LESSON_CONCEPTS)
    concept = LESSON_CONCEPTS[index]
    summary = LESSON_SUMMARIES.get(concept, "")

    concept_display = concept.replace("_", " ").title()
    return f"Today's lesson was {concept_display}. {summary}"


# ═══════════════════════════════════════════════════════════
# CLAUDE TELEGRAM TEXT GENERATION
# ═══════════════════════════════════════════════════════════


async def _generate_telegram_text(review: EveningReview) -> Optional[str]:
    """Call Claude to generate the telegram_text for evening review.

    One call, 1500 tokens max (800 input + 700 output).
    """
    try:
        import httpx

        api_key = _get_anthropic_key()
        if not api_key:
            logger.warning("[EveningReview] No ANTHROPIC_API_KEY — skipping Claude")
            return None

        user_prompt = _build_claude_prompt(review)

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
                    "max_tokens": EVENING_REVIEW_TOKEN_BUDGET["output_budget"],
                    "system": EVENING_REVIEW_SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=30.0,
            )

            data = response.json()

            if response.status_code != 200:
                error = data.get("error", {}).get("message", "Unknown error")
                logger.error(f"[EveningReview] Claude API error: {error}")
                return None

            content_blocks = data.get("content", [])
            text_parts = [
                block["text"]
                for block in content_blocks
                if block.get("type") == "text"
            ]
            telegram_text = "\n".join(text_parts)

            usage = data.get("usage", {})
            review.tokens_used = (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            )
            logger.info(
                f"[EveningReview] Claude: {usage.get('input_tokens', 0)} in + "
                f"{usage.get('output_tokens', 0)} out = {review.tokens_used} tokens"
            )

            return telegram_text

    except Exception as e:
        logger.error(f"[EveningReview] Claude call failed: {e}")
        return None


def _build_claude_prompt(review: EveningReview) -> str:
    """Build Claude prompt for evening review.

    Structured sections — total ~800 input tokens.
    """
    sections: list[str] = []

    # Market close
    sections.append("== MARKET CLOSE ==")
    if review.nifty_close:
        sections.append(
            f"Nifty: {review.nifty_close} ({review.nifty_change_pct:+.2f}%)"
        )
    if review.sensex_close:
        sections.append(
            f"Sensex: {review.sensex_close} ({review.sensex_change_pct:+.2f}%)"
        )
    if review.india_vix:
        sections.append(f"VIX: {review.india_vix} ({review.vix_signal})")
    if review.market_mood:
        sections.append(f"Mood: {review.market_mood}")

    # Top movers
    if review.top_gainers or review.top_losers:
        sections.append("\n== TOP MOVERS ==")
        for g in review.top_gainers:
            sections.append(f"  📈 {g}")
        for l in review.top_losers:
            sections.append(f"  📉 {l}")

    # Positions
    sections.append("\n== PORTFOLIO ==")
    if review.open_positions:
        for p in review.open_positions:
            pnl = ""
            if p.pnl_rupees is not None:
                sign = "+" if p.pnl_rupees >= 0 else ""
                pnl = f", P&L {sign}₹{p.pnl_rupees}"
            sections.append(
                f"  {p.ticker}: {p.shares} shares @ ₹{p.entry_price}"
                f" → ₹{p.current_price or '?'}{pnl}"
            )
        if review.day_pnl is not None:
            sign = "+" if review.day_pnl >= 0 else ""
            sections.append(f"Day P&L: {sign}₹{review.day_pnl}")
    else:
        sections.append("  No open positions")

    # Lesson recap
    if review.lesson_recap:
        sections.append(f"\n== LESSON RECAP ==\n{review.lesson_recap}")

    # Instruction
    sections.append("\n== INSTRUCTION ==")
    sections.append(
        "Write a warm evening review for Vijay. "
        "Summarise what happened today, how positions performed, "
        "recap today's lesson briefly, and give a short outlook for tomorrow. "
        "Use HTML formatting (<b>bold</b>). Keep under 700 tokens."
    )

    return "\n".join(sections)


def _get_anthropic_key() -> str:
    """Get Anthropic API key from environment."""
    import os
    return os.getenv("ANTHROPIC_API_KEY", "")
