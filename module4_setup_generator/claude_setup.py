"""
SwingAdvisorBot — Module 4: Trade Setup Generator
claude_setup.py — Claude API client for setup reasoning

This file handles the Claude API call for generating setup
reasoning fields. It reuses the same httpx pattern as M2's
claude_client.py but is focused on per-setup reasoning.

Token budget per setup: ~1030 input + 600 output = 1630 total.

Caching: same stock + same day → cached for 30 minutes.
M3 check runs BEFORE this — rejected stocks never reach here.

The function returns a dict with 6 reasoning fields:
  setup_reasoning, entry_trigger, exit_strategy,
  risk_warning, macro_context, lesson
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from module1_data_layer.models import StockData
from module2_analysis_engine.models import MarketAnalysis
from module3_risk_engine.models import RiskReport
from module4_setup_generator.config import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_TEMPERATURE,
    SETUP_REASONING_SYSTEM_PROMPT,
    SETUP_REASONING_USER_TEMPLATE,
    get_company_name,
    get_setup_config,
    get_token_budget,
)
from module4_setup_generator.models import SetupFilter
from module4_setup_generator.technical.level_calculator import TechnicalLevels

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.claude_setup")

# ─────────────────────────────────────────────────────────────
# Simple in-memory cache for setup reasoning
# ─────────────────────────────────────────────────────────────

_reasoning_cache: dict[str, tuple[dict[str, str], float]] = {}


def _cache_key(ticker: str) -> str:
    """Generate cache key: ticker + date."""
    today = datetime.now(IST).strftime("%Y-%m-%d")
    raw = f"{ticker}:{today}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(ticker: str) -> Optional[dict[str, str]]:
    """Get cached reasoning if still fresh."""
    config = get_setup_config()
    key = _cache_key(ticker)
    if key in _reasoning_cache:
        result, cached_at = _reasoning_cache[key]
        age_minutes = (time.time() - cached_at) / 60
        if age_minutes < config.cache_ttl_minutes:
            logger.info(f"[ClaudeSetup] Cache hit for {ticker} ({age_minutes:.0f}m old)")
            return result
        else:
            del _reasoning_cache[key]
    return None


def _set_cached(ticker: str, result: dict[str, str]) -> None:
    """Cache reasoning result."""
    key = _cache_key(ticker)
    _reasoning_cache[key] = (result, time.time())


# ─────────────────────────────────────────────────────────────
# Stock summary builder (for prompt)
# ─────────────────────────────────────────────────────────────


def _build_stock_summary(stock: StockData) -> str:
    """Build trimmed stock summary for Claude prompt (~200 tokens)."""
    flag = stock.advisor_flag.value if stock.advisor_flag else "none"
    vol = stock.volume_signal.value if stock.volume_signal else "normal"

    return (
        f"Ticker: {stock.ticker}\n"
        f"Company: {get_company_name(stock.ticker)}\n"
        f"Sector: {stock.sector or 'Other'}\n"
        f"Price: ₹{stock.price}\n"
        f"Change: {stock.change_pct:+.2f}%\n"
        f"52w High: ₹{stock.high_52w}, Low: ₹{stock.low_52w}\n"
        f"Volume ratio: {stock.volume_ratio:.2f}x ({vol})\n"
        f"Advisor flag: {flag}"
    )


def _build_analysis_summary(analysis: MarketAnalysis) -> str:
    """Build trimmed market analysis summary (~150 tokens)."""
    mood = analysis.market_mood.value if hasattr(analysis.market_mood, "value") else str(analysis.market_mood)

    lines = [
        f"Market mood: {mood} (confidence: {analysis.mood_confidence:.0%})",
        f"Situation: {analysis.situation[:150]}",
    ]

    # Add top sector if available
    if analysis.sector_analyses:
        top = analysis.sector_analyses[0]
        lines.append(
            f"Top sector: {top.sector_name} ({top.sector_mood.value}, {top.change_pct:+.1f}%)"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main reasoning function
# ─────────────────────────────────────────────────────────────


def get_setup_reasoning(
    stock: StockData,
    levels: TechnicalLevels,
    risk_report: RiskReport,
    analysis: MarketAnalysis,
    setup_filter: SetupFilter,
) -> dict[str, str]:
    """Get Claude reasoning for a trade setup (synchronous wrapper).

    Checks cache first. If miss, calls Claude API synchronously.

    Args:
        stock: M1 StockData for this stock.
        levels: Calculated technical levels.
        risk_report: M3 RiskReport (must be APPROVED or REDUCE_SIZE).
        analysis: M2 MarketAnalysis.
        setup_filter: User preferences.

    Returns:
        Dict with keys: setup_reasoning, entry_trigger, exit_strategy,
        risk_warning, macro_context, lesson.
    """
    # Check cache
    cached = _get_cached(stock.ticker)
    if cached is not None:
        return cached

    # Build prompt
    user_message = _build_user_message(
        stock=stock,
        levels=levels,
        risk_report=risk_report,
        analysis=analysis,
        setup_filter=setup_filter,
    )

    # Call Claude
    try:
        raw_response = _call_claude_sync(
            system_prompt=SETUP_REASONING_SYSTEM_PROMPT,
            user_message=user_message,
        )
    except Exception as e:
        logger.error(f"[ClaudeSetup] API call failed for {stock.ticker}: {e}")
        return _fallback_reasoning(stock, levels)

    # Parse response
    result = _parse_reasoning_response(raw_response, stock.ticker)

    # Cache result
    _set_cached(stock.ticker, result)

    return result


def _build_user_message(
    stock: StockData,
    levels: TechnicalLevels,
    risk_report: RiskReport,
    analysis: MarketAnalysis,
    setup_filter: SetupFilter,
) -> str:
    """Build the user message from template."""
    position_rupees = Decimal(str(
        risk_report.position_size_shares * float(levels.entry_zone_low)
    )).quantize(Decimal("0.01"))

    return SETUP_REASONING_USER_TEMPLATE.format(
        user_name=setup_filter.display_name,
        capital=f"{setup_filter.capital:,.0f}",
        risk_tolerance=setup_filter.risk_tolerance,
        stock_summary=_build_stock_summary(stock),
        market_analysis_summary=_build_analysis_summary(analysis),
        risk_report_summary=risk_report.to_prompt_context(),
        entry_low=levels.entry_zone_low,
        entry_high=levels.entry_zone_high,
        target=levels.target_price,
        stop_loss=levels.stop_loss,
        risk_reward=levels.risk_reward_ratio,
        shares=risk_report.position_size_shares,
        position_rupees=position_rupees,
    )


def _call_claude_sync(
    system_prompt: str,
    user_message: str,
) -> str:
    """Make synchronous Claude API call via httpx.

    Uses the same Anthropic Messages API as M2's claude_client.py
    but synchronous (no async needed for per-setup calls).

    Returns:
        Raw response text from Claude.

    Raises:
        RuntimeError: On API errors.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    budget = get_token_budget()

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": DEFAULT_CLAUDE_MODEL,
        "max_tokens": budget.output_budget,
        "temperature": DEFAULT_TEMPERATURE,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }

    start = time.monotonic()

    with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
        response = client.post(url, headers=headers, json=body)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    if response.status_code != 200:
        error_text = response.text[:300]
        logger.error(
            f"[ClaudeSetup] API error {response.status_code}: {error_text}"
        )
        raise RuntimeError(
            f"Claude API error {response.status_code}: {error_text}"
        )

    data = response.json()
    usage = data.get("usage", {})

    logger.info(
        f"[ClaudeSetup] API call: {elapsed_ms}ms, "
        f"input={usage.get('input_tokens', '?')}, "
        f"output={usage.get('output_tokens', '?')}"
    )

    # Extract text from response
    content = data.get("content", [])
    if not content:
        raise RuntimeError("Claude returned empty content")

    return content[0].get("text", "")


def _parse_reasoning_response(
    raw: str, ticker: str
) -> dict[str, str]:
    """Parse Claude's JSON response into reasoning fields.

    Expected JSON:
    {
        "setup_reasoning": "...",
        "entry_trigger": "...",
        "exit_strategy": "...",
        "risk_warning": "...",
        "macro_context": "...",
        "lesson": "..."
    }
    """
    expected_keys = [
        "setup_reasoning", "entry_trigger", "exit_strategy",
        "risk_warning", "macro_context", "lesson",
    ]

    try:
        # Strip any markdown backticks if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)

        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")

        result: dict[str, str] = {}
        for key in expected_keys:
            value = parsed.get(key, "")
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()[:200]  # Cap each field

        if not result.get("setup_reasoning"):
            logger.warning(
                f"[ClaudeSetup] {ticker}: Missing setup_reasoning in response"
            )

        return result

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"[ClaudeSetup] {ticker}: Failed to parse response: {e}. "
            f"Raw (first 200 chars): {raw[:200]}"
        )
        return {}


def _fallback_reasoning(
    stock: StockData, levels: TechnicalLevels
) -> dict[str, str]:
    """Generate minimal fallback reasoning when Claude is unavailable."""
    flag = stock.advisor_flag.value if stock.advisor_flag else "no signal"
    return {
        "setup_reasoning": (
            f"{stock.ticker} shows {flag} pattern at ₹{stock.price}. "
            f"Volume ratio {stock.volume_ratio:.1f}x supports the setup."
        ),
        "risk_warning": (
            f"Exit immediately if price drops below ₹{levels.stop_loss}."
        ),
    }


def clear_reasoning_cache() -> None:
    """Clear the in-memory reasoning cache."""
    _reasoning_cache.clear()
    logger.info("[ClaudeSetup] Reasoning cache cleared")
