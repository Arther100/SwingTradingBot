"""
SwingAdvisorBot — Module 4: Trade Setup Generator
config.py — Configuration and Claude prompts for setup generation

This file contains:
  → Claude prompt templates for setup reasoning
  → Token budget configuration per setup
  → Setup generation parameters
  → Company name mapping for Nifty 50 stocks
  → Lesson rotation for educational content

All Claude prompts are defined here — not scattered across files.
Token budgets measured and documented.

Design decisions:
  - Prompts are constants, not built at runtime
  - Token budgets are conservative estimates
  - Company names hardcoded (Nifty 50 is stable)
  - Lessons rotate — no repeat in same package
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# ─────────────────────────────────────────────────────────────
# Claude API Settings for M4
# ─────────────────────────────────────────────────────────────

DEFAULT_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-5")
DEFAULT_TEMPERATURE = 0.4  # Slightly higher than M2 for creative reasoning


# ─────────────────────────────────────────────────────────────
# Token Budgets — Per Setup
# ─────────────────────────────────────────────────────────────

class SetupTokenBudget(BaseModel):
    """Token budget for a single setup reasoning call."""

    system_prompt_tokens: int = Field(default=380, description="Fixed system prompt")
    stock_data_tokens: int = Field(default=200, description="Trimmed M1 stock data")
    analysis_summary_tokens: int = Field(default=150, description="Trimmed M2 analysis")
    risk_report_tokens: int = Field(default=200, description="Trimmed M3 risk report")
    setup_instruction_tokens: int = Field(default=100, description="Fixed instruction")
    total_input_budget: int = Field(default=1030, description="Max input tokens per setup")
    output_budget: int = Field(default=600, description="Max output tokens per setup")
    total_per_setup: int = Field(default=1630, description="Total tokens per setup call")


# ─────────────────────────────────────────────────────────────
# Setup Generation Config
# ─────────────────────────────────────────────────────────────

class SetupConfig(BaseModel):
    """Configuration for setup generation."""

    max_setups: int = Field(default=5, description="Maximum setups to generate")
    min_confidence: float = Field(default=6.0, description="Minimum confidence score")
    max_candidates: int = Field(default=10, description="Max stocks to evaluate")
    cache_ttl_minutes: int = Field(default=30, description="Cache TTL for setup reasoning")
    price_change_threshold_pct: float = Field(
        default=1.0,
        description="Regenerate if price moved > this % since last generation",
    )
    default_hold_days_min: int = Field(default=3, description="Min holding period")
    default_hold_days_max: int = Field(default=10, description="Max holding period")
    market_open_hour: int = Field(default=9, description="Market open hour IST")
    market_open_minute: int = Field(default=15, description="Market open minute IST")
    market_close_hour: int = Field(default=15, description="Market close hour IST")
    market_close_minute: int = Field(default=30, description="Market close minute IST")


@lru_cache(maxsize=1)
def get_setup_config() -> SetupConfig:
    """Get setup configuration singleton."""
    return SetupConfig()


@lru_cache(maxsize=1)
def get_token_budget() -> SetupTokenBudget:
    """Get token budget singleton."""
    return SetupTokenBudget()


# ─────────────────────────────────────────────────────────────
# Claude Prompts — Setup Reasoning
# ─────────────────────────────────────────────────────────────

SETUP_REASONING_SYSTEM_PROMPT = """You are a senior finance advisor with 20+ years of experience in Indian NSE swing trading.

Generate reasoning for a swing trade setup.
Your reasoning must be grounded ONLY in the market data provided — never use training knowledge for current prices or recent events.

Respond in valid JSON only. No text outside JSON.
Start with { end with }. No markdown. No backticks.

Required JSON structure:
{
  "setup_reasoning": "2-3 sentences why this stock why now. Reference specific data points.",
  "entry_trigger": "Specific condition to enter. Price level + confirmation signal.",
  "exit_strategy": "When to take profit. Partial booking instructions.",
  "risk_warning": "What invalidates this setup. Specific price level to exit immediately.",
  "macro_context": "How current macro environment affects this specific stock.",
  "lesson": "One trading concept this setup demonstrates. Explain simply."
}

Keep each field under 100 characters.
Total response under 500 tokens.
Speak directly to the user by name."""

SETUP_REASONING_USER_TEMPLATE = """User: {user_name}
Capital: ₹{capital}
Risk tolerance: {risk_tolerance}

Stock data:
{stock_summary}

Market analysis summary:
{market_analysis_summary}

Risk report:
{risk_report_summary}

Technical levels:
Entry zone: ₹{entry_low} - ₹{entry_high}
Target: ₹{target}
Stop loss: ₹{stop_loss}
Risk/Reward: {risk_reward}
Position: {shares} shares (₹{position_rupees})

Generate complete setup reasoning for this trade.
Base ALL price references on the data above."""


# ─────────────────────────────────────────────────────────────
# Company Names — Nifty 50 stocks
# ─────────────────────────────────────────────────────────────

COMPANY_NAMES: dict[str, str] = {
    "HDFCBANK": "HDFC Bank Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "SBIN": "State Bank of India",
    "AXISBANK": "Axis Bank Limited",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys Limited",
    "WIPRO": "Wipro Limited",
    "HCLTECH": "HCL Technologies",
    "TECHM": "Tech Mahindra Limited",
    "RELIANCE": "Reliance Industries Limited",
    "ONGC": "Oil & Natural Gas Corporation",
    "BPCL": "Bharat Petroleum Corp.",
    "IOC": "Indian Oil Corporation",
    "NTPC": "NTPC Limited",
    "POWERGRID": "Power Grid Corp. of India",
    "BHARTIARTL": "Bharti Airtel Limited",
    "LT": "Larsen & Toubro Limited",
    "TATAMOTORS": "Tata Motors Limited",
    "MARUTI": "Maruti Suzuki India",
    "HINDUNILVR": "Hindustan Unilever",
    "ITC": "ITC Limited",
    "BAJFINANCE": "Bajaj Finance Limited",
    "BAJAJFINSV": "Bajaj Finserv Limited",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports & SEZ",
    "TITAN": "Titan Company Limited",
    "ASIANPAINT": "Asian Paints Limited",
    "ULTRACEMCO": "UltraTech Cement",
    "GRASIM": "Grasim Industries",
    "SUNPHARMA": "Sun Pharma Industries",
    "DRREDDY": "Dr. Reddy's Laboratories",
    "CIPLA": "Cipla Limited",
    "DIVISLAB": "Divi's Laboratories",
    "APOLLOHOSP": "Apollo Hospitals",
    "EICHERMOT": "Eicher Motors Limited",
    "HEROMOTOCO": "Hero MotoCorp Limited",
    "TATASTEEL": "Tata Steel Limited",
    "JSWSTEEL": "JSW Steel Limited",
    "HINDALCO": "Hindalco Industries",
    "COALINDIA": "Coal India Limited",
    "NESTLEIND": "Nestlé India Limited",
    "BRITANNIA": "Britannia Industries",
    "M&M": "Mahindra & Mahindra",
    "INDUSINDBK": "IndusInd Bank Limited",
    "BAJAJ-AUTO": "Bajaj Auto Limited",
    "TATACONSUM": "Tata Consumer Products",
    "SBILIFE": "SBI Life Insurance",
    "HDFCLIFE": "HDFC Life Insurance",
    "SHRIRAMFIN": "Shriram Finance Limited",
}


def get_company_name(ticker: str) -> str:
    """Get full company name for a ticker."""
    return COMPANY_NAMES.get(ticker.upper(), ticker)


# ─────────────────────────────────────────────────────────────
# Advisor Flag Priority — for stock screening
# ─────────────────────────────────────────────────────────────

# Higher priority flags are preferred for setup generation.
# Stocks with these flags are evaluated first.
FLAG_PRIORITY: dict[str, int] = {
    "BREAKOUT_WATCH": 1,
    "UNUSUAL_ACTIVITY": 2,
    "MOMENTUM_BUILDING": 3,
    "ACCUMULATION_ZONE": 4,
    "CONSOLIDATION": 5,
}

# Flags that disqualify a stock from setup generation.
SKIP_FLAGS: set[str] = {
    "SELLING_PRESSURE",
    "DISTRIBUTION_ZONE",
    "NEUTRAL",
}


# ─────────────────────────────────────────────────────────────
# Lesson Rotation — educational content
# ─────────────────────────────────────────────────────────────

SETUP_LESSONS: list[dict[str, str]] = [
    {
        "topic": "accumulation_at_support",
        "template": (
            "This setup demonstrates 'accumulation at support' — "
            "institutions buy quietly near support levels before "
            "a potential reversal. Watch for above-average volume "
            "as confirmation."
        ),
    },
    {
        "topic": "breakout_confirmation",
        "template": (
            "This is a 'breakout confirmation' setup — the stock "
            "is testing a key resistance level with strong volume. "
            "Wait for a close above resistance before entering."
        ),
    },
    {
        "topic": "momentum_continuation",
        "template": (
            "This setup shows 'momentum continuation' — the stock "
            "is already in an uptrend and pulling back to support. "
            "Enter on the pullback, not at the highs."
        ),
    },
    {
        "topic": "risk_reward_discipline",
        "template": (
            "This trade demonstrates risk/reward discipline — "
            "never enter a trade where potential loss exceeds "
            "potential gain. A minimum 1:2 R/R protects your "
            "capital over many trades."
        ),
    },
    {
        "topic": "position_sizing",
        "template": (
            "This setup shows proper position sizing — the 2% rule "
            "ensures no single trade can seriously damage your "
            "portfolio. Risk small, let winners run."
        ),
    },
    {
        "topic": "sector_rotation",
        "template": (
            "This demonstrates 'sector rotation' — money flows "
            "from one sector to another. Identify which sectors "
            "are gaining institutional interest and follow the flow."
        ),
    },
    {
        "topic": "volume_precedes_price",
        "template": (
            "Volume precedes price — unusual volume often signals "
            "a move before it happens. This stock's volume pattern "
            "suggests smart money is positioning ahead of a move."
        ),
    },
    {
        "topic": "stop_loss_discipline",
        "template": (
            "Your stop loss is your safety net. Once set, never "
            "move it further away. This setup has a clear "
            "invalidation level — honour it without question."
        ),
    },
]


def get_lesson_for_index(index: int) -> dict[str, str]:
    """Get a lesson by rotating through the list.

    Uses modulo to cycle through lessons so no
    two setups in the same package get the same lesson.
    """
    return SETUP_LESSONS[index % len(SETUP_LESSONS)]
