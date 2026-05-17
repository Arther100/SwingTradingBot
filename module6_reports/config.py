"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
config.py — Configuration for scheduling, Telegram, and report generation

All times in IST (Asia/Kolkata).
All token budgets aligned with M2 conventions.

Config groups:
  ScheduleConfig   — Job times (kite refresh, morning brief, etc.)
  TelegramConfig   — Bot token, chat ID, message limits
  ReportConfig     — Token budgets, Claude model, retry settings
  LessonConfig     — Daily concept rotation
"""

from __future__ import annotations

import os

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ═══════════════════════════════════════════════════════════
# SCHEDULE TIMES (all IST, weekdays only unless noted)
# ═══════════════════════════════════════════════════════════

# Kite token refresh — runs daily (including weekends for validation)
KITE_REFRESH_HOUR = 5
KITE_REFRESH_MINUTE = 50

# Morning brief — weekdays only
MORNING_BRIEF_HOUR = 8
MORNING_BRIEF_MINUTE = 50

# Watchlist monitoring — every 3 minutes during market hours
WATCHLIST_INTERVAL_MINUTES = 3
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Evening review — weekdays only
EVENING_REVIEW_HOUR = 16
EVENING_REVIEW_MINUTE = 30

# Weekly summary — Saturday only
WEEKLY_SUMMARY_HOUR = 10
WEEKLY_SUMMARY_MINUTE = 0
WEEKLY_SUMMARY_DAY = "sat"

# Misfire grace time (seconds) — how late a job can run after missed time
MISFIRE_GRACE_MORNING = 600     # 10 minutes
MISFIRE_GRACE_EVENING = 600     # 10 minutes
MISFIRE_GRACE_KITE = 300        # 5 minutes
MISFIRE_GRACE_WEEKLY = 1800     # 30 minutes

# ═══════════════════════════════════════════════════════════
# TELEGRAM CONFIG
# ═══════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_SEND_TIMEOUT = 10.0       # seconds
TELEGRAM_RETRY_ATTEMPTS = 2
TELEGRAM_RETRY_DELAY = 3.0         # seconds between retries
TELEGRAM_API_BASE = "https://api.telegram.org/bot"

# ═══════════════════════════════════════════════════════════
# REPORT GENERATION CONFIG
# ═══════════════════════════════════════════════════════════

# Claude model for report generation
CLAUDE_MODEL = "claude-opus-4-5-20250514"

# Token budgets (aligned with Section 6 of M6 prompt)
MORNING_BRIEF_TOKEN_BUDGET = {
    "system_prompt": 380,
    "market_data": 400,
    "analysis": 200,
    "risk_summary": 150,
    "setups": 300,
    "memory_context": 300,
    "instruction": 100,
    "total_input": 1830,
    "output_budget": 800,
    "grand_total": 2630,
}

EVENING_REVIEW_TOKEN_BUDGET = {
    "total_input": 800,
    "output_budget": 700,
    "grand_total": 1500,
}

WEEKLY_SUMMARY_TOKEN_BUDGET = {
    "total_input": 1000,
    "output_budget": 800,
    "grand_total": 1800,
}

# Max setups in morning brief
MAX_SETUPS_IN_BRIEF = 5
MIN_SETUP_CONFIDENCE = 6.0

# ═══════════════════════════════════════════════════════════
# M1 DATA FETCH CONFIG FOR REPORTS
# ═══════════════════════════════════════════════════════════

# Default tickers for morning brief (can be overridden by watchlist)
DEFAULT_TICKERS = [
    "HDFCBANK",
    "RELIANCE",
    "TCS",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "BAJFINANCE",
    "LT",
    "ITC",
    "AXISBANK",
]

# M1 fetch limits for reports (lighter than full pipeline)
REPORT_MAX_STOCKS = 10
REPORT_MAX_NEWS = 3
REPORT_MAX_EVENTS = 2

# ═══════════════════════════════════════════════════════════
# LESSON ROTATION CONFIG
# ═══════════════════════════════════════════════════════════

# All available concepts for daily lessons
# Order matters — taught sequentially, then rotated
LESSON_CONCEPTS = [
    "swing_trading",
    "stop_loss",
    "risk_reward_ratio",
    "position_sizing",
    "india_vix",
    "52_week_high_low",
    "volume_analysis",
    "sector_rotation",
    "trailing_stop_loss",
    "partial_profit_booking",
]

# Short summaries for each concept (used when Claude is unavailable)
LESSON_SUMMARIES = {
    "swing_trading": (
        "Swing trading captures moves over 3-10 days. "
        "Use support/resistance levels and volume confirmation. "
        "Less screen time than intraday, but requires disciplined stop losses."
    ),
    "stop_loss": (
        "A stop loss is your exit price for a losing trade. "
        "Place it at the nearest support or 3-5% below entry. "
        "Never move it further away — only trail in the direction of profit."
    ),
    "risk_reward_ratio": (
        "Risk/reward compares potential loss to potential gain. "
        "Minimum 1:2 required — risk ₹100 to make ₹200. "
        "Higher R/R like 1:3 dramatically improves long-term results."
    ),
    "position_sizing": (
        "The 2% rule: never risk more than 2% of capital per trade. "
        "For ₹50,000 capital, max risk is ₹1,000 per trade. "
        "Divide by per-share risk to get the number of shares."
    ),
    "india_vix": (
        "India VIX measures expected 30-day volatility. "
        "Below 14 = low fear (safe). 14-20 = moderate. Above 20 = high fear. "
        "The bot blocks new trades when VIX exceeds your tolerance limit."
    ),
    "52_week_high_low": (
        "52-week range shows a stock's yearly price boundaries. "
        "Near the high with volume = potential breakout. "
        "Near the low with volume = possible accumulation."
    ),
    "volume_analysis": (
        "Volume confirms price moves. Breakouts need 1.3x average volume. "
        "3x spikes signal institutional activity. "
        "Low volume rallies are unreliable and often reverse."
    ),
    "sector_rotation": (
        "Different sectors lead in different market phases. "
        "Bull markets favour Banking and Auto. Corrections favour Pharma and FMCG. "
        "The bot caps sector exposure at 50% to prevent concentration risk."
    ),
    "trailing_stop_loss": (
        "A trailing stop moves up as price rises, locking in profit. "
        "Trail to previous day's low or a fixed % below the peak. "
        "This lets winners run while protecting gains automatically."
    ),
    "partial_profit_booking": (
        "Book 50% at first target, trail the rest to final target. "
        "Move stop loss to breakeven on remaining position. "
        "This guarantees profit while keeping upside exposure."
    ),
}

# ═══════════════════════════════════════════════════════════
# ERROR HANDLING CONFIG
# ═══════════════════════════════════════════════════════════

# Retry schedule for failed morning brief (minutes after first attempt)
MORNING_BRIEF_RETRY_MINUTES = [10, 30]

# Max retries for any single report
MAX_REPORT_RETRIES = 2

# ═══════════════════════════════════════════════════════════
# CLAUDE SYSTEM PROMPT FOR MORNING BRIEF
# ═══════════════════════════════════════════════════════════

MORNING_BRIEF_SYSTEM_PROMPT = """You are a senior finance advisor with 20+ years of NSE experience.
You are writing a personal morning brief for Vijay — your retail investor client.

RULES:
- Address Vijay by name
- Be warm but professional — like a WhatsApp message from a trusted advisor
- Every number must come from the data provided — never make up prices
- Keep the tone encouraging but realistic
- If VIX is high or no setups found, explain WHY and what to do instead
- Include the lesson of the day naturally — teach while you advise
- Keep total response under 800 tokens
- Use simple language — Vijay is learning

FORMAT:
Return a single Telegram-ready message using HTML formatting.
Use <b>bold</b> for headers and important numbers.
Use emojis sparingly but effectively.
Keep it readable on mobile — short paragraphs, clear sections."""

EVENING_REVIEW_SYSTEM_PROMPT = """You are a senior finance advisor reviewing today's market for Vijay.
Write a concise evening summary — what happened, how positions performed,
and a brief outlook for tomorrow.

RULES:
- Address Vijay by name
- Be honest about losses — every loss is a lesson
- Highlight what went right and what to watch tomorrow
- Recap today's lesson briefly
- Keep total response under 700 tokens
- Use HTML formatting for Telegram"""

WEEKLY_SUMMARY_SYSTEM_PROMPT = """You are a senior finance advisor writing Vijay's weekly review.
Summarise the week's performance, lessons, and outlook for next week.

RULES:
- Address Vijay by name
- Celebrate wins, be constructive about losses
- Highlight the best trade and worst trade
- List lessons covered this week
- Give honest outlook for next week
- Keep total response under 800 tokens
- Use HTML formatting for Telegram"""
