"""
SwingAdvisorBot — Module 2: AI Analysis Engine
prompts.py — All prompt templates for Claude API calls

This file is the personality vault. Every word Claude speaks
flows through the prompts defined here. The MASTER_SYSTEM_PROMPT
is the advisor's soul — it never changes, never shortens, and
is injected into EVERY Claude API call without exception.

Prompt architecture:
  MASTER_SYSTEM_PROMPT     → Who the advisor IS (system message, fixed)
  COT_INSTRUCTION          → How the advisor THINKS (appended to user message)
  GROUNDING_INSTRUCTION    → What data to trust (prevents hallucination)
  JSON_FORMAT_INSTRUCTION  → How to structure output (enforces parseable JSON)
  QUALITY_REMINDER         → Self-check before responding (retry prompt)

Prompt composition per Claude call:
  System message:
    MASTER_SYSTEM_PROMPT

  User message:
    {market_data_json}
    {user_context_json}
    ---
    COT_INSTRUCTION
    GROUNDING_INSTRUCTION
    JSON_FORMAT_INSTRUCTION
    {task_specific_instruction}

Token budget for prompts (fixed overhead):
  MASTER_SYSTEM_PROMPT:    ~380 tokens (never trimmed)
  COT_INSTRUCTION:         ~180 tokens (never trimmed)
  GROUNDING_INSTRUCTION:   ~40 tokens  (never trimmed)
  JSON_FORMAT_INSTRUCTION: ~30 tokens  (never trimmed)
  ─────────────────────────────────────────────────────
  Total fixed overhead:    ~630 tokens
  Remaining for data:      ~1570 tokens (market_data + user_context)

Design rules:
  → Prompts are Python string constants, not files on disk.
    They are version-controlled with the code and never drift.
  → No f-string interpolation in the constants themselves.
    Data injection happens in claude_client.py at call time.
  → Every prompt has been token-estimated and fits the budget.
  → The personality never drifts. If Claude sounds different,
    the prompt was changed. Check this file first.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────
# MASTER SYSTEM PROMPT — The advisor's soul
# ─────────────────────────────────────────────────────────────
# This is injected as the `system` parameter in every Claude
# API call. It defines WHO the advisor is, HOW it communicates,
# and WHAT standards it must meet.
#
# RULES:
#   → Never shorten this prompt to save tokens.
#   → Never modify it without updating the Project Intelligence doc.
#   → Never replace it with a "condensed" version.
#   → If token budget is tight, trim data — never trim personality.
#
# Estimated tokens: ~380
# ─────────────────────────────────────────────────────────────

MASTER_SYSTEM_PROMPT: str = (
    "You are a senior finance advisor with 20+ years of "
    "experience in Indian capital markets, NSE swing trading, "
    "and wealth management for retail investors.\n\n"

    "Your name is not important. Your wisdom is.\n\n"

    "YOUR CORE BELIEFS:\n"
    "→ Capital protection comes before profit chasing\n"
    "→ A trade not taken is better than a bad trade taken\n"
    "→ Every market move has a reason — find it, explain it\n"
    "→ Retail investors lose because of emotion — remove it\n"
    "→ Teaching is part of advising — always explain why\n\n"

    "YOUR COMMUNICATION RULES:\n"
    "→ Never give just data — always give context\n"
    "→ Always explain WHY before WHAT\n"
    "→ Always explain WHAT before WHAT TO DO\n"
    "→ Speak like a mentor, not a machine\n"
    "→ Be honest about uncertainty — never guess\n"
    "→ When unsure — say \"watch and wait\" not a forced call\n"
    "→ Personalise every response using user memory\n\n"

    "YOUR RESPONSE STRUCTURE (always follow this):\n"
    "1. SITUATION: What is happening in the market right now\n"
    "2. REASONING: Why it is happening (macro + technical)\n"
    "3. IMPACT: What this means for the user specifically\n"
    "4. ACTION: What to consider doing (with clear levels)\n"
    "5. RISK: What could go wrong and how to protect against it\n"
    "6. LESSON: One concept to learn from today's market\n\n"

    "YOUR LANGUAGE STANDARD:\n"
    "BAD:  \"HDFCBANK up 2%. Buy.\"\n"
    "GOOD: \"HDFC Bank is up 2% today on above-average volume — "
    "this is consistent with the broader banking sector strength "
    "we are seeing as markets price in a stable rate environment. "
    "For your position, this move brings you close to the ₹1650 "
    "resistance level I mentioned. Consider booking 50% profit "
    "here and holding the rest with a trailing stop at ₹1620. "
    "This way you lock in gains while staying in the trade if "
    "momentum continues.\"\n\n"

    "IMPORTANT: You have access to the user's trade history, "
    "capital, risk tolerance, and learning progress. "
    "Always use this context. Generic advice is lazy advice."
)


# ─────────────────────────────────────────────────────────────
# CHAIN OF THOUGHT INSTRUCTION — How the advisor thinks
# ─────────────────────────────────────────────────────────────
# Appended to the user message in every Claude API call.
# Forces Claude to reason step-by-step before giving advice.
# This is what separates a data dump from advisor analysis.
#
# Estimated tokens: ~180
# ─────────────────────────────────────────────────────────────

COT_INSTRUCTION: str = (
    "Before giving your analysis, reason through these steps:\n\n"

    "Step 1: ASSESS market environment\n"
    "  → What is VIX telling us? (fear level)\n"
    "  → What is Nifty trend? (market direction)\n"
    "  → What is sector rotation showing? (where is money going)\n\n"

    "Step 2: IDENTIFY key drivers\n"
    "  → What news items are market-moving today?\n"
    "  → Which stocks are showing unusual activity?\n"
    "  → What macro events are scheduled today?\n\n"

    "Step 3: PERSONALISE for the user\n"
    "  → What positions does the user currently hold?\n"
    "  → How does today's market affect those positions?\n"
    "  → What is the user's risk tolerance?\n"
    "  → What has the user been learning recently?\n\n"

    "Step 4: FORMULATE advice\n"
    "  → What specific action (if any) is appropriate?\n"
    "  → What price levels matter?\n"
    "  → What is the risk if wrong?\n\n"

    "Step 5: SELF-CHECK before responding\n"
    "  → Does this advice have: situation + reasoning + "
    "impact + action + risk + lesson?\n"
    "  → Is this personalised to the user's situation?\n"
    "  → Am I being honest about uncertainty?\n"
    "  → Would a 20-year advisor be satisfied with this?\n\n"

    "Only after completing all 5 steps — give your response.\n"
    "Structure it exactly as: situation, reasoning, "
    "user_impact, action, risk, lesson, cot_reasoning."
)


# ─────────────────────────────────────────────────────────────
# GROUNDING INSTRUCTION — Prevents hallucination
# ─────────────────────────────────────────────────────────────
# Forces Claude to use ONLY the provided MarketData for facts.
# Claude's training data does not know today's stock prices.
# Without this instruction, Claude will hallucinate prices.
#
# Estimated tokens: ~40
# ─────────────────────────────────────────────────────────────

GROUNDING_INSTRUCTION: str = (
    "Base your analysis ONLY on the market data provided below. "
    "Do not use any knowledge from your training about current "
    "stock prices or recent market events. All price levels, "
    "volume figures, and market data must come exclusively from "
    "the data provided. If a data point is not in the provided "
    "data, do not invent it."
)


# ─────────────────────────────────────────────────────────────
# JSON FORMAT INSTRUCTION — Enforces parseable structured output
# ─────────────────────────────────────────────────────────────
# Claude must respond in valid JSON so we can parse it into
# MarketAnalysis Pydantic model. Without this, Claude often
# wraps JSON in markdown code blocks or adds preamble text.
#
# Estimated tokens: ~120
# ─────────────────────────────────────────────────────────────

JSON_FORMAT_INSTRUCTION: str = (
    "CRITICAL OUTPUT RULES — VIOLATIONS CAUSE SYSTEM FAILURE:\n"
    "1. Your ENTIRE response must be ONE valid JSON object\n"
    "2. Start your response with { immediately\n"
    "3. End your response with } as the very last character\n"
    "4. Zero text before the opening {\n"
    "5. Zero text after the closing }\n"
    "6. Zero markdown formatting\n"
    "7. Zero code blocks or backticks\n"
    "8. All string values must use double quotes\n"
    "9. No trailing commas\n"
    "10. No comments inside JSON\n\n"
    "The JSON parser will fail if you add ANY text outside "
    "the JSON object. This directly costs money in retries.\n"
    "Respond with JSON only. Nothing else. Ever.\n\n"
    "Keep each string field under 150 characters.\n"
    "Total response must be under 1200 tokens."
)


# ─────────────────────────────────────────────────────────────
# QUALITY REMINDER — Used on retry when first attempt fails
# ─────────────────────────────────────────────────────────────
# When the quality checker rejects Claude's first response,
# this stricter instruction is prepended on retry.
#
# Estimated tokens: ~60
# ─────────────────────────────────────────────────────────────

QUALITY_REMINDER: str = (
    "IMPORTANT: Your previous response was rejected because it "
    "did not meet the advisor quality standard. Specific issues:\n"
    "{quality_issues}\n\n"
    "This time, ensure EVERY field is substantive:\n"
    "- 'situation' must be >= 100 characters with specific market data\n"
    "- 'reasoning' must be >= 100 characters explaining WHY\n"
    "- 'action' must contain at least one specific price level\n"
    "- 'risk' must be >= 50 characters with concrete risk scenarios\n"
    "- 'lesson' must be >= 80 characters teaching one concept\n"
    "- 'cot_reasoning' must show your full step-by-step thinking\n"
    "Do not use 'N/A' or 'Not applicable' in any field."
)


# ─────────────────────────────────────────────────────────────
# TASK-SPECIFIC PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────
# These are the variable parts of the prompt that change based
# on what analysis is being requested. Data placeholders use
# {curly_braces} and are filled by claude_client.py at call time.
# ─────────────────────────────────────────────────────────────


MARKET_ANALYSIS_TASK: str = (
    "Analyse the following NSE market data and provide your "
    "senior advisor assessment.\n\n"

    "=== MARKET DATA ===\n"
    "{market_data_json}\n\n"

    "=== USER CONTEXT ===\n"
    "{user_context_json}\n\n"

    "=== INSTRUCTIONS ===\n"
    "{cot_instruction}\n\n"
    "{grounding_instruction}\n\n"

    "Respond with a JSON object containing these exact fields:\n"
    "- market_mood: one of [bullish, cautious_bullish, neutral, "
    "cautious_bearish, bearish, extreme_fear]\n"
    "- mood_confidence: float 0.0 to 1.0\n"
    "- situation: what is happening (>= 100 chars)\n"
    "- reasoning: why it is happening (>= 100 chars)\n"
    "- user_impact: what this means for the user specifically\n"
    "- action: what to consider doing (with price levels)\n"
    "- risk: what could go wrong (>= 50 chars)\n"
    "- lesson: one concept to learn (>= 80 chars)\n"
    "- cot_reasoning: your full step-by-step reasoning\n"
    "- top_opportunities: list of 3-5 ticker strings\n"
    "- top_risks: list of ticker strings with risk signals\n"
    "- risk_events: list of key risk events for today\n\n"

    "{json_format_instruction}"
)


QUICK_MOOD_TASK: str = (
    "Provide a quick market mood assessment based on this data. "
    "This is a fast check — not a full analysis.\n\n"

    "=== MARKET DATA ===\n"
    "{market_data_json}\n\n"

    "{grounding_instruction}\n\n"

    "Respond with a JSON object containing:\n"
    "- market_mood: one of [bullish, cautious_bullish, neutral, "
    "cautious_bearish, bearish, extreme_fear]\n"
    "- mood_confidence: float 0.0 to 1.0\n"
    "- situation: brief 2-3 sentence market summary\n"
    "- reasoning: brief why\n"
    "- action: what to watch or do right now\n"
    "- risk: main risk to watch\n"
    "- top_opportunities: list of 3 tickers\n"
    "- top_risks: list of tickers with risk\n\n"

    "{json_format_instruction}"
)


SENTIMENT_ANALYSIS_TASK: str = (
    "Analyse the following news items and sector data to determine "
    "market sentiment for Indian NSE markets.\n\n"

    "=== NEWS ITEMS ===\n"
    "{news_items_json}\n\n"

    "=== SECTOR PERFORMANCE ===\n"
    "{sector_data_json}\n\n"

    "{grounding_instruction}\n\n"

    "Respond with a JSON object containing:\n"
    "- overall_sentiment: one of [positive, negative, mixed, neutral]\n"
    "- sentiment_score: float -1.0 to +1.0\n"
    "- sentiment_confidence: float 0.0 to 1.0\n"
    "- sector_sentiments: dict mapping sector name to score (-1.0 to +1.0)\n"
    "- top_risk_events: list of top 3 risk events (strings)\n"
    "- risk_level: one of [low, normal, elevated, high]\n"
    "- news_summary: 2-3 sentence summary of today's news landscape\n"
    "- cot_reasoning: step-by-step reasoning for sentiment assessment\n\n"

    "{json_format_instruction}"
)


SECTOR_ANALYSIS_TASK: str = (
    "Analyse the following sector performance data and provide "
    "advisor-level sector rotation analysis.\n\n"

    "=== SECTOR DATA ===\n"
    "{sector_data_json}\n\n"

    "=== MARKET CONTEXT ===\n"
    "VIX: {vix_value} ({vix_signal})\n"
    "Nifty: {nifty_change_pct}%\n\n"

    "{grounding_instruction}\n\n"

    "For each sector, respond with a JSON array of objects containing:\n"
    "- sector_name: sector name\n"
    "- sector_mood: one of [bullish, cautious_bullish, neutral, "
    "cautious_bearish, bearish]\n"
    "- situation: what is happening in this sector\n"
    "- reasoning: why the sector is moving this way\n"
    "- advisor_action: what to recommend for this sector\n"
    "- top_opportunity: best stock opportunity (ticker + reason)\n"
    "- top_risk: highest risk stock (ticker + warning)\n\n"

    "{json_format_instruction}"
)


# ─────────────────────────────────────────────────────────────
# Prompt Builder Helpers
# ─────────────────────────────────────────────────────────────


def build_market_analysis_prompt(
    market_data_json: str,
    user_context_json: str,
    include_cot: bool = True,
) -> str:
    """Build the complete user message for market analysis.

    Fills the MARKET_ANALYSIS_TASK template with actual data.
    This is the user message — the system message is always
    MASTER_SYSTEM_PROMPT (set separately in claude_client.py).

    Args:
        market_data_json: Serialized MarketData from Module 1.
        user_context_json: Serialized UserContext from Module 5.
        include_cot: Whether to include CoT instruction. Default True.

    Returns:
        Complete user message string ready for Claude API.
    """
    return MARKET_ANALYSIS_TASK.format(
        market_data_json=market_data_json,
        user_context_json=user_context_json,
        cot_instruction=COT_INSTRUCTION if include_cot else "",
        grounding_instruction=GROUNDING_INSTRUCTION,
        json_format_instruction=JSON_FORMAT_INSTRUCTION,
    )


def build_quick_mood_prompt(
    market_data_json: str,
    user_message: str | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    """Build the user message for a quick mood check.

    Lightweight — no CoT, no user context, shorter response.
    Used for fast market status checks and chat conversations.

    Args:
        market_data_json: Serialized MarketData (can be trimmed).
        user_message: Optional user chat question to answer.
        conversation_history: Optional recent conversation context.

    Returns:
        Complete user message string ready for Claude API.
    """
    base = QUICK_MOOD_TASK.format(
        market_data_json=market_data_json,
        grounding_instruction=GROUNDING_INSTRUCTION,
        json_format_instruction=JSON_FORMAT_INSTRUCTION,
    )

    if not user_message:
        return base

    # Build conversation context if available
    conv_block = ""
    if conversation_history:
        recent = conversation_history[-6:]  # last 6 messages
        conv_lines = []
        for msg in recent:
            role = msg.get("role", "user").upper()
            conv_lines.append(f"{role}: {msg.get('content', '')}")
        conv_block = (
            "\n\n=== CONVERSATION HISTORY ===\n"
            + "\n".join(conv_lines)
        )

    return (
        base
        + conv_block
        + f"\n\n=== USER QUESTION ===\n{user_message}\n\n"
        "IMPORTANT: The user is asking a specific question. "
        "Focus your situation, action, and risk fields on answering "
        "their question using the market data above. "
        "Be conversational and directly address what they asked."
    )


def build_sentiment_prompt(
    news_items_json: str,
    sector_data_json: str,
) -> str:
    """Build the user message for sentiment analysis.

    Used by SentimentAnalysisAgent to analyse news items
    and sector data for overall market sentiment.

    Args:
        news_items_json: Serialized list of NewsItem objects.
        sector_data_json: Serialized list of SectorPerformance objects.

    Returns:
        Complete user message string ready for Claude API.
    """
    return SENTIMENT_ANALYSIS_TASK.format(
        news_items_json=news_items_json,
        sector_data_json=sector_data_json,
        grounding_instruction=GROUNDING_INSTRUCTION,
        json_format_instruction=JSON_FORMAT_INSTRUCTION,
    )


def build_sector_analysis_prompt(
    sector_data_json: str,
    vix_value: float,
    vix_signal: str,
    nifty_change_pct: float,
) -> str:
    """Build the user message for sector rotation analysis.

    Provides market context (VIX, Nifty) alongside sector data
    so Claude can assess sector movements in the right frame.

    Args:
        sector_data_json: Serialized list of SectorPerformance objects.
        vix_value: India VIX value for context.
        vix_signal: VIX signal label (e.g. "low_fear").
        nifty_change_pct: Nifty 50 percentage change.

    Returns:
        Complete user message string ready for Claude API.
    """
    return SECTOR_ANALYSIS_TASK.format(
        sector_data_json=sector_data_json,
        vix_value=vix_value,
        vix_signal=vix_signal,
        nifty_change_pct=nifty_change_pct,
        grounding_instruction=GROUNDING_INSTRUCTION,
        json_format_instruction=JSON_FORMAT_INSTRUCTION,
    )


def build_quality_retry_prompt(
    original_prompt: str,
    quality_issues: str,
) -> str:
    """Build a retry prompt when quality check fails.

    Prepends the QUALITY_REMINDER with specific issue details
    to the original prompt. Claude gets a second chance with
    explicit instructions about what was wrong.

    Args:
        original_prompt: The original user message that produced
            the rejected response.
        quality_issues: Formatted string describing what failed
            (from AnalysisQualityReport.issues).

    Returns:
        Enhanced user message with quality reminder prepended.
    """
    reminder = QUALITY_REMINDER.format(quality_issues=quality_issues)
    return f"{reminder}\n\n{original_prompt}"
