"""
SwingAdvisorBot — Module 2: AI Analysis Engine
config.py — Central configuration for Claude API and analysis parameters

This file is the single source of truth for all Module 2 configuration:
  → Claude API settings (model, temperature, max tokens, timeout)
  → Token budget breakdown (system prompt, CoT, data, memory, output)
  → Quality gate thresholds (minimum character counts per field)
  → Retry configuration (max retries, backoff)
  → Analysis cache settings (TTL, key format)
  → Environment variable loading (ANTHROPIC_API_KEY, CLAUDE_MODEL)

Design decisions:
  - Follows Module 1's config.py pattern for consistency
  - All secrets loaded from .env via python-dotenv (never hardcoded)
  - ClaudeConfig is frozen — no runtime mutations
  - AnalysisConfig controls quality gates and retry behaviour
  - get_claude_settings() is cached with lru_cache (singleton pattern)

Token budget awareness:
  Every Claude API call is budgeted before it fires.
  Hard limit: 3000 tokens per call (never exceed).
  Input budget: 1500 tokens (system + CoT + data + memory).
  Output budget: 1500 tokens (Claude response).
  TokenController (File 04) enforces these budgets.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# ─────────────────────────────────────────────────────────────
# Claude API Constants — Never hardcode these elsewhere
# ─────────────────────────────────────────────────────────────

# Model selection — locked to claude-opus-4-5 per project spec.
# Never change without updating the Project Intelligence doc.
DEFAULT_CLAUDE_MODEL = "claude-opus-4-5"

# Temperature — low for consistent, reliable financial advice.
# Financial analysis demands reproducibility over creativity.
# 0.3 gives slight variation in phrasing while keeping advice stable.
DEFAULT_TEMPERATURE = 0.3

# Token limits — these define the hard budget per Claude API call.
# Exceeding these wastes money and risks truncated responses.
HARD_TOKEN_LIMIT = 3000        # Absolute maximum per call (input + output)
INPUT_TOKEN_LIMIT = 1500       # Maximum input tokens sent to Claude
OUTPUT_TOKEN_LIMIT = 1500      # Maximum output tokens requested from Claude

# Token budget breakdown — how the 1500 input tokens are allocated.
# These are estimates used by TokenController for budget planning.
# Fixed tokens: 380 + 180 + 300 = 860 (system + CoT + memory)
# Market data budget: 3000 - 860 - 1500(output) = 640 tokens max
SYSTEM_PROMPT_TOKENS = 380     # MASTER_SYSTEM_PROMPT (fixed, never trimmed)
COT_INSTRUCTION_TOKENS = 180   # Chain of Thought instruction (fixed, never trimmed)
MARKET_DATA_TOKENS = 640       # MarketData from Module 1 (variable, trimmable)
USER_MEMORY_TOKENS = 300       # UserContext from Module 5 (variable, keep if possible)
STRUCTURE_OVERHEAD_TOKENS = 100  # JSON structure, message framing overhead
FIXED_PROMPT_TOKENS = 860      # system(380) + CoT(180) + memory(300) — never trimmed
MARKET_DATA_BUDGET = 640       # 3000 - 860 - 1500 = max tokens for MarketData

# API call settings
DEFAULT_API_TIMEOUT = 30       # Seconds — Claude API call timeout
DEFAULT_API_BASE_URL = "https://api.anthropic.com"
DEFAULT_API_VERSION = "2023-06-01"  # Anthropic API version header


# ─────────────────────────────────────────────────────────────
# Quality Gate Thresholds — Minimum content for advisor-grade output
# If Claude's response is below these, the quality checker rejects it.
# ─────────────────────────────────────────────────────────────

# Character minimums per field (from Section 5 CoT Pattern + Section 9 Constraint 6)
QUALITY_MIN_SITUATION = 100       # "What is happening" must be substantial
QUALITY_MIN_REASONING = 100       # "Why it is happening" must explain causation
QUALITY_MIN_ACTION = 50           # "What to do" must be specific (price levels)
QUALITY_MIN_RISK = 50             # "What could go wrong" must be concrete
QUALITY_MIN_LESSON = 80           # "What to learn" must be educational
QUALITY_MIN_COT = 50              # CoT reasoning must be present and meaningful

# Banned content — fields containing these are treated as empty
QUALITY_BANNED_PHRASES: list[str] = [
    "N/A",
    "Not applicable",
    "Not available",
    "No data",
    "TBD",
    "To be determined",
    "None available",
]


# ─────────────────────────────────────────────────────────────
# Retry Configuration — How the system handles Claude API failures
# ─────────────────────────────────────────────────────────────

MAX_RETRIES = 2                   # Maximum retry attempts after initial failure
RETRY_BACKOFF_SECONDS = 5.0       # Wait time before first retry (connection errors)
QUALITY_RETRY_BACKOFF = 1.0       # Wait time before quality/parse retry (fast retry)


# ─────────────────────────────────────────────────────────────
# Analysis Cache Configuration
# ─────────────────────────────────────────────────────────────

ANALYSIS_CACHE_TTL = 600          # 10 minutes — Claude analysis cache TTL
SENTIMENT_CACHE_TTL = 600         # 10 minutes — sentiment report cache TTL
MOOD_CACHE_TTL = 300              # 5 minutes — quick mood check cache TTL

# Minimum stocks required for quality analysis
MIN_STOCKS_FOR_ANALYSIS = 10      # Below this → InsufficientDataError


# ─────────────────────────────────────────────────────────────
# ClaudeConfig — Frozen configuration for Claude API calls
# ─────────────────────────────────────────────────────────────


class ClaudeConfig(BaseModel):
    """Configuration for Claude API calls.

    This model controls how the system interacts with Claude.
    It is frozen — no runtime mutations allowed. Change settings
    via environment variables or by creating a new instance.

    Key parameters:
      model          → claude-opus-4-5 (locked per project spec)
      temperature    → 0.3 (consistency over creativity for finance)
      max_output     → 1500 tokens (hard limit on Claude response)
      timeout        → 30 seconds (API call timeout)

    Token budget is NOT in this config — it lives in the constants
    above and is enforced by TokenController (File 04).
    """

    model_config = {"frozen": True}

    model: str = Field(
        default=DEFAULT_CLAUDE_MODEL,
        description=(
            "Claude model identifier. Locked to claude-opus-4-5 per project spec. "
            "Never change without updating the Project Intelligence doc."
        ),
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=1.0,
        description=(
            "Sampling temperature. 0.3 for financial advice — "
            "consistent and reliable over creative and varied."
        ),
    )
    max_output_tokens: int = Field(
        default=OUTPUT_TOKEN_LIMIT,
        ge=100,
        le=4096,
        description="Maximum output tokens requested from Claude.",
    )
    timeout_seconds: int = Field(
        default=DEFAULT_API_TIMEOUT,
        ge=5,
        le=120,
        description="API call timeout in seconds.",
    )
    api_base_url: str = Field(
        default=DEFAULT_API_BASE_URL,
        description="Anthropic API base URL.",
    )
    api_version: str = Field(
        default=DEFAULT_API_VERSION,
        description="Anthropic API version header value.",
    )


# ─────────────────────────────────────────────────────────────
# AnalysisConfig — Controls analysis behaviour and quality gates
# ─────────────────────────────────────────────────────────────


class AnalysisConfig(BaseModel):
    """Controls how the analysis engine operates.

    Separate from ClaudeConfig because these settings control
    the engine's behaviour (retries, caching, quality gates)
    rather than the API call parameters.

    Usage:
        config = AnalysisConfig()
        # or override for testing:
        config = AnalysisConfig(max_retries=0, cache_ttl=0)
    """

    max_retries: int = Field(
        default=MAX_RETRIES,
        ge=0,
        le=5,
        description=(
            "Maximum retry attempts for Claude API calls. "
            "Includes quality retries and parse retries."
        ),
    )
    cache_ttl: int = Field(
        default=ANALYSIS_CACHE_TTL,
        ge=0,
        description=(
            "Cache TTL for Claude analysis responses in seconds. "
            "0 disables caching (useful for testing)."
        ),
    )
    min_stocks_required: int = Field(
        default=MIN_STOCKS_FOR_ANALYSIS,
        ge=1,
        le=50,
        description=(
            "Minimum stocks required in MarketData for analysis. "
            "Below this → InsufficientDataError."
        ),
    )
    enable_quality_check: bool = Field(
        default=True,
        description=(
            "Whether to run the quality gate after Claude responds. "
            "Disable only for debugging — never in production."
        ),
    )
    enable_hallucination_guard: bool = Field(
        default=True,
        description=(
            "Whether to run hallucination detection on Claude responses. "
            "Verifies price levels and tickers against MarketData."
        ),
    )
    enable_cot_reasoning: bool = Field(
        default=True,
        description=(
            "Whether to include CoT instruction in Claude prompts. "
            "Disable to save ~180 tokens when CoT is not needed."
        ),
    )


# ─────────────────────────────────────────────────────────────
# Settings — Environment variables for Module 2
# ─────────────────────────────────────────────────────────────


class ClaudeSettings(BaseModel):
    """Module 2 settings loaded from environment variables.

    All API keys come from .env — never hardcoded.
    Missing ANTHROPIC_API_KEY will cause claude_client.py
    to raise a clear error on the first API call.
    """

    model_config = {"frozen": True}

    # Claude API
    anthropic_api_key: str = Field(
        default="",
        description=(
            "Anthropic API key for Claude calls. "
            "Get from https://console.anthropic.com/settings/keys"
        ),
    )
    claude_model: str = Field(
        default=DEFAULT_CLAUDE_MODEL,
        description=(
            "Claude model to use. Defaults to claude-opus-4-5. "
            "Can be overridden via CLAUDE_MODEL env var for testing."
        ),
    )

    # MCP Server (Module 1)
    mcp_base_url: str = Field(
        default="http://127.0.0.1:8001",
        description="Module 1 MCP server base URL for data fetching.",
    )

    # App config
    log_level: str = Field(
        default="INFO",
        description="Logging level for Module 2.",
    )
    token_budget: int = Field(
        default=HARD_TOKEN_LIMIT,
        description="Hard token limit per Claude API call.",
    )


@lru_cache(maxsize=1)
def get_claude_settings() -> ClaudeSettings:
    """Load and cache Module 2 settings from environment variables.

    Called once at startup. Uses lru_cache to avoid re-reading .env
    on every access. The claude_client and engine call this to get
    API keys and configuration.

    If ANTHROPIC_API_KEY is missing, it defaults to empty string.
    claude_client.py validates the key before making API calls
    and raises a clear error with setup instructions.
    """
    return ClaudeSettings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        claude_model=os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
        mcp_base_url=os.getenv(
            "MCP_BASE_URL",
            f"http://127.0.0.1:{os.getenv('MCP_PORT', '8001')}",
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        token_budget=int(os.getenv("TOKEN_BUDGET", str(HARD_TOKEN_LIMIT))),
    )
