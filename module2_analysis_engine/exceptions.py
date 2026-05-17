"""
SwingAdvisorBot — Module 2: AI Analysis Engine
exceptions.py — All custom exceptions for Claude API error handling

These exceptions provide clear, actionable error messages
when the Claude API fails. Each exception tells the user
exactly what went wrong and how to fix it.

Existing exceptions in models.py remain untouched:
  - AnalysisQualityError: Claude output fails quality gate
  - AnalysisParseError: Claude output is not valid JSON
  - InsufficientDataError: Not enough stocks for analysis
  - FinalAnalysisError: All retries exhausted

Existing exception in module1_data_layer/models.py:
  - TokenBudgetError: MarketData cannot be trimmed to budget

This file adds Claude API-specific exceptions only.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────
# Claude API Exceptions — Specific HTTP error code handling
# ─────────────────────────────────────────────────────────────

class ClaudeCreditsError(Exception):
    """Anthropic account has insufficient credits.

    Raised when Claude API returns 400 with a credit balance message.
    This is NOT a code bug — the user must add credits.

    Fix: Go to console.anthropic.com/settings/billing
    """

    def __init__(self, message: str = ""):
        super().__init__(
            f"[ClaudeCreditsError] {message or 'Anthropic account has no credits.'} "
            f"Add credits at: console.anthropic.com/settings/billing"
        )


class ClaudeAuthError(Exception):
    """Invalid or missing Anthropic API key.

    Raised when Claude API returns 401 Unauthorized.
    Key must start with sk-ant-api03-.

    Fix: Check ANTHROPIC_API_KEY in .env
    """

    def __init__(self, message: str = ""):
        super().__init__(
            f"[ClaudeAuthError] {message or 'Invalid Anthropic API key.'} "
            f"Check ANTHROPIC_API_KEY in .env. "
            f"Key must start with sk-ant-api03-"
        )


class ClaudeModelError(Exception):
    """Invalid Claude model name in configuration.

    Raised when Claude API returns 400 with a model-related error.
    Model is locked to claude-opus-4-5 per project spec.

    Fix: Check CLAUDE_MODEL in .env or remove it to use default
    """

    def __init__(self, model: str = "", message: str = ""):
        super().__init__(
            f"[ClaudeModelError] Invalid model: {model or 'unknown'}. "
            f"{message or 'Check CLAUDE_MODEL in .env.'} "
            f"Valid: claude-opus-4-5"
        )


class ClaudeRateLimitError(Exception):
    """Anthropic rate limit exceeded (HTTP 429).

    Raised when too many requests hit the API in a short window.
    The retry loop handles this with exponential backoff,
    but if retries are also exhausted, this surfaces.

    Fix: Wait 60 seconds and retry
    """

    def __init__(self, message: str = ""):
        super().__init__(
            f"[ClaudeRateLimitError] {message or 'Anthropic rate limit hit.'} "
            f"Waiting before retry."
        )


class ClaudeOverloadError(Exception):
    """Anthropic API temporarily overloaded (HTTP 529).

    Raised when Anthropic infrastructure is under heavy load.
    This is transient — retry after a short delay.

    Fix: Retry in 30 seconds
    """

    def __init__(self, message: str = ""):
        super().__init__(
            f"[ClaudeOverloadError] {message or 'Anthropic API overloaded.'} "
            f"Retry in 30 seconds."
        )


class ClaudeTimeoutError(Exception):
    """Claude API request timed out.

    Raised when the HTTP request exceeds the configured timeout
    (default 30 seconds). Could be network issue or Claude
    processing a complex prompt.

    Fix: Check internet connection and retry
    """

    def __init__(self, message: str = ""):
        super().__init__(
            f"[ClaudeTimeoutError] {message or 'Claude API timeout.'} "
            f"Check internet connection and retry."
        )


class ClaudeConnectionError(Exception):
    """Cannot connect to Anthropic API.

    Raised when the HTTP client cannot establish a connection.
    Network is unreachable, DNS failure, or firewall blocking.

    Fix: Check internet connection
    """

    def __init__(self, message: str = ""):
        super().__init__(
            f"[ClaudeConnectionError] {message or 'Cannot connect to Anthropic API.'} "
            f"Check internet connection."
        )


class ClaudeEmptyResponseError(Exception):
    """Claude returned empty or invalid response.

    Raised when the API returns 200 but the response body
    has no content blocks or no text. This should not happen
    with a well-formed prompt but is caught as a safety net.

    Fix: Check prompt structure, retry
    """

    def __init__(self, message: str = ""):
        super().__init__(
            f"[ClaudeEmptyResponseError] {message or 'Claude returned empty response.'} "
            f"Check prompt structure."
        )


class ClaudeAPIError(Exception):
    """Generic catch-all for unexpected Claude API errors.

    Raised when an error doesn't match any specific category.
    The error message from the API is included for debugging.

    Fix: Check logs for root cause
    """

    def __init__(self, message: str = ""):
        super().__init__(
            f"[ClaudeAPIError] {message or 'Unexpected Claude API error.'} "
            f"Check logs for root cause."
        )
