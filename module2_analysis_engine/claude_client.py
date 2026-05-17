"""
SwingAdvisorBot — Module 2: AI Analysis Engine
claude_client.py — Async Claude API wrapper

This is the single point of contact between SwingAdvisorBot and
the Claude API. Every Claude call in the entire project goes
through this client — no direct API calls anywhere else.

The client handles:
  → Building the request (system message + user message)
  → Token budget enforcement via TokenController
  → Making the async HTTP call via httpx
  → Parsing the JSON response
  → Retry logic for connection, parse, and quality failures
  → Response caching to save tokens (10 min TTL)
  → Timing and token accounting for AnalysisResult metadata

API details:
  Endpoint: https://api.anthropic.com/v1/messages
  Auth: x-api-key header with ANTHROPIC_API_KEY
  Model: claude-opus-4-5 (locked per project spec)
  Temperature: 0.3 (consistency for financial advice)
  Max output tokens: 1500
  Timeout: 30 seconds

Error handling (from Section 9 Constraint 8):
  → httpx connection/timeout → retry once after 5 seconds
  → JSON parse failure → retry with stricter JSON instruction
  → Quality gate failure → retry with QUALITY_REMINDER
  → After 2 retries → raise FinalAnalysisError
  → Never return empty or partial analysis silently

Async throughout (Constraint 7):
  Uses httpx.AsyncClient — never blocking requests.
  All public methods are async.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime

import httpx
from zoneinfo import ZoneInfo

from module1_data_layer.cache import cache
from module2_analysis_engine.config import (
    ANALYSIS_CACHE_TTL,
    MAX_RETRIES,
    QUALITY_RETRY_BACKOFF,
    RETRY_BACKOFF_SECONDS,
    ClaudeConfig,
    get_claude_settings,
)
from module2_analysis_engine.exceptions import (
    ClaudeAPIError,
    ClaudeAuthError,
    ClaudeConnectionError,
    ClaudeCreditsError,
    ClaudeEmptyResponseError,
    ClaudeModelError,
    ClaudeOverloadError,
    ClaudeRateLimitError,
    ClaudeTimeoutError,
)
from module2_analysis_engine.models import (
    AnalysisParseError,
    FinalAnalysisError,
)
from module2_analysis_engine.prompts import (
    JSON_FORMAT_INSTRUCTION,
    MASTER_SYSTEM_PROMPT,
)
from module2_analysis_engine.token_controller import token_controller

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.claude_client")


class ClaudeClient:
    """Async Claude API client for SwingAdvisorBot.

    This client wraps the Anthropic Messages API with:
      → Token budget enforcement (via TokenController)
      → Structured JSON response parsing
      → Retry logic (connection, parse, quality)
      → Response caching (10 min TTL)
      → Full timing and token accounting

    Every Claude API call in the project flows through
    call_claude() or call_claude_raw(). No direct API
    calls anywhere else in the codebase.

    Usage:
        client = ClaudeClient()
        response_dict = await client.call_claude(
            system_prompt=MASTER_SYSTEM_PROMPT,
            user_message="Analyse this market data...",
        )
        # response_dict is the parsed JSON from Claude
    """

    def __init__(self, config: ClaudeConfig | None = None):
        """Initialize the Claude client.

        Args:
            config: ClaudeConfig with model, temperature, timeout.
                    Defaults to standard production config.
        """
        self._config = config or ClaudeConfig()
        self._settings = get_claude_settings()

    async def call_claude(
        self,
        system_prompt: str,
        user_message: str,
        cache_key: str | None = None,
        cache_ttl: int = ANALYSIS_CACHE_TTL,
    ) -> dict:
        """Call Claude API and return parsed JSON response.

        This is the primary entry point for all structured Claude calls.
        It handles caching, retries, JSON parsing, and token accounting.

        Flow:
          1. Check cache → return if fresh
          2. Validate API key is present
          3. Make API call via _make_api_call()
          4. Parse JSON response
          5. Cache the result
          6. Return parsed dict

        Retry logic (up to MAX_RETRIES attempts):
          → Connection/timeout error → wait RETRY_BACKOFF_SECONDS, retry
          → JSON parse error → retry with stricter JSON instruction
          → After all retries exhausted → raise FinalAnalysisError

        Args:
            system_prompt: System message (usually MASTER_SYSTEM_PROMPT).
            user_message: User message with data and instructions.
            cache_key: Optional cache key. If provided, response is cached.
            cache_ttl: Cache TTL in seconds. Default 10 minutes.

        Returns:
            Parsed JSON dict from Claude's response.

        Raises:
            FinalAnalysisError: After all retries exhausted.
            AnalysisParseError: If JSON parsing fails on final attempt
                (only if not caught by retry logic).
        """
        # ── Check cache ──
        if cache_key:
            cached = cache.get(cache_key)
            if cached is not None:
                logger.info(
                    f"Claude response served from cache. "
                    f"Key: {cache_key[:40]}... "
                    f"Age: {cache.get_age(cache_key):.0f}s."
                )
                return cached

        # ── Validate API key ──
        if not self._settings.anthropic_api_key:
            raise FinalAnalysisError(
                attempts=0,
                last_error=(
                    "ANTHROPIC_API_KEY not found in .env file. "
                    "Add ANTHROPIC_API_KEY=<your_key> to .env. "
                    "Get a key from https://console.anthropic.com/settings/keys"
                ),
            )

        # ── Retry loop ──
        last_error = ""
        current_user_message = user_message

        for attempt in range(MAX_RETRIES + 1):
            try:
                # Make the API call
                raw_response, api_latency_ms, usage = await self._make_api_call(
                    system_prompt=system_prompt,
                    user_message=current_user_message,
                )

                # Validate output token budget
                token_controller.validate_output(raw_response)

                # Parse JSON
                parsed = self._parse_json_response(raw_response)

                # Cache the result
                if cache_key:
                    cache.set(cache_key, parsed, ttl=cache_ttl)

                logger.info(
                    f"Claude API call successful (attempt {attempt + 1}). "
                    f"Latency: {api_latency_ms}ms. "
                    f"Input tokens: {usage.get('input_tokens', 0)}, "
                    f"Output tokens: {usage.get('output_tokens', 0)}."
                )

                return parsed

            except AnalysisParseError as e:
                last_error = str(e)
                logger.warning(
                    f"Claude response JSON parse failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): "
                    f"{e.parse_error}. Retrying with stricter JSON instruction."
                )
                # Retry with stricter JSON formatting
                current_user_message = (
                    f"CRITICAL: Your previous response was NOT valid JSON. "
                    f"Parse error: {e.parse_error}. "
                    f"You MUST respond with ONLY valid JSON. "
                    f"No text before or after the JSON. "
                    f"Start with {{ and end with }}.\n\n"
                    f"{user_message}"
                )
                await asyncio.sleep(QUALITY_RETRY_BACKOFF)

            except httpx.TimeoutException:
                last_error = (
                    f"Claude API timeout after {self._config.timeout_seconds}s"
                )
                logger.warning(
                    f"Claude API timeout (attempt {attempt + 1}/{MAX_RETRIES + 1}). "
                    f"Retrying after {RETRY_BACKOFF_SECONDS}s backoff."
                )
                await asyncio.sleep(RETRY_BACKOFF_SECONDS)

            except httpx.HTTPStatusError as e:
                last_error = f"Claude API HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.error(
                    f"Claude API HTTP error (attempt {attempt + 1}/{MAX_RETRIES + 1}): "
                    f"Status {e.response.status_code}."
                )
                if e.response.status_code == 429:
                    # Rate limited — back off longer
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * 2)
                elif e.response.status_code >= 500:
                    # Server error — retry
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                else:
                    # Client error (401, 400) — don't retry
                    raise FinalAnalysisError(
                        attempts=attempt + 1,
                        last_error=last_error,
                    )

            except (ClaudeCreditsError, ClaudeAuthError, ClaudeModelError):
                # Non-retryable Claude-specific errors — surface immediately
                raise

            except ClaudeRateLimitError:
                last_error = "Claude API rate limit hit"
                logger.warning(
                    f"Claude rate limited (attempt {attempt + 1}/{MAX_RETRIES + 1}). "
                    f"Backing off {RETRY_BACKOFF_SECONDS * 2}s."
                )
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * 2)

            except ClaudeOverloadError:
                last_error = "Claude API overloaded"
                logger.warning(
                    f"Claude overloaded (attempt {attempt + 1}/{MAX_RETRIES + 1}). "
                    f"Backing off {RETRY_BACKOFF_SECONDS}s."
                )
                await asyncio.sleep(RETRY_BACKOFF_SECONDS)

            except (ClaudeEmptyResponseError, ClaudeAPIError) as e:
                last_error = str(e)
                logger.warning(
                    f"Claude API error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. "
                    f"Retrying."
                )
                await asyncio.sleep(RETRY_BACKOFF_SECONDS)

            except httpx.ConnectError:
                last_error = "Claude API connection failed — network unreachable"
                logger.warning(
                    f"Claude API connection error (attempt {attempt + 1}/{MAX_RETRIES + 1}). "
                    f"Retrying after {RETRY_BACKOFF_SECONDS}s."
                )
                await asyncio.sleep(RETRY_BACKOFF_SECONDS)

        # All retries exhausted
        raise FinalAnalysisError(
            attempts=MAX_RETRIES + 1,
            last_error=last_error,
        )

    async def call_claude_raw(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, int, dict]:
        """Call Claude API and return raw text response with metadata.

        Lower-level call that returns the raw text without JSON parsing.
        Used when the caller needs to handle parsing themselves
        (e.g., for quality retry logic in agents).

        Args:
            system_prompt: System message.
            user_message: User message.

        Returns:
            Tuple of:
              - str: Raw response text from Claude
              - int: API latency in milliseconds
              - dict: Usage info {input_tokens, output_tokens}

        Raises:
            FinalAnalysisError: If API key is missing.
            httpx.TimeoutException: On timeout (caller handles retry).
            httpx.HTTPStatusError: On HTTP errors (caller handles).
        """
        if not self._settings.anthropic_api_key:
            raise FinalAnalysisError(
                attempts=0,
                last_error="ANTHROPIC_API_KEY not found in .env",
            )

        return await self._make_api_call(
            system_prompt=system_prompt,
            user_message=user_message,
        )

    async def _make_api_call(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, int, dict]:
        """Execute the actual HTTP request to Claude API.

        Builds the request body per Anthropic Messages API spec
        and makes the async HTTP call via httpx.

        Request format:
          POST /v1/messages
          Headers: x-api-key, anthropic-version, content-type
          Body: {model, max_tokens, temperature, system, messages}

        Args:
            system_prompt: System message text.
            user_message: User message text.

        Returns:
            Tuple of:
              - str: Response text (content[0].text)
              - int: API latency in milliseconds
              - dict: Usage info from API response

        Raises:
            httpx.TimeoutException: On request timeout.
            httpx.HTTPStatusError: On non-2xx response.
            httpx.ConnectError: On connection failure.
        """
        url = f"{self._config.api_base_url}/v1/messages"

        headers = {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": self._config.api_version,
            "content-type": "application/json",
        }

        body = {
            "model": self._settings.claude_model,
            "max_tokens": self._config.max_output_tokens,
            "temperature": self._config.temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        }

        start_time = time.monotonic()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout_seconds)
        ) as client:
            response = await client.post(url, headers=headers, json=body)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # ── Handle specific HTTP errors before raise_for_status ──
        if response.status_code == 400:
            error_body = response.json()
            error_text = str(error_body).lower()
            if "credit" in error_text or "balance" in error_text:
                raise ClaudeCreditsError(
                    "Anthropic account has no credits. "
                    "Add credits at: console.anthropic.com/settings/billing"
                )
            elif "model" in error_text:
                raise ClaudeModelError(
                    model=self._settings.claude_model,
                    message=error_body.get("error", {}).get("message", ""),
                )
            else:
                raise ClaudeAPIError(
                    f"Bad request: {error_body.get('error', {}).get('message', response.text[:200])}"
                )

        if response.status_code == 401:
            raise ClaudeAuthError(
                "Invalid Anthropic API key. "
                "Check ANTHROPIC_API_KEY in .env."
            )

        if response.status_code == 429:
            raise ClaudeRateLimitError(
                "Anthropic rate limit hit. "
                "Waiting before retry."
            )

        if response.status_code == 529:
            raise ClaudeOverloadError(
                "Anthropic API overloaded. "
                "Retry in 30 seconds."
            )

        # Any other non-2xx error
        response.raise_for_status()

        response_data = response.json()

        # Extract text from the response
        content_blocks = response_data.get("content", [])
        if not content_blocks:
            raise ClaudeEmptyResponseError(
                "Claude returned empty content array. "
                "Check prompt structure."
            )

        response_text = content_blocks[0].get("text", "")
        if not response_text:
            raise ClaudeEmptyResponseError(
                "Claude returned empty text in content block."
            )

        # Extract usage info
        usage = response_data.get("usage", {})

        logger.debug(
            f"Claude API response received. "
            f"Latency: {elapsed_ms}ms. "
            f"Response length: {len(response_text)} chars. "
            f"Model: {response_data.get('model', 'unknown')}."
        )

        return response_text, elapsed_ms, usage

    def _parse_json_response(self, raw_response: str) -> dict:
        """Parse Claude's response text as JSON.

        Handles common Claude response quirks:
          → Response wrapped in markdown code blocks (```json ... ```)
          → Leading/trailing whitespace
          → BOM characters

        If parsing fails after cleanup, raises AnalysisParseError
        which triggers a retry with stricter JSON instructions.

        Args:
            raw_response: Raw text from Claude API.

        Returns:
            Parsed JSON dict.

        Raises:
            AnalysisParseError: If JSON parsing fails.
        """
        text = raw_response.strip()

        # Strip markdown code blocks if Claude wrapped the JSON
        if text.startswith("```"):
            # Remove opening ```json or ```
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Remove closing ```
            if text.endswith("```"):
                text = text[:-3].strip()

        # Strip BOM if present
        text = text.lstrip("\ufeff")

        # Try to find JSON object boundaries
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise AnalysisParseError(
                raw_response=raw_response,
                parse_error="No JSON object found in response",
            )

        json_text = text[start_idx:end_idx + 1]

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise AnalysisParseError(
                raw_response=raw_response,
                parse_error=str(e),
            )

        if not isinstance(parsed, dict):
            raise AnalysisParseError(
                raw_response=raw_response,
                parse_error=f"Expected JSON object, got {type(parsed).__name__}",
            )

        return parsed

    @staticmethod
    def generate_cache_key(
        market_data_timestamp: str,
        user_id: str,
        analysis_type: str = "full",
    ) -> str:
        """Generate a deterministic cache key for Claude responses.

        The key is based on the market data timestamp and user ID,
        ensuring that the same market conditions for the same user
        return cached results. When market data refreshes, the
        timestamp changes and the cache key changes → fresh analysis.

        Args:
            market_data_timestamp: ISO format timestamp of MarketData.
            user_id: User identifier (e.g., "XCU700").
            analysis_type: Type of analysis ("full", "quick", "sentiment").

        Returns:
            Cache key string for use with Module 1's cache.
        """
        raw = f"{analysis_type}:{user_id}:{market_data_timestamp}"
        key_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"claude:{analysis_type}:{key_hash}"


# Module-level singleton — used across the analysis engine
claude_client = ClaudeClient()
