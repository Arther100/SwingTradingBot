"""
SwingAdvisorBot — Module 5: Memory & Personalization
verification/verification_engine.py — 2-round verification with Claude

Round 1: Another module generates advice (e.g. M4 trade setup).
Round 2: This engine verifies the advice against user history.

Verification checks:
  1. Consistency with user's risk tolerance
  2. Price levels grounded in provided data
  3. No contradictions with past trade history
  4. No unsupported claims or hallucinations
  5. Position sizing within safe limits

If confidence < 0.7 → flagged for user review.
If verified=True → original advice passes through.
If issues found → corrected_advice returned.

Uses httpx.AsyncClient (same pattern as M2 claude_client).

Usage:
    engine = VerificationEngine()
    result = await engine.verify(
        round1_advice="Buy HDFCBANK at 769...",
        user_profile=profile,
        relevant_chunks=chunks,
    )
    if result.needs_review:
        print("⚠️ Advice flagged for review")
    print(result.final_advice)
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx

from module5_memory.config import (
    VERIFICATION_CLAUDE_MODEL,
    VERIFICATION_MAX_TOKENS,
    VERIFICATION_MIN_CONFIDENCE,
    VERIFICATION_PROMPT,
    VERIFICATION_TEMPERATURE,
)
from module5_memory.models import RetrievedChunk, UserProfile, VerificationResult

logger = logging.getLogger("swing_advisor.m5_verification")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"


class VerificationEngine:
    """2-round verification engine for generated advice.

    Round 1 happens in another module (M2/M4) — they generate advice.
    Round 2 happens here — we verify against user history via Claude.

    Usage:
        engine = VerificationEngine()
        result = await engine.verify(
            round1_advice="...",
            user_profile=profile,
            relevant_chunks=[...],
        )
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self._api_key)

    async def verify(
        self,
        round1_advice: str,
        user_profile: UserProfile,
        relevant_chunks: list[RetrievedChunk] | None = None,
        available_prices: str = "",
    ) -> VerificationResult:
        """Verify round-1 advice against user history.

        Args:
            round1_advice: The advice generated in round 1.
            user_profile: User's profile for risk/capital checks.
            relevant_chunks: Semantic search results for history context.
            available_prices: Price data string for grounding check.

        Returns:
            VerificationResult with verified, issues, corrected_advice, confidence.
            If Claude unavailable → auto-verified with confidence 0.5.
        """
        if not self.is_available:
            logger.warning("[Verification] No API key — auto-passing with low confidence.")
            return VerificationResult(
                verified=True,
                issues_found=[],
                corrected_advice=None,
                confidence=0.5,
            )

        if not round1_advice or not round1_advice.strip():
            return VerificationResult(
                verified=True,
                issues_found=[],
                corrected_advice=None,
                confidence=1.0,
            )

        # Build history context from chunks
        history_text = self._build_history(relevant_chunks)

        # Format verification prompt
        prompt = VERIFICATION_PROMPT.format(
            round1_advice=round1_advice,
            risk_tolerance=user_profile.risk_tolerance,
            capital=f"{user_profile.capital:,.0f}",
            available_prices=available_prices or "Not provided",
            relevant_history=history_text or "No prior history available",
        )

        # Call Claude for verification
        try:
            response_text = await self._call_claude(prompt)
            result = self._parse_response(response_text, round1_advice)
            logger.info(
                f"[Verification] verified={result.verified}, "
                f"confidence={result.confidence}, "
                f"issues={len(result.issues_found)}"
            )
            return result

        except Exception as e:
            logger.error(f"[Verification] Claude call failed: {e}")
            # Graceful degradation: pass through with low confidence
            return VerificationResult(
                verified=True,
                issues_found=[f"Verification failed: {e}"],
                corrected_advice=None,
                confidence=0.5,
            )

    def verify_sync(
        self,
        round1_advice: str,
        user_profile: UserProfile,
        relevant_chunks: list[RetrievedChunk] | None = None,
        available_prices: str = "",
    ) -> VerificationResult:
        """Synchronous wrapper for verify().

        For use in non-async contexts. Creates a new event loop.
        """
        import asyncio

        return asyncio.run(
            self.verify(round1_advice, user_profile, relevant_chunks, available_prices)
        )

    # ═══════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════

    def _build_history(self, chunks: list[RetrievedChunk] | None) -> str:
        """Build history string from retrieved chunks."""
        if not chunks:
            return ""

        lines: list[str] = []
        for i, chunk in enumerate(chunks[:5], 1):  # Max 5 chunks
            lines.append(f"{i}. {chunk.content}")

        return "\n".join(lines)

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API for verification.

        Uses the same httpx pattern as M2 claude_client.
        """
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

        body = {
            "model": VERIFICATION_CLAUDE_MODEL,
            "max_tokens": VERIFICATION_MAX_TOKENS,
            "temperature": VERIFICATION_TEMPERATURE,
            "system": (
                "You are a senior risk reviewer for an Indian stock market swing trading advisor. "
                "Respond ONLY in valid JSON. No markdown, no backticks, no text outside JSON."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        start_time = time.monotonic()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0)
        ) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=body,
            )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        logger.debug(f"[Verification] Claude responded in {elapsed_ms}ms")

        if response.status_code != 200:
            error_msg = response.text[:200]
            raise RuntimeError(f"Claude API error {response.status_code}: {error_msg}")

        data = response.json()
        content = data.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0]["text"]

        raise RuntimeError("Unexpected Claude response format")

    def _parse_response(
        self,
        response_text: str,
        original_advice: str,
    ) -> VerificationResult:
        """Parse Claude's JSON response into VerificationResult.

        Handles malformed JSON gracefully.
        """
        # Strip any markdown fencing Claude might add despite instructions
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"[Verification] Invalid JSON from Claude: {text[:200]}")
            return VerificationResult(
                verified=True,
                issues_found=["Could not parse verification response"],
                corrected_advice=None,
                confidence=0.5,
            )

        verified = data.get("verified", True)
        issues = data.get("issues_found", [])
        corrected = data.get("corrected_advice")
        confidence = float(data.get("confidence", 0.5))

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # If not verified but no corrected advice, use original
        if not verified and not corrected:
            corrected = original_advice

        return VerificationResult(
            verified=verified,
            issues_found=issues if isinstance(issues, list) else [str(issues)],
            corrected_advice=corrected,
            confidence=confidence,
        )
