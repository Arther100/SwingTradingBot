"""
Module 7 — Lesson Generator

CoT Steps 6-7, 9: Generate lesson + quiz via Claude,
fall back to CONCEPT_LIBRARY if Claude unavailable.
Cache lesson for the full day — one Claude call per day.

Token budget: 760 input + 400 output = 1 160 tokens.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from module7_education.config import (
    CLAUDE_MODEL,
    CONCEPT_LIBRARY,
    EDUCATION_SYSTEM_PROMPT,
    EDUCATION_USER_TEMPLATE,
    LESSON_TOKEN_BUDGET,
)
from module7_education.models import (
    DifficultyLevel,
    Lesson,
    Quiz,
    RealExample,
)

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.education.lesson_generator")

# ── Day-level cache ──────────────────────────────────────────
_lesson_cache: dict[str, Lesson] = {}  # key = "user_id:YYYY-MM-DD"


def _cache_key(user_id: str) -> str:
    today = datetime.now(IST).strftime("%Y-%m-%d")
    return f"{user_id}:{today}"


def get_cached_lesson(user_id: str = "XCU700") -> Lesson | None:
    """Return today's cached lesson, or None."""
    return _lesson_cache.get(_cache_key(user_id))


def cache_lesson(lesson: Lesson, user_id: str = "XCU700") -> None:
    """Store lesson in day cache."""
    _lesson_cache[_cache_key(user_id)] = lesson


def clear_cache() -> None:
    """Clear the lesson cache (for testing)."""
    _lesson_cache.clear()


# ── Fallback lesson (no Claude) ──────────────────────────────

def generate_fallback_lesson(
    concept: str,
    market_anchor: str,
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
) -> Lesson:
    """Generate lesson from CONCEPT_LIBRARY without Claude.

    Used when:
      - Claude API key missing
      - Claude call fails after retry
      - Test mode (no API credits needed)

    Returns:
        Complete Lesson with is_fallback=True.
    """
    defn = CONCEPT_LIBRARY.get(concept)
    if not defn:
        # Ultimate fallback — first concept in library
        defn = next(iter(CONCEPT_LIBRARY.values()))
        concept = defn.concept

    now = datetime.now(IST)
    lesson_id = f"L-{now.strftime('%Y%m%d')}-{concept}"

    return Lesson(
        lesson_id=lesson_id,
        concept=concept,
        title=f"{concept.replace('_', ' ').title()} — Today's Lesson",
        difficulty=difficulty,
        market_trigger=market_anchor.split("\n")[0] if market_anchor else "Market data",
        real_example=RealExample(event=market_anchor or "See market data above"),
        lesson_text=defn.fallback_lesson,
        key_takeaway=defn.definition,
        quiz=Quiz(
            question=defn.fallback_quiz_question,
            options=defn.fallback_quiz_options,
            correct=defn.fallback_quiz_correct,
            explanation=defn.fallback_quiz_explanation,
        ),
        tokens_used=0,
        generated_at=now,
        is_fallback=True,
    )


# ── Claude lesson generation ─────────────────────────────────

def _get_anthropic_key() -> str:
    """Get Anthropic API key from environment."""
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("ANTHROPIC_API_KEY", "")


def _build_user_prompt(
    concept: str,
    difficulty: DifficultyLevel,
    market_anchor: str,
    taught_this_week: list[str],
    user_name: str = "Vijay",
) -> str:
    """Build the user message from template."""
    defn = CONCEPT_LIBRARY.get(concept)
    definition = defn.definition if defn else concept.replace("_", " ")

    taught_str = ", ".join(taught_this_week) if taught_this_week else "None"

    return EDUCATION_USER_TEMPLATE.format(
        user_name=user_name,
        difficulty=difficulty.value,
        concept=concept,
        definition=definition,
        market_anchor=market_anchor,
        taught_this_week=taught_str,
    )


def _parse_claude_response(
    raw_text: str,
    concept: str,
    market_anchor: str,
    difficulty: DifficultyLevel,
) -> Lesson | None:
    """Parse Claude JSON response into a Lesson object.

    Returns None if parsing fails.
    """
    try:
        # Strip markdown fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        data = json.loads(text)

        now = datetime.now(IST)
        lesson_id = f"L-{now.strftime('%Y%m%d')}-{concept}"

        # Extract real example
        real_example_text = data.get("real_example", market_anchor)
        real_example = RealExample(event=str(real_example_text))

        return Lesson(
            lesson_id=lesson_id,
            concept=concept,
            title=data.get("title", f"{concept} — Lesson"),
            difficulty=difficulty,
            market_trigger=market_anchor.split("\n")[0] if market_anchor else "",
            real_example=real_example,
            lesson_text=data.get("lesson_text", ""),
            key_takeaway=data.get("key_takeaway", ""),
            quiz=Quiz(
                question=data.get("quiz_question", ""),
                options={
                    "A": data.get("quiz_option_a", "Option A"),
                    "B": data.get("quiz_option_b", "Option B"),
                },
                correct=data.get("quiz_correct", "B"),
                explanation=data.get("quiz_explanation", ""),
            ),
            tokens_used=0,  # updated after API call
            generated_at=now,
            is_fallback=False,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"[LessonGen] Failed to parse Claude response: {exc}")
        return None


async def generate_lesson_with_claude(
    concept: str,
    market_anchor: str,
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
    taught_this_week: list[str] | None = None,
    user_name: str = "Vijay",
) -> Lesson:
    """CoT Step 6+7: Generate lesson + quiz via Claude API.

    Makes one httpx call to Anthropic Messages API.
    Falls back to CONCEPT_LIBRARY on any failure.

    Args:
        concept: selected concept key (e.g. "volume_analysis").
        market_anchor: real market data text from build_market_anchor().
        difficulty: current difficulty level.
        taught_this_week: concepts already taught this week.
        user_name: user's display name.

    Returns:
        Lesson object (is_fallback=False on success, True on failure).
    """
    import httpx

    if taught_this_week is None:
        taught_this_week = []

    api_key = _get_anthropic_key()
    if not api_key:
        logger.warning("[LessonGen] No ANTHROPIC_API_KEY — using fallback lesson")
        return generate_fallback_lesson(concept, market_anchor, difficulty)

    user_prompt = _build_user_prompt(
        concept=concept,
        difficulty=difficulty,
        market_anchor=market_anchor,
        taught_this_week=taught_this_week,
        user_name=user_name,
    )

    try:
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
                    "max_tokens": LESSON_TOKEN_BUDGET["output_budget"],
                    "system": EDUCATION_SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=30.0,
            )

            data = response.json()

            if response.status_code != 200:
                error = data.get("error", {}).get("message", "Unknown error")
                logger.error(f"[LessonGen] Claude API error {response.status_code}: {error}")
                return generate_fallback_lesson(concept, market_anchor, difficulty)

            # Extract text from response
            content_blocks = data.get("content", [])
            text_parts = [
                block["text"]
                for block in content_blocks
                if block.get("type") == "text"
            ]
            raw_text = "\n".join(text_parts)

            # Parse JSON response into Lesson
            lesson = _parse_claude_response(raw_text, concept, market_anchor, difficulty)
            if lesson is None:
                logger.warning("[LessonGen] Parse failed — using fallback")
                return generate_fallback_lesson(concept, market_anchor, difficulty)

            # Record token usage
            usage = data.get("usage", {})
            lesson.tokens_used = (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            )

            logger.info(
                f"[LessonGen] Lesson generated via Claude: "
                f"concept={concept}, tokens={lesson.tokens_used}"
            )
            return lesson

    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.error(f"[LessonGen] Claude call failed: {exc}")
        return generate_fallback_lesson(concept, market_anchor, difficulty)
    except Exception as exc:
        logger.error(f"[LessonGen] Unexpected error: {exc}")
        return generate_fallback_lesson(concept, market_anchor, difficulty)


# ── Validation (CoT self-reflection) ─────────────────────────

def validate_lesson(
    lesson: Lesson,
    taught_this_week: list[str] | None = None,
    expected_difficulty: DifficultyLevel | None = None,
) -> bool:
    """Self-reflection checks on generated lesson.

    Q1: Lesson anchored to real market data?
    Q2: Concept not taught in last 7 days?
    Q3: Difficulty appropriate?
    Q4: Quiz has correct answer defined?
    Q5: Under token limits?

    Returns True if all checks pass. Logs warnings for failures.
    """
    if taught_this_week is None:
        taught_this_week = []

    ok = True

    # Q1: Real market data anchoring
    if not lesson.market_trigger or lesson.market_trigger == "Market data":
        logger.warning("[Validate] Lesson not anchored to real market data")
        ok = False

    # Q2: Not taught this week
    if lesson.concept in taught_this_week:
        logger.warning(f"[Validate] Concept '{lesson.concept}' already taught this week")
        ok = False

    # Q3: Difficulty match
    if expected_difficulty and lesson.difficulty != expected_difficulty:
        logger.warning(
            f"[Validate] Difficulty mismatch: "
            f"expected={expected_difficulty.value}, got={lesson.difficulty.value}"
        )
        # Not a hard failure — still usable

    # Q4: Quiz integrity
    if lesson.quiz.correct not in ("A", "B"):
        logger.warning(f"[Validate] Invalid quiz correct answer: {lesson.quiz.correct}")
        ok = False
    if not lesson.quiz.question:
        logger.warning("[Validate] Quiz question is empty")
        ok = False

    # Q5: Token check (lesson_text should be reasonable)
    if len(lesson.lesson_text) > 2000:
        logger.warning("[Validate] Lesson text exceeds 2000 chars")
        ok = False

    return ok
