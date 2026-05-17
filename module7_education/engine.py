"""
SwingAdvisorBot — Module 7: Education Layer
engine.py — M7 public API singleton

Single entry point for all M7 operations.
Wraps EducationAgent with simple async methods.

Usage:
    from module7_education.engine import education_engine

    # Generate today's lesson (cached for the day)
    lesson = await education_engine.get_lesson()

    # Get brief for M6 morning report
    brief = await education_engine.get_lesson_brief()

    # Handle quiz answer from Telegram
    feedback = await education_engine.submit_quiz("B")

    # Check learning progress
    progress = education_engine.get_progress()

    # Status
    status = education_engine.get_status()
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from zoneinfo import ZoneInfo

from module7_education.agents.education_agent import education_agent
from module7_education.generator.lesson_generator import clear_cache, get_cached_lesson
from module7_education.models import Lesson, LessonBrief, QuizFeedback

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.m7_engine")


class EducationEngine:
    """Module 7 public API — lessons, quizzes, and learning progress.

    All methods are async where needed. All methods catch errors internally
    and return sensible defaults on failure (never raises to caller).
    """

    def __init__(self) -> None:
        self._agent = education_agent

    # ── Lesson generation ─────────────────────────────────────

    async def get_lesson(
        self,
        stocks: list[dict] | None = None,
        india_vix: float = 0.0,
        nifty_value: float = 0.0,
        nifty_change_pct: float = 0.0,
        use_claude: bool = True,
    ) -> Lesson:
        """Generate or retrieve today's lesson.

        Cached for the full day — one Claude call maximum.
        Falls back to CONCEPT_LIBRARY on any failure.

        Args:
            stocks: M1 stock data as dicts.
            india_vix: today's VIX.
            nifty_value: Nifty 50 value.
            nifty_change_pct: Nifty 50 change %.
            use_claude: False for testing (skip API call).

        Returns:
            Complete Lesson object (never None).
        """
        try:
            return await self._agent.generate_daily_lesson(
                stocks=stocks,
                india_vix=india_vix,
                nifty_value=nifty_value,
                nifty_change_pct=nifty_change_pct,
                use_claude=use_claude,
            )
        except Exception as exc:
            logger.error(f"[M7Engine] get_lesson failed: {exc}")
            from module7_education.generator.lesson_generator import (
                generate_fallback_lesson,
            )
            return generate_fallback_lesson("candlestick_basics", "")

    async def get_lesson_brief(self, **kwargs) -> Optional[LessonBrief]:
        """Get today's lesson trimmed for M6 morning brief (≤200 tokens).

        Returns None only on catastrophic failure.
        """
        try:
            return await self._agent.get_lesson_brief(**kwargs)
        except Exception as exc:
            logger.error(f"[M7Engine] get_lesson_brief failed: {exc}")
            return None

    # ── Quiz ──────────────────────────────────────────────────

    async def submit_quiz(
        self,
        answer: str,
        lesson_id: str | None = None,
    ) -> QuizFeedback:
        """Submit quiz answer and get feedback.

        Args:
            answer: "A" or "B".
            lesson_id: optional lesson ID.

        Returns:
            QuizFeedback with HTML, correctness, streak, score.
        """
        try:
            return await self._agent.handle_quiz(
                answer=answer,
                lesson_id=lesson_id,
            )
        except Exception as exc:
            logger.error(f"[M7Engine] submit_quiz failed: {exc}")
            return QuizFeedback(
                correct=False,
                feedback_html="Sorry, something went wrong processing your quiz answer. Try again tomorrow! 📚",
            )

    # ── Progress ──────────────────────────────────────────────

    def get_progress(self) -> dict:
        """Return full learning state.

        Returns dict with:
          - current_difficulty
          - concepts_taught_this_week
          - total_concepts_taught
          - current_streak
          - recent_weeks (list of weekly scores)
          - should_upgrade / should_downgrade
        """
        try:
            return self._agent.get_learning_progress()
        except Exception as exc:
            logger.error(f"[M7Engine] get_progress failed: {exc}")
            return {
                "user_id": "XCU700",
                "current_difficulty": "beginner",
                "concepts_taught_this_week": [],
                "total_concepts_taught": 0,
                "current_streak": 0,
                "recent_weeks": [],
                "should_upgrade": False,
                "should_downgrade": False,
            }

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return M7 engine status for health checks."""
        cached = get_cached_lesson()
        return {
            "module": "M7_Education",
            "status": "ready",
            "has_today_lesson": cached is not None,
            "today_concept": cached.concept if cached else None,
            "is_fallback": cached.is_fallback if cached else None,
            "timestamp": datetime.now(IST).isoformat(),
        }

    # ── Cache management ──────────────────────────────────────

    def clear_lesson_cache(self) -> None:
        """Clear the lesson cache (for testing or forced regeneration)."""
        clear_cache()
        logger.info("[M7Engine] Lesson cache cleared")


# ── Singleton ────────────────────────────────────────────────

education_engine = EducationEngine()
