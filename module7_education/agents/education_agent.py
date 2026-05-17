"""
SwingAdvisorBot — Module 7: Education Layer
agents/education_agent.py — CrewAI education orchestration agent

Orchestrates the full 10-step CoT:
  1. Get market triggers (M1)
  2. Map triggers to concepts
  3. Check learning history (M5)
  4. Check difficulty level
  5. Select best concept
  6. Generate lesson with Claude
  7. Generate quiz (same call)
  8. Store in M5
  9. Format for Telegram
  10. Return Lesson

Usage:
    from module7_education.agents.education_agent import education_agent

    lesson = await education_agent.generate_daily_lesson()
    feedback = await education_agent.handle_quiz("B")
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from zoneinfo import ZoneInfo

from module7_education.config import CONCEPT_LIBRARY
from module7_education.generator.lesson_generator import (
    cache_lesson,
    generate_fallback_lesson,
    generate_lesson_with_claude,
    get_cached_lesson,
    validate_lesson,
)
from module7_education.generator.quiz_handler import handle_quiz_response
from module7_education.models import (
    DifficultyLevel,
    Lesson,
    LessonBrief,
    QuizFeedback,
    QuizResponse,
)
from module7_education.selector.concept_selector import (
    build_market_anchor,
    select_concept,
)
from module7_education.selector.difficulty_adapter import (
    build_learning_snapshot,
    compute_difficulty,
)

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.education_agent")


class EducationAgent:
    """Module 7's orchestration agent.

    Teaches Vijay one concept per day tied to real market events.
    Gets smarter as it learns what Vijay knows.

    Crew position:
      DataCollectorAgent (M1) → MarketAnalysisAgent (M2) →
      RiskAssessmentAgent (M3) → TradeSetupAgent (M4) →
      MemoryProvider (M5) → ReportAgent (M6) →
      EducationAgent (M7) ← YOU ARE HERE
    """

    agent_name: str = "EducationAgent"
    agent_role: str = "Senior Trading Mentor"
    agent_goal: str = (
        "Teach Vijay one financial concept per day tied to real "
        "market events. Track learning progress. Adapt difficulty. "
        "Make every lesson memorable with real examples."
    )

    def __init__(self) -> None:
        self._user_id = "XCU700"
        self._user_name = "Vijay"

    # ── Main entry point ─────────────────────────────────────

    async def generate_daily_lesson(
        self,
        stocks: list[dict] | None = None,
        india_vix: float = 0.0,
        nifty_value: float = 0.0,
        nifty_change_pct: float = 0.0,
        use_claude: bool = True,
    ) -> Lesson:
        """Full 10-step CoT lesson generation pipeline.

        Args:
            stocks: M1 stock data as dicts (with advisor_flag, ticker, etc.).
            india_vix: today's VIX value.
            nifty_value: Nifty 50 current value.
            nifty_change_pct: Nifty 50 change %.
            use_claude: if False, always use fallback (for testing).

        Returns:
            Complete Lesson object, cached for the day.
        """
        # Check day cache first
        cached = get_cached_lesson(self._user_id)
        if cached is not None:
            logger.info("[EducationAgent] Returning cached lesson")
            return cached

        if stocks is None:
            stocks = []

        # Step 1-3, 5: Get learning history from M5 and select concept
        taught_history, taught_this_week, current_difficulty = (
            self._get_learning_state()
        )

        # Step 4: Build learning snapshot (checks difficulty upgrade/downgrade)
        snapshot = build_learning_snapshot(
            taught_history, taught_this_week, current_difficulty
        )
        effective_difficulty = snapshot.current_difficulty

        # Step 5: Select concept
        concept = select_concept(
            stocks=stocks,
            india_vix=india_vix,
            taught_history=taught_history,
            current_difficulty=effective_difficulty,
        )
        logger.info(
            f"[EducationAgent] Step 5: Selected concept={concept}, "
            f"difficulty={effective_difficulty.value}"
        )

        # Build market anchor for Claude prompt
        market_anchor = build_market_anchor(
            concept=concept or "candlestick_basics",
            stocks=stocks,
            india_vix=india_vix,
            nifty_value=nifty_value,
            nifty_change_pct=nifty_change_pct,
        )

        # Step 6-7: Generate lesson + quiz
        if use_claude and concept:
            lesson = await generate_lesson_with_claude(
                concept=concept,
                market_anchor=market_anchor,
                difficulty=effective_difficulty,
                taught_this_week=taught_this_week,
                user_name=self._user_name,
            )
        else:
            lesson = generate_fallback_lesson(
                concept=concept or "candlestick_basics",
                market_anchor=market_anchor,
                difficulty=effective_difficulty,
            )

        # Validate (self-reflection)
        if not validate_lesson(lesson, taught_this_week, effective_difficulty):
            logger.warning("[EducationAgent] Validation failed — using fallback")
            lesson = generate_fallback_lesson(
                concept=concept or "candlestick_basics",
                market_anchor=market_anchor,
                difficulty=effective_difficulty,
            )

        # Step 8: Store in M5
        self._store_lesson_in_m5(lesson)

        # Cache for the day
        cache_lesson(lesson, self._user_id)

        logger.info(
            f"[EducationAgent] Step 10: Lesson ready — "
            f"{lesson.concept}, fallback={lesson.is_fallback}, "
            f"tokens={lesson.tokens_used}"
        )
        return lesson

    # ── Brief for M6 ─────────────────────────────────────────

    async def get_lesson_brief(self, **kwargs) -> LessonBrief | None:
        """Get today's lesson trimmed for M6 morning brief (≤200 tokens).

        Generates if not cached. Returns None only on catastrophic failure.
        """
        try:
            lesson = await self.generate_daily_lesson(**kwargs)
            return lesson.to_brief()
        except Exception as exc:
            logger.error(f"[EducationAgent] Failed to get brief: {exc}")
            return None

    # ── Quiz handling ─────────────────────────────────────────

    async def handle_quiz(
        self,
        answer: str,
        lesson_id: str | None = None,
    ) -> QuizFeedback:
        """Handle quiz response from Telegram.

        Args:
            answer: "A" or "B".
            lesson_id: optional lesson ID (uses cached if not provided).

        Returns:
            QuizFeedback with HTML, correctness, streak, score.
        """
        lesson = get_cached_lesson(self._user_id)

        quiz_response = QuizResponse(
            user_id=self._user_id,
            lesson_id=lesson_id or (lesson.lesson_id if lesson else "unknown"),
            answer=answer,
        )

        # Get quiz history from M5
        taught_history, _, _ = self._get_learning_state()

        feedback = await handle_quiz_response(
            quiz_response=quiz_response,
            lesson=lesson,
            quiz_history=taught_history,
            user_name=self._user_name,
        )

        # Update M5 with quiz score
        if lesson and feedback.correct is not None:
            self._update_quiz_score(
                lesson.concept,
                100 if feedback.correct else 0,
            )

        return feedback

    # ── Learning progress ─────────────────────────────────────

    def get_learning_progress(self) -> dict:
        """Return full learning state for MCP tool."""
        taught_history, taught_this_week, current_difficulty = (
            self._get_learning_state()
        )
        snapshot = build_learning_snapshot(
            taught_history, taught_this_week, current_difficulty
        )
        return {
            "user_id": self._user_id,
            "current_difficulty": snapshot.current_difficulty.value,
            "concepts_taught_this_week": snapshot.concepts_taught_this_week,
            "total_concepts_taught": snapshot.total_concepts_taught,
            "current_streak": snapshot.current_streak,
            "recent_weeks": [
                {
                    "week_start": w.week_start,
                    "total_quizzes": w.total_quizzes,
                    "correct_answers": w.correct_answers,
                    "score_pct": w.score_pct,
                }
                for w in snapshot.recent_weeks
            ],
            "should_upgrade": snapshot.should_upgrade,
            "should_downgrade": snapshot.should_downgrade,
        }

    # ── M5 integration (private) ─────────────────────────────

    def _get_learning_state(
        self,
    ) -> tuple[list[dict], list[str], DifficultyLevel]:
        """Query M5 for learning history.

        Returns:
            (taught_history, taught_this_week, current_difficulty)
        """
        try:
            from module5_memory.engine import memory_engine

            all_progress = memory_engine.get_all_learning(self._user_id)

            taught_history = [
                {
                    "concept": p.concept,
                    "taught_at": p.taught_at.isoformat()
                    if hasattr(p.taught_at, "isoformat")
                    else str(p.taught_at),
                    "last_taught": p.last_taught.isoformat()
                    if hasattr(p.last_taught, "isoformat")
                    else str(p.last_taught),
                    "quiz_score": p.quiz_score,
                }
                for p in all_progress
            ]

            # Concepts taught this week (Monday-Sunday)
            now = datetime.now(IST)
            from module7_education.selector.difficulty_adapter import get_week_start

            current_week = get_week_start(now)
            taught_this_week = [
                h["concept"]
                for h in taught_history
                if h.get("last_taught", "")[:10] >= current_week
            ]

            # Determine current difficulty from recent scores
            from module7_education.selector.difficulty_adapter import (
                build_weekly_scores,
                compute_difficulty,
            )

            weekly_scores = build_weekly_scores(taught_history)
            current_difficulty = compute_difficulty(
                DifficultyLevel.BEGINNER, weekly_scores
            )

            return taught_history, taught_this_week, current_difficulty

        except Exception as exc:
            logger.warning(
                f"[EducationAgent] M5 unavailable, using defaults: {exc}"
            )
            return [], [], DifficultyLevel.BEGINNER

    def _store_lesson_in_m5(self, lesson: Lesson) -> None:
        """Store lesson in M5 learning_progress table."""
        try:
            from module5_memory.engine import memory_engine
            from module5_memory.models import LearningProgress

            now = datetime.now(IST)
            progress_id = f"lp-{now.strftime('%Y%m%d')}-{lesson.concept}"

            # Check if already exists for today
            existing = memory_engine.get_learning(self._user_id, lesson.concept)
            if existing:
                # Update times_taught and last_taught
                existing.times_taught += 1
                existing.last_taught = now
                memory_engine.update_learning(existing)
            else:
                progress = LearningProgress(
                    progress_id=progress_id,
                    user_id=self._user_id,
                    concept=lesson.concept,
                    taught_at=now,
                    quiz_score=None,  # updated when quiz answered
                    times_taught=1,
                    last_taught=now,
                )
                memory_engine.update_learning(progress)

            logger.info(f"[EducationAgent] Step 8: Lesson stored in M5: {lesson.concept}")

        except Exception as exc:
            logger.warning(f"[EducationAgent] Failed to store in M5: {exc}")

    def _update_quiz_score(self, concept: str, score: int) -> None:
        """Update quiz score in M5."""
        try:
            from module5_memory.engine import memory_engine

            existing = memory_engine.get_learning(self._user_id, concept)
            if existing:
                existing.quiz_score = score
                existing.last_taught = datetime.now(IST)
                memory_engine.update_learning(existing)
                logger.info(
                    f"[EducationAgent] Quiz score updated: "
                    f"{concept}={score}"
                )
        except Exception as exc:
            logger.warning(f"[EducationAgent] Failed to update quiz score: {exc}")


# ── Singleton ────────────────────────────────────────────────

education_agent = EducationAgent()
