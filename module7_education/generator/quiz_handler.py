"""
Module 7 — Quiz Handler

Processes quiz responses from Telegram.
Updates learning progress in M5.
Generates feedback HTML for immediate delivery.

CoT Steps from Section 9:
  1. Get lesson from cache (or M5)
  2. Check if response correct
  3. Calculate new score
  4. Update M5 learning progress
  5. Generate feedback message
  6. Return feedback for Telegram send
  7. Check if difficulty should change
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from module7_education.generator.lesson_generator import get_cached_lesson
from module7_education.models import (
    Lesson,
    QuizFeedback,
    QuizResponse,
)
from module7_education.selector.difficulty_adapter import (
    build_weekly_scores,
    compute_difficulty,
    compute_streak,
)

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.education.quiz_handler")


def check_answer(lesson: Lesson, answer: str) -> bool:
    """Check if quiz answer is correct.

    Args:
        lesson: the Lesson containing the quiz.
        answer: user's answer ("A" or "B").

    Returns:
        True if correct.
    """
    return answer.strip().upper() == lesson.quiz.correct.upper()


def generate_feedback_html(
    is_correct: bool,
    lesson: Lesson,
    user_name: str = "Vijay",
    streak: int = 0,
    weekly_score_pct: float | None = None,
) -> str:
    """Generate HTML feedback for Telegram.

    Correct → celebration + explanation + streak
    Incorrect → encouragement + correct answer + explanation
    """
    if is_correct:
        parts = [
            "✅ <b>Correct!</b>",
            "",
            lesson.quiz.explanation,
        ]
        if streak >= 2:
            parts.append(f"\n🔥 {streak} correct in a row!")
        if weekly_score_pct is not None:
            parts.append(f"\nWin rate this week: {weekly_score_pct:.0f}% 🎯")
        parts.append(
            f"\nYou're building real market knowledge {user_name}! 🎯"
        )
    else:
        correct_letter = lesson.quiz.correct
        correct_text = lesson.quiz.options.get(correct_letter, "")
        parts = [
            f"Not quite. The answer is <b>{correct_letter}: {correct_text}</b>",
            "",
            lesson.quiz.explanation,
            "",
            f"Keep going — this is how you learn {user_name}! 💪",
        ]

    return "\n".join(parts)


async def handle_quiz_response(
    quiz_response: QuizResponse,
    lesson: Lesson | None = None,
    quiz_history: list[dict] | None = None,
    user_name: str = "Vijay",
) -> QuizFeedback:
    """Full quiz response pipeline.

    Steps:
      1. Resolve lesson (from arg or cache)
      2. Check answer correctness
      3. Compute streak and weekly score
      4. Generate feedback HTML
      5. Return QuizFeedback (caller stores in M5 + sends via Telegram)

    Args:
        quiz_response: incoming answer from Telegram.
        lesson: the lesson to check against (or None to use cache).
        quiz_history: M5 learning_progress rows for streak/score calc.
        user_name: user's display name.

    Returns:
        QuizFeedback with correct flag, HTML, score, and streak.
    """
    if quiz_history is None:
        quiz_history = []

    # Step 1: resolve lesson
    if lesson is None:
        lesson = get_cached_lesson(quiz_response.user_id)
    if lesson is None:
        logger.warning(
            f"[QuizHandler] No lesson found for {quiz_response.lesson_id}"
        )
        return QuizFeedback(
            correct=False,
            feedback_html=(
                "Sorry, I couldn't find today's lesson. "
                "The quiz may have expired. A new lesson will "
                "arrive tomorrow morning! 📚"
            ),
        )

    # Step 2: check answer
    is_correct = check_answer(lesson, quiz_response.answer)

    # Step 3: compute streak + weekly score
    # Add this response to history for calculation
    now = datetime.now(IST)
    updated_history = quiz_history + [
        {
            "concept": lesson.concept,
            "taught_at": now.isoformat(),
            "quiz_score": 100 if is_correct else 0,
        }
    ]
    streak = compute_streak(updated_history)
    weekly_scores = build_weekly_scores(updated_history)
    weekly_pct = weekly_scores[0].score_pct if weekly_scores else None

    # Step 4: generate feedback HTML
    feedback_html = generate_feedback_html(
        is_correct=is_correct,
        lesson=lesson,
        user_name=user_name,
        streak=streak,
        weekly_score_pct=weekly_pct,
    )

    # Step 5: return feedback
    return QuizFeedback(
        correct=is_correct,
        feedback_html=feedback_html,
        new_score=int(weekly_pct) if weekly_pct is not None else None,
        streak=streak,
    )
