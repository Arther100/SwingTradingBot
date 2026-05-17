"""
Module 7 — Difficulty Adapter

Tracks Vijay's quiz performance and adapts difficulty level.

Rules:
  ≥80% for 2 consecutive weeks → upgrade difficulty
  <60% for 1 week           → downgrade difficulty

Difficulty ladder:
  beginner → beginner+ → intermediate → intermediate+
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from module7_education.config import (
    DIFFICULTY_LADDER,
    DOWNGRADE_THRESHOLD_PCT,
    UPGRADE_THRESHOLD_PCT,
    UPGRADE_WEEKS_REQUIRED,
)
from module7_education.models import DifficultyLevel, LearningSnapshot, WeeklyScore

IST = ZoneInfo("Asia/Kolkata")


def get_week_start(dt: datetime) -> str:
    """Return Monday of the week containing *dt* as YYYY-MM-DD."""
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def build_weekly_scores(quiz_history: list[dict]) -> list[WeeklyScore]:
    """Aggregate quiz results into weekly scores.

    Args:
        quiz_history: list of dicts with 'taught_at' (ISO str or datetime)
                      and 'quiz_score' (int 0-100 or None).

    Returns:
        List of WeeklyScore, most recent first, up to 4 weeks.
    """
    buckets: dict[str, dict] = {}  # week_start → {total, correct}

    for entry in quiz_history:
        score = entry.get("quiz_score")
        if score is None:
            continue

        taught_at = entry.get("taught_at") or entry.get("last_taught")
        if not taught_at:
            continue

        if isinstance(taught_at, str):
            try:
                dt = datetime.fromisoformat(taught_at)
            except ValueError:
                continue
        else:
            dt = taught_at

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)

        week = get_week_start(dt)
        if week not in buckets:
            buckets[week] = {"total": 0, "correct": 0}
        buckets[week]["total"] += 1
        if score >= 50:  # 50+ counts as correct for binary A/B quiz
            buckets[week]["correct"] += 1

    # Sort by week descending, take last 4
    sorted_weeks = sorted(buckets.keys(), reverse=True)[:4]
    return [
        WeeklyScore(
            week_start=w,
            total_quizzes=buckets[w]["total"],
            correct_answers=buckets[w]["correct"],
        )
        for w in sorted_weeks
    ]


def compute_difficulty(
    current: DifficultyLevel,
    weekly_scores: list[WeeklyScore],
) -> DifficultyLevel:
    """Determine new difficulty based on recent weekly scores.

    Args:
        current: current difficulty level.
        weekly_scores: most-recent-first list of WeeklyScore.

    Returns:
        New difficulty level (may be same, higher, or lower).
    """
    idx = DIFFICULTY_LADDER.index(current) if current in DIFFICULTY_LADDER else 0

    # Check upgrade: ≥80% for 2 consecutive weeks
    if len(weekly_scores) >= UPGRADE_WEEKS_REQUIRED:
        recent = weekly_scores[:UPGRADE_WEEKS_REQUIRED]
        if all(w.score_pct >= UPGRADE_THRESHOLD_PCT for w in recent):
            new_idx = min(idx + 1, len(DIFFICULTY_LADDER) - 1)
            return DIFFICULTY_LADDER[new_idx]

    # Check downgrade: <60% for most recent week
    if len(weekly_scores) >= 1:
        if weekly_scores[0].score_pct < DOWNGRADE_THRESHOLD_PCT:
            new_idx = max(idx - 1, 0)
            return DIFFICULTY_LADDER[new_idx]

    return current


def compute_streak(quiz_history: list[dict]) -> int:
    """Count consecutive correct answers from most recent backward.

    Args:
        quiz_history: list of dicts with 'quiz_score' and 'taught_at'.
                      Need not be sorted.

    Returns:
        Number of consecutive correct answers (streak).
    """
    # Sort by taught_at descending
    dated: list[tuple[datetime, int]] = []
    for entry in quiz_history:
        score = entry.get("quiz_score")
        if score is None:
            continue
        taught_at = entry.get("taught_at") or entry.get("last_taught")
        if not taught_at:
            continue
        if isinstance(taught_at, str):
            try:
                dt = datetime.fromisoformat(taught_at)
            except ValueError:
                continue
        else:
            dt = taught_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        dated.append((dt, score))

    dated.sort(key=lambda x: x[0], reverse=True)

    streak = 0
    for _, score in dated:
        if score >= 50:
            streak += 1
        else:
            break
    return streak


def build_learning_snapshot(
    quiz_history: list[dict],
    taught_this_week: list[str],
    current_difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
) -> LearningSnapshot:
    """Build a complete LearningSnapshot from M5 data.

    Args:
        quiz_history: all learning_progress rows for the user.
        taught_this_week: concept names taught in the current week.
        current_difficulty: stored difficulty level.

    Returns:
        LearningSnapshot with computed fields.
    """
    weekly_scores = build_weekly_scores(quiz_history)
    new_difficulty = compute_difficulty(current_difficulty, weekly_scores)
    streak = compute_streak(quiz_history)

    return LearningSnapshot(
        current_difficulty=new_difficulty,
        concepts_taught_this_week=taught_this_week,
        total_concepts_taught=len(
            {e.get("concept") for e in quiz_history if e.get("concept")}
        ),
        recent_weeks=weekly_scores,
        current_streak=streak,
    )
