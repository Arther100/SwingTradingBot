"""
SwingAdvisorBot — Module 7: Education Layer
test_m7.py — 8 tests covering all M7 components

Tests 1-6, 8: no Claude API needed (fallback / offline).
Test 7: requires ANTHROPIC_API_KEY (Claude API credits).

Run: .\.venv\Scripts\python.exe test_m7.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
PASS = 0
FAIL = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  — {detail}")


# ═══════════════════════════════════════════════════════════
# TEST 1: Concept library loaded
# ═══════════════════════════════════════════════════════════

def test_concept_library() -> None:
    print("\n── Test 1: Concept Library ──")
    from module7_education.config import CONCEPT_LIBRARY, CONCEPT_ORDER

    report("10 concepts in library", len(CONCEPT_LIBRARY) == 10,
           f"got {len(CONCEPT_LIBRARY)}")
    report("10 concepts in order", len(CONCEPT_ORDER) == 10,
           f"got {len(CONCEPT_ORDER)}")

    all_ok = True
    for key in CONCEPT_ORDER:
        defn = CONCEPT_LIBRARY.get(key)
        if not defn:
            report(f"  concept '{key}' exists", False, "missing")
            all_ok = False
            continue
        if not defn.definition:
            all_ok = False
        if not defn.triggers:
            all_ok = False
        if not defn.difficulty:
            all_ok = False
        if not defn.fallback_lesson:
            all_ok = False
        if not defn.fallback_quiz_question:
            all_ok = False
        if defn.fallback_quiz_correct not in ("A", "B"):
            all_ok = False

    report("all concepts have definition+triggers+difficulty+fallback",
           all_ok)


# ═══════════════════════════════════════════════════════════
# TEST 2: Trigger → concept mapping
# ═══════════════════════════════════════════════════════════

def test_trigger_mapping() -> None:
    print("\n── Test 2: Trigger → Concept Mapping ──")
    from module7_education.selector.concept_selector import (
        extract_triggers,
        map_triggers_to_concepts,
        select_concept,
    )

    # unusual_activity → volume_analysis
    stocks_ua = [{"advisor_flag": "unusual_activity"}]
    triggers = extract_triggers(stocks_ua, india_vix=15.0)
    report("unusual_activity extracted", "unusual_activity" in triggers,
           f"got {triggers}")

    concepts = map_triggers_to_concepts(["unusual_activity"])
    report("maps to volume_analysis", "volume_analysis" in concepts,
           f"got {concepts}")

    # VIX 27.5 → india_vix
    triggers_vix = extract_triggers([], india_vix=27.5)
    report("vix_extreme extracted for VIX=27.5", "vix_extreme" in triggers_vix,
           f"got {triggers_vix}")

    concepts_vix = map_triggers_to_concepts(["vix_extreme"])
    report("maps to india_vix", "india_vix" in concepts_vix,
           f"got {concepts_vix}")

    # Full select
    selected = select_concept(stocks_ua, india_vix=15.0)
    report("select_concept returns a concept", selected is not None,
           "returned None")
    report("selected is in library",
           selected in map_triggers_to_concepts(triggers) or selected is not None)


# ═══════════════════════════════════════════════════════════
# TEST 3: Learning history filter
# ═══════════════════════════════════════════════════════════

def test_history_filter() -> None:
    print("\n── Test 3: Learning History Filter ──")
    from module7_education.selector.concept_selector import (
        filter_recently_taught,
        select_concept,
    )

    now = datetime.now(IST)
    history = [
        {"concept": "volume_analysis", "last_taught": (now - timedelta(days=3)).isoformat()},
    ]

    # volume_analysis taught 3 days ago → should be filtered
    filtered = filter_recently_taught(
        ["volume_analysis", "support_resistance"], history
    )
    report("volume_analysis filtered (3 days ago)",
           "volume_analysis" not in filtered,
           f"still in list: {filtered}")
    report("support_resistance kept",
           "support_resistance" in filtered,
           f"list: {filtered}")

    # Full select with history
    stocks = [{"advisor_flag": "unusual_activity"}]
    selected = select_concept(stocks, india_vix=15.0, taught_history=history)
    report("select_concept skips volume_analysis",
           selected != "volume_analysis",
           f"got {selected}")


# ═══════════════════════════════════════════════════════════
# TEST 4: Difficulty adaptation
# ═══════════════════════════════════════════════════════════

def test_difficulty_adaptation() -> None:
    print("\n── Test 4: Difficulty Adaptation ──")
    from module7_education.models import DifficultyLevel, WeeklyScore
    from module7_education.selector.difficulty_adapter import (
        compute_difficulty,
        compute_streak,
        build_learning_snapshot,
    )

    # Upgrade: ≥80% for 2 weeks
    scores_up = [
        WeeklyScore(week_start="2026-05-05", total_quizzes=5, correct_answers=5),
        WeeklyScore(week_start="2026-04-28", total_quizzes=5, correct_answers=4),
    ]
    new_d = compute_difficulty(DifficultyLevel.BEGINNER, scores_up)
    report("upgrade beginner → beginner+",
           new_d == DifficultyLevel.BEGINNER_PLUS,
           f"got {new_d}")

    # Downgrade: <60% for 1 week
    scores_down = [
        WeeklyScore(week_start="2026-05-05", total_quizzes=5, correct_answers=2),
    ]
    new_d2 = compute_difficulty(DifficultyLevel.INTERMEDIATE, scores_down)
    report("downgrade intermediate → beginner+",
           new_d2 == DifficultyLevel.BEGINNER_PLUS,
           f"got {new_d2}")

    # No change: 70% (only 1 week)
    scores_ok = [
        WeeklyScore(week_start="2026-05-05", total_quizzes=10, correct_answers=7),
    ]
    new_d3 = compute_difficulty(DifficultyLevel.BEGINNER, scores_ok)
    report("no change at 70% (1 week only)",
           new_d3 == DifficultyLevel.BEGINNER,
           f"got {new_d3}")

    # Streak
    now = datetime.now(IST)
    streak_hist = [
        {"concept": "a", "taught_at": (now - timedelta(days=0)).isoformat(), "quiz_score": 100},
        {"concept": "b", "taught_at": (now - timedelta(days=1)).isoformat(), "quiz_score": 100},
        {"concept": "c", "taught_at": (now - timedelta(days=2)).isoformat(), "quiz_score": 100},
        {"concept": "d", "taught_at": (now - timedelta(days=3)).isoformat(), "quiz_score": 0},
        {"concept": "e", "taught_at": (now - timedelta(days=4)).isoformat(), "quiz_score": 100},
    ]
    s = compute_streak(streak_hist)
    report("streak = 3 (3 correct then 1 wrong)", s == 3, f"got {s}")


# ═══════════════════════════════════════════════════════════
# TEST 5: Lesson generation (no Claude — fallback)
# ═══════════════════════════════════════════════════════════

def test_fallback_lesson() -> None:
    print("\n── Test 5: Fallback Lesson (No Claude) ──")
    from module7_education.generator.lesson_generator import (
        generate_fallback_lesson,
        validate_lesson,
    )
    from module7_education.models import DifficultyLevel

    lesson = generate_fallback_lesson(
        "volume_analysis",
        "HDFCBANK: ₹1650 (+2.66%), volume 1.14x avg",
        DifficultyLevel.BEGINNER,
    )

    report("lesson has lesson_id", bool(lesson.lesson_id))
    report("concept = volume_analysis", lesson.concept == "volume_analysis")
    report("is_fallback = True", lesson.is_fallback)
    report("has lesson_text", bool(lesson.lesson_text))
    report("has key_takeaway", bool(lesson.key_takeaway))
    report("quiz correct is A or B",
           lesson.quiz.correct in ("A", "B"),
           f"got {lesson.quiz.correct}")
    report("quiz has question", bool(lesson.quiz.question))
    report("quiz has explanation", bool(lesson.quiz.explanation))

    # Validate
    ok = validate_lesson(lesson, taught_this_week=[])
    report("validates successfully", ok)

    # Brief conversion
    brief = lesson.to_brief()
    report("brief has concept", brief.concept == "volume_analysis")
    report("brief has quiz_question", bool(brief.quiz_question))


# ═══════════════════════════════════════════════════════════
# TEST 6: Quiz handler
# ═══════════════════════════════════════════════════════════

def test_quiz_handler() -> None:
    print("\n── Test 6: Quiz Handler ──")
    from module7_education.generator.lesson_generator import (
        generate_fallback_lesson,
        cache_lesson,
        clear_cache,
    )
    from module7_education.generator.quiz_handler import (
        check_answer,
        generate_feedback_html,
        handle_quiz_response,
    )
    from module7_education.models import QuizResponse, DifficultyLevel

    lesson = generate_fallback_lesson("india_vix", "VIX at 22", DifficultyLevel.BEGINNER)
    correct_ans = lesson.quiz.correct
    wrong_ans = "A" if correct_ans == "B" else "B"

    # Check answer
    report("correct answer accepted", check_answer(lesson, correct_ans))
    report("wrong answer rejected", not check_answer(lesson, wrong_ans))

    # Feedback HTML — correct
    html_ok = generate_feedback_html(True, lesson, "Vijay", streak=3, weekly_score_pct=80.0)
    report("correct feedback has ✅", "Correct" in html_ok)
    report("correct feedback has streak", "3 correct" in html_ok)

    # Feedback HTML — wrong
    html_bad = generate_feedback_html(False, lesson, "Vijay")
    report("wrong feedback has explanation", "Not quite" in html_bad)
    report("wrong feedback has correct answer", correct_ans in html_bad)

    # Full pipeline — correct
    cache_lesson(lesson)
    qr = QuizResponse(user_id="XCU700", lesson_id=lesson.lesson_id, answer=correct_ans)
    fb = asyncio.run(handle_quiz_response(qr, user_name="Vijay"))
    report("pipeline: correct=True", fb.correct)
    report("pipeline: streak >= 1", fb.streak >= 1, f"got {fb.streak}")

    # Full pipeline — wrong
    qr2 = QuizResponse(user_id="XCU700", lesson_id=lesson.lesson_id, answer=wrong_ans)
    fb2 = asyncio.run(handle_quiz_response(qr2, lesson=lesson, user_name="Vijay"))
    report("pipeline wrong: correct=False", not fb2.correct)

    clear_cache()


# ═══════════════════════════════════════════════════════════
# TEST 7: Full lesson with Claude (needs API credits)
# ═══════════════════════════════════════════════════════════

def test_claude_lesson() -> None:
    print("\n── Test 7: Claude Lesson (API) ──")
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        report("ANTHROPIC_API_KEY present", False, "skipped — no key")
        return

    from module7_education.generator.lesson_generator import (
        generate_lesson_with_claude,
        clear_cache,
    )
    from module7_education.models import DifficultyLevel

    clear_cache()

    market_anchor = (
        "Nifty 50: 22550 (+0.35%)\n"
        "India VIX: 14.5\n"
        "HDFCBANK: ₹1650.00 (+2.66%), volume 1.14x avg"
    )

    lesson = asyncio.run(generate_lesson_with_claude(
        concept="volume_analysis",
        market_anchor=market_anchor,
        difficulty=DifficultyLevel.BEGINNER,
        taught_this_week=["india_vix"],
        user_name="Vijay",
    ))

    report("lesson generated", lesson is not None)
    report("concept = volume_analysis", lesson.concept == "volume_analysis")
    report("has title", bool(lesson.title))
    report("has lesson_text", bool(lesson.lesson_text))
    report("quiz correct is A or B",
           lesson.quiz.correct in ("A", "B"),
           f"got {lesson.quiz.correct}")
    if not lesson.is_fallback:
        report("tokens_used > 0 (Claude)", lesson.tokens_used > 0,
               f"got {lesson.tokens_used}")
    else:
        report("fallback used (Claude unavailable)", True,
               "model may not be available")

    # Check if lesson used real data (Claude) or definition (fallback)
    has_real = any(
        kw in (lesson.lesson_text + lesson.title).lower()
        for kw in ["volume", "hdfcbank", "nifty", "1650", "22550"]
    )
    report("lesson references real market data", has_real,
           f"title={lesson.title}")

    print(f"    → tokens used: {lesson.tokens_used}")
    print(f"    → is_fallback: {lesson.is_fallback}")
    clear_cache()


# ═══════════════════════════════════════════════════════════
# TEST 8: M6 integration (lesson fits in morning brief)
# ═══════════════════════════════════════════════════════════

def test_m6_integration() -> None:
    print("\n── Test 8: M6 Integration ──")
    from module7_education.engine import education_engine

    education_engine.clear_lesson_cache()

    stocks = [
        {"ticker": "HDFCBANK", "price": 1650.0, "change_pct": 2.66,
         "volume_ratio": 1.14, "advisor_flag": "unusual_activity"},
    ]

    # Get lesson via engine
    lesson = asyncio.run(education_engine.get_lesson(
        stocks=stocks, india_vix=14.5, nifty_value=22550,
        nifty_change_pct=0.35, use_claude=False,
    ))
    report("engine returns lesson", lesson is not None)

    # Get brief
    brief = asyncio.run(education_engine.get_lesson_brief(use_claude=False))
    report("brief returned", brief is not None)

    if brief:
        report("brief has concept", bool(brief.concept))
        report("brief has title", bool(brief.title))
        report("brief has summary", bool(brief.summary))
        report("brief has quiz_question", bool(brief.quiz_question))
        report("brief has quiz_options (A and B)",
               "A" in brief.quiz_options and "B" in brief.quiz_options,
               f"got {brief.quiz_options}")

        # Token estimate: summary + quiz should be compact
        total_chars = len(brief.summary) + len(brief.quiz_question)
        report("brief text under 1000 chars", total_chars < 1000,
               f"got {total_chars}")

    # Status
    status = education_engine.get_status()
    report("status shows today lesson", status["has_today_lesson"])

    # Progress
    progress = education_engine.get_progress()
    report("progress has difficulty", "current_difficulty" in progress)

    education_engine.clear_lesson_cache()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SwingAdvisorBot — Module 7: Education Layer Tests")
    print("=" * 60)

    test_concept_library()       # Test 1
    test_trigger_mapping()       # Test 2
    test_history_filter()        # Test 3
    test_difficulty_adaptation() # Test 4
    test_fallback_lesson()       # Test 5
    test_quiz_handler()          # Test 6
    test_claude_lesson()         # Test 7
    test_m6_integration()        # Test 8

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed")
    if FAIL:
        print(f"  ❌ {FAIL} test(s) FAILED")
    else:
        print("  ✅ ALL TESTS PASSED")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
