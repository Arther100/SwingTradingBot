"""
Module 7 — Concept Selector

Maps today's market triggers (M1 advisor_flags + VIX) to teachable concepts.
Filters out concepts taught in last 7 days (from M5 learning_progress).
Returns the best concept for today's lesson.

CoT Steps 1-3 and 5 from the education pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from module7_education.config import (
    CONCEPT_LIBRARY,
    CONCEPT_ORDER,
    NO_REPEAT_DAYS,
    TRIGGER_MAP,
)
from module7_education.models import DifficultyLevel

IST = ZoneInfo("Asia/Kolkata")


def extract_triggers(
    stocks: list[dict],
    india_vix: float = 0.0,
) -> list[str]:
    """CoT Step 1 — Extract market triggers from M1 data.

    Args:
        stocks: list of stock dicts with optional 'advisor_flag' key.
        india_vix: today's India VIX value.

    Returns:
        De-duplicated list of trigger strings (ordered by priority).
    """
    triggers: list[str] = []

    # VIX triggers (highest priority)
    if india_vix >= 25:
        triggers.append("vix_extreme")
    elif india_vix >= 20:
        triggers.append("vix_high")

    # Stock advisor_flag triggers
    for stock in stocks:
        flag = stock.get("advisor_flag")
        if flag and flag not in ("neutral", None):
            triggers.append(flag)

    # Always include fallback trigger
    if not triggers:
        triggers.append("any")

    # De-duplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in triggers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def map_triggers_to_concepts(triggers: list[str]) -> list[str]:
    """CoT Step 2 — Map triggers to candidate concepts.

    Returns ordered list of concept keys (no duplicates).
    """
    concepts: list[str] = []
    seen: set[str] = set()

    for trigger in triggers:
        for mapping in TRIGGER_MAP:
            if mapping.trigger == trigger:
                for concept in mapping.concepts:
                    if concept not in seen and concept in CONCEPT_LIBRARY:
                        seen.add(concept)
                        concepts.append(concept)

    # If nothing matched, add all concepts as candidates
    if not concepts:
        for c in CONCEPT_ORDER:
            if c not in seen:
                seen.add(c)
                concepts.append(c)

    return concepts


def filter_recently_taught(
    concepts: list[str],
    taught_history: list[dict],
) -> list[str]:
    """CoT Step 3 — Remove concepts taught within the last 7 days.

    Args:
        concepts: candidate concept keys.
        taught_history: list of dicts with 'concept' and 'last_taught'
                        (ISO datetime string or datetime object).

    Returns:
        Filtered list with recently-taught concepts removed.
    """
    now = datetime.now(IST)
    cutoff = now - timedelta(days=NO_REPEAT_DAYS)

    recently_taught: set[str] = set()
    for entry in taught_history:
        concept = entry.get("concept", "")
        last_taught = entry.get("last_taught")
        if not last_taught:
            continue

        if isinstance(last_taught, str):
            try:
                dt = datetime.fromisoformat(last_taught)
            except ValueError:
                continue
        else:
            dt = last_taught

        # Make timezone-aware if naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)

        if dt >= cutoff:
            recently_taught.add(concept)

    return [c for c in concepts if c not in recently_taught]


def filter_by_difficulty(
    concepts: list[str],
    current_difficulty: DifficultyLevel,
) -> list[str]:
    """Filter concepts to match current difficulty level.

    Beginner students get beginner concepts only.
    Beginner+ and above get both beginner and intermediate.
    """
    filtered: list[str] = []
    for concept in concepts:
        defn = CONCEPT_LIBRARY.get(concept)
        if not defn:
            continue

        if current_difficulty in (
            DifficultyLevel.BEGINNER,
            DifficultyLevel.BEGINNER_PLUS,
        ):
            # Beginner levels: prefer beginner concepts, but allow intermediate if nothing else
            if defn.difficulty == DifficultyLevel.BEGINNER:
                filtered.append(concept)
        else:
            # Intermediate levels: accept everything
            filtered.append(concept)

    # If all filtered out, allow all concepts (don't block learning)
    return filtered if filtered else concepts


def select_concept(
    stocks: list[dict],
    india_vix: float = 0.0,
    taught_history: list[dict] | None = None,
    current_difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
) -> Optional[str]:
    """CoT Step 5 — Full concept selection pipeline.

    Steps:
        1. Extract triggers from M1 data
        2. Map triggers to candidate concepts
        3. Filter out recently taught (7-day rule)
        4. Filter by difficulty level
        5. Return best match (first in priority order)

    Args:
        stocks: list of stock dicts with 'advisor_flag', 'ticker', 'price', etc.
        india_vix: today's VIX value.
        taught_history: M5 learning_progress rows.
        current_difficulty: Vijay's current level.

    Returns:
        Best concept key, or None if exhausted (shouldn't happen with 10 concepts).
    """
    if taught_history is None:
        taught_history = []

    # Step 1
    triggers = extract_triggers(stocks, india_vix)

    # Step 2
    candidates = map_triggers_to_concepts(triggers)

    # Step 3
    available = filter_recently_taught(candidates, taught_history)

    # Step 4
    matched = filter_by_difficulty(available, current_difficulty)

    # Step 5 — pick first (highest priority)
    if matched:
        return matched[0]

    # Fallback: if everything was filtered, try without difficulty filter
    if available:
        return available[0]

    # Ultimate fallback: rotate through CONCEPT_ORDER skipping recent
    all_available = filter_recently_taught(CONCEPT_ORDER, taught_history)
    if all_available:
        return all_available[0]

    # All 10 taught this week — just pick first in order
    return CONCEPT_ORDER[0]


def build_market_anchor(
    concept: str,
    stocks: list[dict],
    india_vix: float = 0.0,
    nifty_value: float = 0.0,
    nifty_change_pct: float = 0.0,
) -> str:
    """Build the real market data anchor text for Claude prompt.

    Finds the most relevant stock for the selected concept
    and formats a 2-3 line market snapshot.
    """
    lines: list[str] = []

    # Index data
    if nifty_value > 0:
        direction = "+" if nifty_change_pct >= 0 else ""
        lines.append(f"Nifty 50: {nifty_value:.0f} ({direction}{nifty_change_pct:.2f}%)")

    # VIX
    if india_vix > 0:
        lines.append(f"India VIX: {india_vix:.1f}")

    # Find best stock for this concept
    defn = CONCEPT_LIBRARY.get(concept)
    if defn and stocks:
        # Match by trigger
        for stock in stocks:
            flag = stock.get("advisor_flag", "")
            if flag in defn.triggers:
                ticker = stock.get("ticker", "?")
                price = stock.get("price", 0)
                change_pct = stock.get("change_pct", 0)
                volume_ratio = stock.get("volume_ratio", 0)
                direction = "+" if change_pct >= 0 else ""
                line = f"{ticker}: ₹{price:.2f} ({direction}{change_pct:.2f}%)"
                if volume_ratio > 0:
                    line += f", volume {volume_ratio:.2f}x avg"
                lines.append(line)
                break
        else:
            # No trigger match — use first non-neutral stock
            for stock in stocks:
                flag = stock.get("advisor_flag", "")
                if flag and flag != "neutral":
                    ticker = stock.get("ticker", "?")
                    price = stock.get("price", 0)
                    change_pct = stock.get("change_pct", 0)
                    direction = "+" if change_pct >= 0 else ""
                    lines.append(f"{ticker}: ₹{price:.2f} ({direction}{change_pct:.2f}%)")
                    break

    if not lines:
        lines.append("No specific market data available today")

    return "\n".join(lines)
