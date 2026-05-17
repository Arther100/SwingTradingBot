"""
SwingAdvisorBot — Module 1: Data Layer
signals/news_scorer.py — Chain of Thought news relevance scorer

The news_fetcher gives us raw headlines. This module gives them meaning.
A headline like "RBI holds repo rate at 6.5%" is just text until we score
it with sentiment, sector impact, relevance, and an advisor note.

Every news item is scored through a 6-step Chain of Thought process:
  Step 1: Identify the news category (monetary_policy, earnings, macro, etc.)
  Step 2: Check which sectors are affected
  Step 3: Assess magnitude of potential market impact (high/medium/low)
  Step 4: Check recency (older news scores lower)
  Step 5: Calculate final relevance_score (0.0–1.0)
  Step 6: Generate advisor_note explaining the impact

Only news items with relevance_score >= 0.70 reach the advisor.
Below that threshold, it's noise — the advisor's time is valuable.

Scoring approach:
  This is a rule-based keyword scorer, not an LLM scorer. Why?
  - Zero API cost (no Claude tokens consumed for scoring)
  - Deterministic (same headline always scores the same)
  - Fast (< 1ms per headline, all 30 headlines in < 30ms)
  - Sufficient for swing trading context (we're filtering, not analyzing)
  
  Module 2 (Claude API) does the deep analysis. Our job is to filter
  noise and provide structured metadata so Claude can focus on what matters.

Data flow:
  NewsItem (raw from news_fetcher) → score_news_item() → NewsItem (scored)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from zoneinfo import ZoneInfo

from module1_data_layer.models import MarketImpact, NewsItem, NewsSentiment

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.signals.news_scorer")


# ─────────────────────────────────────────────────────────────
# Keyword dictionaries — the scoring engine's knowledge base
# Each dict maps keywords/phrases → score contribution or label.
# Curated for Indian capital markets context.
# ─────────────────────────────────────────────────────────────

# News categories with base relevance scores.
# Higher base score = more likely to matter to Indian equity markets.
NEWS_CATEGORIES: dict[str, tuple[str, float]] = {
    # Monetary policy — directly moves banking, real estate, broad market
    "repo rate": ("monetary_policy", 0.90),
    "rbi": ("monetary_policy", 0.85),
    "interest rate": ("monetary_policy", 0.85),
    "monetary policy": ("monetary_policy", 0.90),
    "rate cut": ("monetary_policy", 0.92),
    "rate hike": ("monetary_policy", 0.92),
    "inflation": ("macro_economic", 0.80),
    "cpi": ("macro_economic", 0.78),
    "gdp": ("macro_economic", 0.75),
    "fiscal deficit": ("macro_economic", 0.75),
    # Market events — direct market impact
    "nifty": ("market_event", 0.82),
    "sensex": ("market_event", 0.82),
    "fii": ("market_flow", 0.85),
    "fpi": ("market_flow", 0.85),
    "dii": ("market_flow", 0.80),
    "ipo": ("market_event", 0.70),
    "buyback": ("corporate_action", 0.72),
    "dividend": ("corporate_action", 0.65),
    "stock split": ("corporate_action", 0.68),
    "bonus": ("corporate_action", 0.65),
    # Earnings — high impact during result season
    "quarterly results": ("earnings", 0.88),
    "q1 results": ("earnings", 0.85),
    "q2 results": ("earnings", 0.85),
    "q3 results": ("earnings", 0.85),
    "q4 results": ("earnings", 0.85),
    "profit": ("earnings", 0.78),
    "revenue": ("earnings", 0.75),
    "earnings": ("earnings", 0.80),
    "net income": ("earnings", 0.78),
    # Global macro — indirect but significant for India
    "fed": ("global_macro", 0.80),
    "federal reserve": ("global_macro", 0.82),
    "crude oil": ("global_macro", 0.85),
    "oil price": ("global_macro", 0.83),
    "dollar": ("global_macro", 0.72),
    "rupee": ("global_macro", 0.78),
    "usd": ("global_macro", 0.70),
    "china": ("global_macro", 0.65),
    "us economy": ("global_macro", 0.72),
    "recession": ("global_macro", 0.80),
    "tariff": ("global_macro", 0.75),
    "trade war": ("global_macro", 0.78),
    # Regulatory — can move sectors significantly
    "sebi": ("regulatory", 0.80),
    "regulation": ("regulatory", 0.72),
    "gst": ("regulatory", 0.70),
    "tax": ("regulatory", 0.68),
    "budget": ("regulatory", 0.85),
    "union budget": ("regulatory", 0.90),
    # Sector-specific triggers
    "banking": ("sector_news", 0.72),
    "pharma": ("sector_news", 0.68),
    "auto": ("sector_news", 0.68),
    "real estate": ("sector_news", 0.65),
    "infrastructure": ("sector_news", 0.65),
}

# Sentiment keywords — simple but effective for headline classification.
# Headlines are short (< 20 words) so keyword matching works well.
POSITIVE_KEYWORDS: set[str] = {
    "surge", "surges", "rally", "rallies", "gain", "gains", "soar", "soars",
    "jump", "jumps", "rise", "rises", "bull", "bullish", "boom", "booms",
    "record high", "all-time high", "outperform", "upgrade", "upbeat",
    "positive", "growth", "recovery", "rebound", "optimism", "strong",
    "beat", "beats", "exceeded", "above estimate", "record profit",
}

NEGATIVE_KEYWORDS: set[str] = {
    "crash", "crashes", "plunge", "plunges", "fall", "falls", "drop", "drops",
    "decline", "declines", "bear", "bearish", "slump", "slumps", "tumble",
    "sink", "sinks", "sell-off", "selloff", "panic", "fear", "crisis",
    "downgrade", "weak", "miss", "misses", "below estimate", "loss",
    "losses", "default", "fraud", "scam", "scandal", "warning",
    "recession", "slowdown", "contraction",
}

# Sector mapping — maps keywords in headlines to affected sectors.
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Banking": [
        "bank", "banking", "rbi", "repo rate", "interest rate", "npa",
        "hdfc", "icici", "sbi", "kotak", "axis", "loan", "credit",
    ],
    "IT": [
        "it sector", "tcs", "infosys", "wipro", "hcl tech", "tech",
        "software", "digital", "outsourcing",
    ],
    "Pharma": [
        "pharma", "drug", "fda", "medicine", "healthcare", "hospital",
        "cipla", "sun pharma", "dr reddy",
    ],
    "Auto": [
        "auto", "automobile", "car", "vehicle", "ev", "electric vehicle",
        "maruti", "tata motors", "mahindra",
    ],
    "Energy": [
        "oil", "crude", "gas", "energy", "reliance", "ongc", "petrol",
        "diesel", "fuel", "power",
    ],
    "Finance": [
        "nbfc", "finance", "insurance", "mutual fund", "bajaj finance",
        "lending", "fintech",
    ],
    "FMCG": [
        "fmcg", "consumer", "itc", "hindustan unilever", "nestle",
        "goods", "retail",
    ],
    "Metal": [
        "metal", "steel", "aluminium", "copper", "iron", "tata steel",
        "mining",
    ],
    "RealEstate": [
        "real estate", "realty", "housing", "property", "home loan",
        "emi", "dlf", "godrej properties",
    ],
    "Telecom": [
        "telecom", "airtel", "jio", "vodafone", "5g", "spectrum",
    ],
}

# Impact classification by news category
CATEGORY_IMPACT: dict[str, MarketImpact] = {
    "monetary_policy": MarketImpact.HIGH,
    "market_flow": MarketImpact.HIGH,
    "earnings": MarketImpact.HIGH,
    "global_macro": MarketImpact.MEDIUM,
    "market_event": MarketImpact.MEDIUM,
    "regulatory": MarketImpact.MEDIUM,
    "corporate_action": MarketImpact.LOW,
    "sector_news": MarketImpact.LOW,
}


def score_news_item(item: NewsItem) -> NewsItem:
    """Score a single news item through the 6-step CoT process.

    Transforms a raw NewsItem (from news_fetcher) into a scored item
    with sentiment, affected_sectors, relevance_score, market_impact,
    cot_reasoning, and advisor_note populated.

    Args:
        item: NewsItem with headline and source populated.
              Other fields will be overwritten by scoring.

    Returns:
        The same NewsItem object with all scoring fields populated.
    """
    headline_lower = item.headline.lower()
    reasoning_steps: list[str] = []

    # ── Step 1: Identify news category ──
    category, base_score = _identify_category(headline_lower)
    reasoning_steps.append(
        f"Step 1: Category identified as '{category}' "
        f"(base relevance: {base_score:.2f})."
    )

    # ── Step 2: Identify affected sectors ──
    affected_sectors = _identify_sectors(headline_lower)
    reasoning_steps.append(
        f"Step 2: Affected sectors: {affected_sectors if affected_sectors else 'none identified'}."
    )

    # ── Step 3: Assess market impact ──
    market_impact = CATEGORY_IMPACT.get(category, MarketImpact.LOW)
    reasoning_steps.append(
        f"Step 3: Market impact assessed as '{market_impact.value}' "
        f"based on category '{category}'."
    )

    # ── Step 4: Check recency ──
    recency_multiplier = _calculate_recency_multiplier(item.published_at)
    reasoning_steps.append(
        f"Step 4: Recency multiplier: {recency_multiplier:.2f} "
        f"(published: {item.published_at.strftime('%H:%M IST')})."
    )

    # ── Step 5: Calculate final relevance score ──
    sentiment = _classify_sentiment(headline_lower)
    sentiment_bonus = 0.05 if sentiment in (NewsSentiment.POSITIVE, NewsSentiment.NEGATIVE) else 0.0
    sector_bonus = min(len(affected_sectors) * 0.03, 0.10)

    relevance_score = min(
        (base_score + sentiment_bonus + sector_bonus) * recency_multiplier,
        1.0,
    )
    relevance_score = round(relevance_score, 2)

    reasoning_steps.append(
        f"Step 5: Final score = (base {base_score:.2f} + sentiment {sentiment_bonus:.2f} "
        f"+ sector {sector_bonus:.2f}) × recency {recency_multiplier:.2f} "
        f"= {relevance_score:.2f}."
    )

    # ── Step 6: Generate advisor note ──
    advisor_note = _generate_advisor_note(
        headline=item.headline,
        category=category,
        sentiment=sentiment,
        affected_sectors=affected_sectors,
        market_impact=market_impact,
    )
    reasoning_steps.append(f"Step 6: Advisor note generated.")

    # ── Populate the NewsItem ──
    item.sentiment = sentiment
    item.market_impact = market_impact
    item.affected_sectors = affected_sectors
    item.relevance_score = relevance_score
    item.cot_reasoning = " | ".join(reasoning_steps)
    item.advisor_note = advisor_note

    return item


def score_news_items(items: list[NewsItem]) -> list[NewsItem]:
    """Score all news items and sort by relevance.

    Primary entry point called by the pipeline (Step 4, after fetching).
    Scores each item through the 6-step CoT process, then sorts
    by relevance_score descending (most relevant first).

    Args:
        items: List of raw NewsItem objects from news_fetcher.

    Returns:
        Same list with all scoring fields populated, sorted by
        relevance_score descending. Items below min_news_relevance
        are NOT filtered here — that's the pipeline's job.
    """
    if not items:
        logger.info("No news items to score.")
        return []

    scored = [score_news_item(item) for item in items]
    scored.sort(key=lambda n: n.relevance_score, reverse=True)

    high_relevance = sum(1 for n in scored if n.relevance_score >= 0.70)
    logger.info(
        f"Scored {len(scored)} news items. "
        f"{high_relevance} items above 0.70 relevance threshold. "
        f"Top headline: \"{scored[0].headline[:60]}...\" "
        f"(score: {scored[0].relevance_score:.2f}, "
        f"sentiment: {scored[0].sentiment.value})."
    )

    return scored


def _identify_category(headline_lower: str) -> tuple[str, float]:
    """Step 1: Identify news category from headline keywords.

    Scans the headline for known keywords and returns the highest-scoring
    category match. If multiple keywords match, the one with the highest
    base score wins — this ensures "RBI rate cut" matches monetary_policy
    (0.92) rather than just "rbi" (0.85).

    Args:
        headline_lower: Lowercase headline text.

    Returns:
        Tuple of (category_name, base_relevance_score).
        Returns ("general", 0.40) if no keywords match.
    """
    best_category = "general"
    best_score = 0.40  # Default for unrecognized headlines

    for keyword, (category, score) in NEWS_CATEGORIES.items():
        if keyword in headline_lower and score > best_score:
            best_category = category
            best_score = score

    return best_category, best_score


def _identify_sectors(headline_lower: str) -> list[str]:
    """Step 2: Identify affected sectors from headline keywords.

    Scans the headline against sector keyword lists and returns
    all matching sectors. A headline can affect multiple sectors
    (e.g., "RBI rate hold" affects Banking, Finance, RealEstate).

    Args:
        headline_lower: Lowercase headline text.

    Returns:
        List of affected sector names. May be empty if no sector
        keywords are found.
    """
    affected: list[str] = []

    for sector_name, keywords in SECTOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword in headline_lower:
                if sector_name not in affected:
                    affected.append(sector_name)
                break  # One match per sector is enough

    return affected


def _classify_sentiment(headline_lower: str) -> NewsSentiment:
    """Step 3 (sub-step): Classify headline sentiment.

    Counts positive and negative keyword matches in the headline.
    If both positive and negative keywords are found → mixed.
    If only one type → that sentiment.
    If neither → neutral.

    This is intentionally simple — deep sentiment analysis is
    Module 2's job (Claude API). We just need coarse classification
    for relevance scoring and initial filtering.

    Args:
        headline_lower: Lowercase headline text.

    Returns:
        NewsSentiment enum member.
    """
    positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in headline_lower)
    negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in headline_lower)

    if positive_count > 0 and negative_count > 0:
        return NewsSentiment.MIXED
    elif positive_count > 0:
        return NewsSentiment.POSITIVE
    elif negative_count > 0:
        return NewsSentiment.NEGATIVE
    else:
        return NewsSentiment.NEUTRAL


def _calculate_recency_multiplier(published_at: datetime) -> float:
    """Step 4: Calculate recency multiplier for relevance scoring.

    Recent news matters more. The multiplier decays over time:
      < 1 hour old  → 1.00 (full weight)
      1–3 hours old → 0.90
      3–6 hours old → 0.80
      6–12 hours old → 0.70
      12–24 hours old → 0.60
      > 24 hours old → 0.50 (half weight — stale but still contextual)

    Args:
        published_at: Publication timestamp (IST).

    Returns:
        Multiplier between 0.50 and 1.00.
    """
    now = datetime.now(IST)

    # Handle timezone-naive datetimes gracefully
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=IST)

    age_hours = (now - published_at).total_seconds() / 3600

    if age_hours < 0:
        # Future timestamp (clock skew) — treat as fresh
        return 1.00
    elif age_hours < 1:
        return 1.00
    elif age_hours < 3:
        return 0.90
    elif age_hours < 6:
        return 0.80
    elif age_hours < 12:
        return 0.70
    elif age_hours < 24:
        return 0.60
    else:
        return 0.50


def _generate_advisor_note(
    headline: str,
    category: str,
    sentiment: NewsSentiment,
    affected_sectors: list[str],
    market_impact: MarketImpact,
) -> str:
    """Step 6: Generate a concise advisor note for the news item.

    The advisor reads this note to quickly understand the market
    implication without reading the full article. Must be actionable
    and sector-aware.

    Examples:
      "Rate hold positive for banking sector. Watch HDFCBANK, ICICIBANK for reaction."
      "Crude oil spike may pressure energy costs. Monitor rupee weakness."
      "Strong Q3 results beat estimates — positive for IT sector momentum."

    Args:
        headline: Original headline text.
        category: Identified news category.
        sentiment: Classified sentiment.
        affected_sectors: List of affected sector names.
        market_impact: Assessed impact level.

    Returns:
        1-2 sentence advisor note string.
    """
    sentiment_label = {
        NewsSentiment.POSITIVE: "positive",
        NewsSentiment.NEGATIVE: "negative",
        NewsSentiment.NEUTRAL: "neutral",
        NewsSentiment.MIXED: "mixed",
    }[sentiment]

    sector_str = ", ".join(affected_sectors[:3]) if affected_sectors else "broad market"

    # Category-specific note templates
    category_notes: dict[str, str] = {
        "monetary_policy": (
            f"Monetary policy update — {sentiment_label} for {sector_str}. "
            f"Watch rate-sensitive stocks for reaction at next open."
        ),
        "earnings": (
            f"Earnings update — {sentiment_label} signal for {sector_str}. "
            f"Review results detail for guidance impact."
        ),
        "market_flow": (
            f"Institutional flow activity — {sentiment_label} for {sector_str}. "
            f"FII/DII flows influence near-term direction."
        ),
        "global_macro": (
            f"Global macro event — {sentiment_label} impact on {sector_str}. "
            f"Monitor international cues for spillover effect on NSE."
        ),
        "market_event": (
            f"Market event — {sentiment_label} for {sector_str}. "
            f"Assess whether this changes the near-term trend."
        ),
        "regulatory": (
            f"Regulatory development — {sentiment_label} for {sector_str}. "
            f"Policy changes may alter sector dynamics."
        ),
        "corporate_action": (
            f"Corporate action announced — {sentiment_label} for {sector_str}. "
            f"Check if this affects holdings or watchlist."
        ),
        "sector_news": (
            f"Sector-specific news — {sentiment_label} for {sector_str}. "
            f"Review for individual stock impact."
        ),
    }

    note = category_notes.get(
        category,
        f"Market news — {sentiment_label} for {sector_str}. "
        f"Impact: {market_impact.value}.",
    )

    return note
