"""
SwingAdvisorBot — Module 7: Education Layer
config.py — Concept library, difficulty rules, Claude prompts, token budgets

Concept library: 10 pre-built concepts with fallback lessons + quizzes.
Difficulty rules: upgrade at ≥80 % for 2 weeks, downgrade at <60 % for 1 week.
Token budget: 1 Claude call/day, 1160 tokens total (760 input + 400 output).
"""

from __future__ import annotations

from module7_education.models import ConceptDefinition, ConceptTrigger, DifficultyLevel

# ═══════════════════════════════════════════════════════════
# TRIGGER → CONCEPT MAPPING
# ═══════════════════════════════════════════════════════════

TRIGGER_MAP: list[ConceptTrigger] = [
    ConceptTrigger(
        trigger="unusual_activity",
        concepts=["volume_analysis", "institutional_buying"],
    ),
    ConceptTrigger(
        trigger="above_average",
        concepts=["volume_analysis"],
    ),
    ConceptTrigger(
        trigger="breakout_watch",
        concepts=["support_resistance", "breakout_confirmation"],
    ),
    ConceptTrigger(
        trigger="accumulation_zone",
        concepts=["support_resistance", "position_sizing"],
    ),
    ConceptTrigger(
        trigger="value_zone",
        concepts=["support_resistance", "risk_reward"],
    ),
    ConceptTrigger(
        trigger="vix_high",
        concepts=["india_vix", "risk_reward"],
    ),
    ConceptTrigger(
        trigger="vix_extreme",
        concepts=["india_vix"],
    ),
    ConceptTrigger(
        trigger="strong_momentum",
        concepts=["trailing_stop", "partial_profit"],
    ),
    ConceptTrigger(
        trigger="target_near",
        concepts=["partial_profit", "trailing_stop"],
    ),
    ConceptTrigger(
        trigger="sector_bullish",
        concepts=["sector_rotation", "fii_dii_data"],
    ),
    ConceptTrigger(
        trigger="sector_bearish",
        concepts=["sector_rotation", "fii_dii_data"],
    ),
    ConceptTrigger(
        trigger="market_mood",
        concepts=["fii_dii_data", "india_vix"],
    ),
    ConceptTrigger(
        trigger="any_setup",
        concepts=["risk_reward", "position_sizing"],
    ),
    ConceptTrigger(
        trigger="any",
        concepts=["candlestick_basics"],
    ),
]


# ═══════════════════════════════════════════════════════════
# CONCEPT LIBRARY — 10 concepts with full fallback data
# ═══════════════════════════════════════════════════════════

CONCEPT_LIBRARY: dict[str, ConceptDefinition] = {
    "volume_analysis": ConceptDefinition(
        concept="volume_analysis",
        definition=(
            "Volume confirms price moves. High volume = institutional "
            "participation. Low volume = retail only. Always check both."
        ),
        triggers=["unusual_activity", "above_average"],
        difficulty=DifficultyLevel.BEGINNER,
        fallback_lesson=(
            "When a stock rises on above-average volume, institutions are "
            "buying. Price tells you WHAT happened. Volume tells you WHY. "
            "Always check volume before entering any trade — high volume "
            "confirms the move is real."
        ),
        fallback_quiz_question="A stock rises 3% on 0.5x average volume. Strong or weak signal?",
        fallback_quiz_options={"A": "Strong — price is rising", "B": "Weak — volume doesn't confirm"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "Low volume = no institutional support. Moves without volume often reverse."
        ),
    ),
    "support_resistance": ConceptDefinition(
        concept="support_resistance",
        definition=(
            "Support = price floor where buyers step in. Resistance = price "
            "ceiling where sellers appear. 52-week high/low are key levels. "
            "Price memory is real."
        ),
        triggers=["breakout_watch", "value_zone", "accumulation_zone"],
        difficulty=DifficultyLevel.BEGINNER,
        fallback_lesson=(
            "Support is where buyers consistently step in — the price floor. "
            "Resistance is where sellers appear — the price ceiling. "
            "52-week highs and lows are the strongest levels because every "
            "trader watches them. Price has memory."
        ),
        fallback_quiz_question="A stock bounces off the same price 3 times. What is that level?",
        fallback_quiz_options={"A": "Resistance", "B": "Support"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "A price floor that buyers defend repeatedly is called support."
        ),
    ),
    "india_vix": ConceptDefinition(
        concept="india_vix",
        definition=(
            "Fear index for Indian markets. Below 15 = calm. 15-20 = moderate. "
            "Above 20 = high fear. Above 25 = extreme. Controls position sizing."
        ),
        triggers=["vix_high", "vix_extreme"],
        difficulty=DifficultyLevel.BEGINNER,
        fallback_lesson=(
            "India VIX measures how much the market expects prices to swing "
            "over the next 30 days. High VIX = fear, low VIX = calm. "
            "Our rule: VIX above 20 means no new swing trades. "
            "Cash is a position in high-fear environments."
        ),
        fallback_quiz_question="VIX is at 22. Should you enter a new swing trade?",
        fallback_quiz_options={"A": "Yes — opportunity in volatility", "B": "No — VIX above our 20 limit"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "VIX above 20 means markets are fearful. Our rule is to avoid "
            "new swing trades until VIX drops below 20."
        ),
    ),
    "risk_reward": ConceptDefinition(
        concept="risk_reward",
        definition=(
            "Ratio of potential loss to potential gain. Minimum 1:2. Ideal 1:3. "
            "Even with 50% win rate — 1:3 R/R is profitable over time."
        ),
        triggers=["any_setup"],
        difficulty=DifficultyLevel.BEGINNER,
        fallback_lesson=(
            "Risk/reward compares what you could lose vs what you could gain. "
            "A 1:3 ratio means you risk ₹1 to make ₹3. Even if you're wrong "
            "half the time, 1:3 R/R keeps you profitable. Never take a trade "
            "below 1:2."
        ),
        fallback_quiz_question="Entry ₹100, stop ₹95, target ₹115. What is the R/R ratio?",
        fallback_quiz_options={"A": "1:3 — risk 5, reward 15", "B": "1:1 — risk 5, reward 5"},
        fallback_quiz_correct="A",
        fallback_quiz_explanation=(
            "Risk = 100-95 = ₹5. Reward = 115-100 = ₹15. Ratio = 5:15 = 1:3."
        ),
    ),
    "position_sizing": ConceptDefinition(
        concept="position_sizing",
        definition=(
            "2% rule: never risk more than 2% of capital per trade. Protects "
            "from catastrophic loss. Formula: shares = max_risk / risk_per_share."
        ),
        triggers=["any_setup"],
        difficulty=DifficultyLevel.BEGINNER,
        fallback_lesson=(
            "The 2% rule: never risk more than 2% of your total capital on "
            "a single trade. With ₹50,000 capital, max risk per trade is ₹1,000. "
            "This protects you from a string of losses wiping out your account. "
            "Position size = max risk ÷ risk per share."
        ),
        fallback_quiz_question="Capital ₹50,000. Max risk per trade using the 2% rule?",
        fallback_quiz_options={"A": "₹5,000", "B": "₹1,000"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "2% of ₹50,000 = ₹1,000. Never risk more than this on a single trade."
        ),
    ),
    "trailing_stop": ConceptDefinition(
        concept="trailing_stop",
        definition=(
            "Move stop loss up as price rises. Locks in profits while letting "
            "winners run. Stop only moves UP, never DOWN. Creates risk-free positions."
        ),
        triggers=["strong_momentum", "target_near"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        fallback_lesson=(
            "A trailing stop moves up with the price — never down. "
            "Once price rises enough, move stop to entry = risk-free trade. "
            "This lets winners run while protecting profits. "
            "Key rule: stop only moves in your favour, never backwards."
        ),
        fallback_quiz_question="Your trailing stop is at ₹105. Stock dips to ₹103. What do you do?",
        fallback_quiz_options={"A": "Lower stop to ₹101", "B": "Keep stop at ₹105"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "Trailing stops only move UP, never down. The stop stays at ₹105."
        ),
    ),
    "sector_rotation": ConceptDefinition(
        concept="sector_rotation",
        definition=(
            "Money flows from weak sectors to strong ones. Follow FII/DII data. "
            "Trading with sector momentum improves win rate significantly."
        ),
        triggers=["sector_bullish", "sector_bearish"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        fallback_lesson=(
            "Money doesn't leave the market — it rotates between sectors. "
            "When Banking is strong and IT is weak, institutions are moving "
            "money. Trading with sector momentum (buying strong sectors) "
            "improves your win rate. Check FII/DII data daily."
        ),
        fallback_quiz_question="Banking sector is +2%, IT is -1%. Where is money flowing?",
        fallback_quiz_options={"A": "Into IT — it's cheaper", "B": "Into Banking — follow the flow"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "Money rotates from weak sectors to strong ones. Follow the flow, "
            "don't fight it."
        ),
    ),
    "partial_profit": ConceptDefinition(
        concept="partial_profit",
        definition=(
            "Book 50% at first target. Move stop to entry on remaining 50%. "
            "Risk-free position on remainder. Balances locking gains vs letting run."
        ),
        triggers=["target_near", "strong_momentum"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        fallback_lesson=(
            "Partial profit booking: sell 50% when the stock hits your first "
            "target. Then move your stop loss to entry on the remaining 50%. "
            "Now you have a risk-free position that can still run higher. "
            "This balances locking in gains with letting winners run."
        ),
        fallback_quiz_question="Stock hits your target. Should you sell 100% or 50%?",
        fallback_quiz_options={"A": "100% — take all profit now", "B": "50% — let the rest run risk-free"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "Selling 50% locks in profit. Moving stop to entry on the rest "
            "makes it risk-free while allowing more upside."
        ),
    ),
    "fii_dii_data": ConceptDefinition(
        concept="fii_dii_data",
        definition=(
            "FII = Foreign Institutional Investors. DII = Domestic. "
            "FII buying = bullish signal. FII selling = caution. "
            "Check daily before market analysis."
        ),
        triggers=["market_mood"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        fallback_lesson=(
            "FII (Foreign) and DII (Domestic) institutions move markets. "
            "When FIIs are buying, it is a bullish signal — foreign money "
            "is flowing in. When FIIs sell, be cautious. "
            "Check FII/DII data every morning before analysing the market."
        ),
        fallback_quiz_question="FIIs sold ₹2,000 Cr today. What does this signal?",
        fallback_quiz_options={"A": "Bullish — buying opportunity", "B": "Caution — foreign money leaving"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "FII selling means foreign institutions are pulling money out. "
            "This is a cautionary signal — wait for buying to resume."
        ),
    ),
    "candlestick_basics": ConceptDefinition(
        concept="candlestick_basics",
        definition=(
            "Each candle = one session. Body = open to close. Wicks = extremes. "
            "Green = closed higher. Red = closed lower. Patterns predict direction."
        ),
        triggers=["any"],
        difficulty=DifficultyLevel.BEGINNER,
        fallback_lesson=(
            "Each candlestick shows one trading session. The body shows open "
            "to close. Wicks show the high and low extremes. Green candle = "
            "stock closed higher than it opened. Red = closed lower. "
            "Long lower wick = buyers defended that level strongly."
        ),
        fallback_quiz_question="A candle has a long lower wick. What does this mean?",
        fallback_quiz_options={"A": "Sellers are in control", "B": "Buyers defended the low"},
        fallback_quiz_correct="B",
        fallback_quiz_explanation=(
            "A long lower wick means the price dipped but buyers pushed it "
            "back up. Buyers are defending that level."
        ),
    ),
}


# ═══════════════════════════════════════════════════════════
# CONCEPT LIST (ordered — used for fallback rotation)
# ═══════════════════════════════════════════════════════════

CONCEPT_ORDER: list[str] = [
    "volume_analysis",
    "support_resistance",
    "india_vix",
    "risk_reward",
    "position_sizing",
    "trailing_stop",
    "sector_rotation",
    "partial_profit",
    "fii_dii_data",
    "candlestick_basics",
]


# ═══════════════════════════════════════════════════════════
# DIFFICULTY RULES
# ═══════════════════════════════════════════════════════════

UPGRADE_THRESHOLD_PCT = 80.0   # ≥80% for 2 consecutive weeks → upgrade
UPGRADE_WEEKS_REQUIRED = 2
DOWNGRADE_THRESHOLD_PCT = 60.0  # <60% for 1 week → downgrade

DIFFICULTY_LADDER: list[DifficultyLevel] = [
    DifficultyLevel.BEGINNER,
    DifficultyLevel.BEGINNER_PLUS,
    DifficultyLevel.INTERMEDIATE,
    DifficultyLevel.INTERMEDIATE_PLUS,
]

NO_REPEAT_DAYS = 7  # never teach same concept within 7 days


# ═══════════════════════════════════════════════════════════
# TOKEN BUDGETS
# ═══════════════════════════════════════════════════════════

LESSON_TOKEN_BUDGET = {
    "system_prompt": 380,
    "concept_definition": 100,
    "market_anchor": 150,
    "learning_history": 50,
    "instruction": 80,
    "total_input": 760,
    "output_budget": 400,
    "grand_total": 1160,
}

LESSON_BRIEF_MAX_TOKENS = 200
LESSON_FULL_MAX_TOKENS = 400
QUIZ_MAX_TOKENS = 100

CLAUDE_MODEL = "claude-opus-4-5-20250514"


# ═══════════════════════════════════════════════════════════
# CLAUDE PROMPTS
# ═══════════════════════════════════════════════════════════

EDUCATION_SYSTEM_PROMPT = """\
You are a senior trading mentor teaching Indian retail investors \
about NSE swing trading.

Your teaching style:
→ Always use real market examples from today
→ Plain English — no jargon without explanation
→ One concept per lesson — not multiple
→ Make it memorable with a real story
→ End with one practical takeaway
→ Quiz must have one clearly correct answer

Respond in valid JSON only.
Start with { end with }
No markdown. No backticks. No text outside JSON.

Required structure:
{
  "title": "Concept Name — Hook in subtitle",
  "lesson_text": "3-4 sentences. Use real data.",
  "real_example": "How today's market shows this",
  "key_takeaway": "One sentence. Action oriented.",
  "quiz_question": "One clear question",
  "quiz_option_a": "First option",
  "quiz_option_b": "Second option",
  "quiz_correct": "A or B",
  "quiz_explanation": "Why correct answer is right"
}

Keep lesson_text under 150 words.
Keep quiz_question under 100 characters.
Total response under 400 tokens.
Speak directly to the user by name."""

EDUCATION_USER_TEMPLATE = """\
User: {user_name}
Current difficulty: {difficulty}
Concept to teach: {concept}
Concept definition: {definition}

Today's real market anchor:
{market_anchor}

Concepts already taught this week:
{taught_this_week}

Generate a lesson for {concept} anchored to today's real market data above.
Make {user_name} understand this concept using what actually happened today."""
