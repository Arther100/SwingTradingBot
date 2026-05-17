"""
SwingAdvisorBot — Module 2 Integration Tests
=============================================

Run:  python test_m2.py

Tests:
  1. Claude API Connection         — verify API key + raw call
  2. Advisor Personality Test       — full pipeline M1 → M2
  3. Quality Gate Check             — self-reflection on output
  4. Token Budget Check             — verify within 3000 limit

Requirements:
  - .env file with ANTHROPIC_API_KEY set
  - Kite Connect token valid (for M1 data fetch)
  - Market hours not required (works with cached/closed data)
"""

import asyncio
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

# Shared ticker list for all tests — 10 Nifty 50 stocks
TICKERS = [
    "HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "WIPRO",
]


# ──────────────────────────────────────────────
# Test 1 — Claude API Connection
# ──────────────────────────────────────────────
async def test_connection():
    """Verify Claude API key works with a minimal call."""
    print("\n" + "=" * 50)
    print("TEST 1 — Claude API Connection")
    print("=" * 50)

    from module2_analysis_engine.claude_client import claude_client

    # call_claude_raw returns (response_text, token_count, usage_dict)
    response_text, token_count, usage = await claude_client.call_claude_raw(
        system_prompt="You are a test assistant. Reply in one sentence.",
        user_message="Say hello and confirm you are working.",
    )

    print(f"✅ Claude API connected")
    print(f"Response: {response_text}")
    print(f"Tokens used: {token_count}")
    print(f"Usage: {usage}")


# ──────────────────────────────────────────────
# Test 2 — Advisor Personality Test
# ──────────────────────────────────────────────
async def test_personality():
    """Run full M1 → M2 pipeline and verify all advisor fields."""
    print("\n" + "=" * 50)
    print("TEST 2 — Advisor Personality Test")
    print("=" * 50)

    from module1_data_layer.config import DataFetchConfig
    from module1_data_layer.pipeline import run_data_pipeline
    from module2_analysis_engine.engine import analyse_market

    # Step 1: Get real data from M1
    print("Fetching real market data from M1...")
    config = DataFetchConfig()
    market_data = await run_data_pipeline(
        tickers=TICKERS,
        config=config,
    )
    print(f"✅ M1 data fetched: {len(market_data.stocks)} stocks")
    print(f"Market status: {market_data.market_status}")

    # Step 2: Run M2 analysis
    print("\nRunning M2 analysis...")
    result = await analyse_market(
        market_data=market_data,
        user_context=None,  # no memory yet — M5 not built
    )

    # Step 3: Check all fields present
    # AnalysisResult wraps MarketAnalysis in result.analysis
    analysis = result.analysis

    print("\n=== ADVISOR OUTPUT ===")
    print(f"Market mood: {analysis.market_mood.value}")
    print(f"Mood confidence: {analysis.mood_confidence:.2f}")
    print(f"Situation: {analysis.situation}")
    print(f"Reasoning: {analysis.reasoning}")
    print(f"Action: {analysis.action}")
    print(f"Risk: {analysis.risk}")
    print(f"Lesson: {analysis.lesson}")
    print(f"CoT: {analysis.cot_reasoning[:200]}...")
    print(f"Analysis depth: {analysis.analysis_depth.value}")
    print(f"Tokens used: {result.total_tokens}")
    print(f"API latency: {result.api_latency_ms}ms")
    print(f"Total latency: {result.total_latency_ms}ms")
    print(f"From cache: {result.from_cache}")
    print(f"Retries: {result.retry_count}")

    # Validate all critical fields are present and non-empty
    fields_ok = all([
        analysis.situation,
        analysis.reasoning,
        analysis.action,
        analysis.risk,
        analysis.lesson,
    ])

    if fields_ok:
        print(f"\n✅ All fields present")
    else:
        print(f"\n❌ Missing fields")

    return result


# ──────────────────────────────────────────────
# Test 3 — Quality Gate Check
# ──────────────────────────────────────────────
async def test_quality():
    """Run quality checker on analysis output."""
    print("\n" + "=" * 50)
    print("TEST 3 — Quality Gate Check")
    print("=" * 50)

    from module1_data_layer.config import DataFetchConfig
    from module1_data_layer.pipeline import run_data_pipeline
    from module2_analysis_engine.engine import analyse_market
    from module2_analysis_engine.quality_checker import quality_checker

    config = DataFetchConfig()
    market_data = await run_data_pipeline(
        tickers=TICKERS,
        config=config,
    )
    result = await analyse_market(
        market_data=market_data,
        user_context=None,
    )

    # QualityChecker.check() takes MarketAnalysis, not AnalysisResult
    report = quality_checker.check(analysis=result.analysis)

    print("\n=== QUALITY REPORT ===")
    print(f"Verdict: {report.verdict.value}")
    print(f"Situation OK:  {'✅' if report.situation_ok else '❌'}")
    print(f"Reasoning OK:  {'✅' if report.reasoning_ok else '❌'}")
    print(f"Action OK:     {'✅' if report.action_ok else '❌'}")
    print(f"Risk OK:       {'✅' if report.risk_ok else '❌'}")
    print(f"Lesson OK:     {'✅' if report.lesson_ok else '❌'}")
    print(f"CoT present:   {'✅' if report.cot_present else '❌'}")
    print(f"No N/A fields: {'✅' if report.no_na_fields else '❌'}")
    print(f"--- Self-reflection Q1-Q4 ---")
    print(f"Q1 Advisor satisfied: {'✅' if report.advisor_satisfied else '❌'}")
    print(f"Q2 Full structure:    {'✅' if report.has_full_structure else '❌'}")
    print(f"Q3 Personalised:      {'✅' if report.is_personalised else '❌'}")
    print(f"Q4 Honest:            {'✅' if report.is_honest else '❌'}")

    if report.missing_fields:
        print(f"Missing fields: {report.missing_fields}")
    if report.shallow_fields:
        print(f"Shallow fields: {report.shallow_fields}")
    if report.issues:
        print(f"Issues: {report.issues}")

    passed = report.verdict.value == "PASSED"
    print(f"\nOverall: {'✅ PASSED' if passed else '❌ ' + report.verdict.value}")


# ──────────────────────────────────────────────
# Test 4 — Token Budget Check
# ──────────────────────────────────────────────
async def test_tokens():
    """Verify total tokens stay within the 3000 hard limit."""
    print("\n" + "=" * 50)
    print("TEST 4 — Token Budget Check")
    print("=" * 50)

    from module1_data_layer.config import DataFetchConfig
    from module1_data_layer.pipeline import run_data_pipeline
    from module2_analysis_engine.config import HARD_TOKEN_LIMIT
    from module2_analysis_engine.engine import analyse_market

    config = DataFetchConfig()
    market_data = await run_data_pipeline(
        tickers=TICKERS,
        config=config,
    )
    result = await analyse_market(
        market_data=market_data,
        user_context=None,
    )

    budget = HARD_TOKEN_LIMIT  # 3000
    within = result.total_tokens <= budget

    print("\n=== TOKEN REPORT ===")
    print(f"Input tokens:  {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Total tokens:  {result.total_tokens}")
    print(f"Budget:        {budget}")
    print(f"Remaining:     {budget - result.total_tokens} tokens")
    print(f"Status: {'✅ Within budget' if within else '❌ Over budget'}")
    print("(Remaining budget goes to RAG context in M5)")


# ──────────────────────────────────────────────
# Test 5 — Token Diagnosis
# ──────────────────────────────────────────────
async def diagnose_tokens():
    """Diagnose token budget breakdown — where are tokens going?"""
    print("\n" + "=" * 50)
    print("TEST 5 — Token Budget Diagnosis")
    print("=" * 50)

    from module1_data_layer.config import DataFetchConfig
    from module1_data_layer.pipeline import run_data_pipeline

    config = DataFetchConfig()
    market_data = await run_data_pipeline(
        tickers=TICKERS,
        config=config,
    )

    # Check M1 output size first
    m1_tokens = market_data.estimate_tokens()
    system_prompt_est = 380
    cot_instruction_est = 180
    user_memory_est = 300

    print(f"\n=== TOKEN DIAGNOSIS ===")
    print(f"M1 MarketData tokens:   {m1_tokens}")
    print(f"System prompt tokens:   ~{system_prompt_est}")
    print(f"CoT instruction tokens: ~{cot_instruction_est}")
    print(f"User memory tokens:     ~{user_memory_est}")
    print(f"─────────────────────────")
    total_input = m1_tokens + system_prompt_est + cot_instruction_est + user_memory_est
    print(f"Total input estimate:   {total_input}")
    print(f"Output budget left:     {3000 - total_input}")

    if total_input > 2200:
        print(f"\n❌ INPUT OVER BUDGET by {total_input - 2200} tokens")
        print(f"Fix: Reduce max_stocks in DataFetchConfig")
        print(f"     Current: {config.max_stocks} stocks")
        print(f"     Try: 8 stocks")
    else:
        print(f"\n✅ Input within budget ({total_input}/2200)")


# ──────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────
async def main():
    """Run all tests sequentially."""
    tests = [
        ("Test 1 — Claude API Connection", test_connection),
        ("Test 2 — Advisor Personality", test_personality),
        ("Test 3 — Quality Gate", test_quality),
        ("Test 4 — Token Budget", test_tokens),
        ("Test 5 — Token Diagnosis", diagnose_tokens),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED: {e}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
