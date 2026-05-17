"""
SwingAdvisorBot — smoke_test.py
End-to-end pipeline test M1 → M5.
No Claude API needed — tests data flow only.

Usage:
    python smoke_test.py

Prerequisites:
    python seed_vijay.py  (run first to create profile)
"""

import asyncio
import sys
import traceback
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────
# Results tracker
# ─────────────────────────────────────────────────────────────

results = {
    "M1 Data Layer": None,
    "M5 User Profile": None,
    "M5 Memory Context": None,
    "M3 VIX Gate": None,
    "M4 Stock Screener": None,
    "M5 Trade Storage": None,
}


def mark(step: str, passed: bool, error: str = ""):
    results[step] = (passed, error)


# ─────────────────────────────────────────────────────────────
# Step 1 — M1: Fetch real NSE data
# ─────────────────────────────────────────────────────────────

async def step1_m1_data():
    print("=" * 58)
    print("  STEP 1 — M1: Fetch Real NSE Data")
    print("=" * 58)

    from module1_data_layer.config import DataFetchConfig
    from module1_data_layer.pipeline import run_data_pipeline

    tickers = ["HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK"]
    config = DataFetchConfig(max_stocks=5, max_news=0, max_economic_events=0)

    market_data = await run_data_pipeline(tickers=tickers, config=config)

    stock_count = len(market_data.stocks)
    vix = market_data.india_vix
    status = market_data.market_status.value
    freshness = market_data.data_freshness.value
    pipeline = market_data.pipeline_status.value

    assert stock_count > 0, f"No stocks fetched (got {stock_count})"
    assert vix is not None, "VIX not available"
    assert status is not None, "Market status missing"

    print(f"  Stocks fetched:  {stock_count}")
    print(f"  India VIX:       {vix}")
    print(f"  Market status:   {status}")
    print(f"  Data freshness:  {freshness}")
    print(f"  Pipeline health: {pipeline}")
    print()
    for s in market_data.stocks:
        print(f"    {s.ticker:12s} ₹{s.price:>10.2f}  "
              f"{s.change_pct:>+6.2f}%  {s.advisor_flag.value}")
    print()
    print(f"  ✅ STEP 1 PASSED — {stock_count} stocks, VIX={vix}, status={status}")

    mark("M1 Data Layer", True)
    return market_data


# ─────────────────────────────────────────────────────────────
# Step 2 — M5: Load Vijay's profile
# ─────────────────────────────────────────────────────────────

def step2_m5_profile():
    print()
    print("=" * 58)
    print("  STEP 2 — M5: Load Vijay's Profile")
    print("=" * 58)

    from module5_memory.engine import memory_engine

    profile = memory_engine.get_user_profile("XCU700")

    assert profile is not None, "Profile not found — run seed_vijay.py first"
    assert profile.name == "Vijay", f"Name mismatch: {profile.name}"
    assert profile.capital == Decimal("50000.00"), f"Capital mismatch: {profile.capital}"
    assert profile.risk_tolerance == "moderate", f"Tolerance mismatch: {profile.risk_tolerance}"

    print(f"  User ID:    {profile.user_id}")
    print(f"  Name:       {profile.name}")
    print(f"  Capital:    ₹{profile.capital:,.2f}")
    print(f"  Tolerance:  {profile.risk_tolerance}")
    print(f"  Win rate:   {profile.win_rate}%")
    print()
    print(f"  ✅ STEP 2 PASSED — Profile loaded: {profile.name}, "
          f"₹{profile.capital}, {profile.risk_tolerance}")

    mark("M5 User Profile", True)


# ─────────────────────────────────────────────────────────────
# Step 3 — M5: Get memory context
# ─────────────────────────────────────────────────────────────

def step3_m5_context():
    print()
    print("=" * 58)
    print("  STEP 3 — M5: Memory Context")
    print("=" * 58)

    from module5_memory.engine import memory_engine

    ctx = memory_engine.get_memory_context(
        "XCU700",
        query="NSE market analysis swing trade",
    )

    assert ctx.within_budget, f"Context exceeds 300 token budget: {ctx.token_estimate}"

    chunks = ctx.chunks_used
    print(f"  Token estimate:   {ctx.token_estimate} / 300")
    print(f"  Within budget:    {ctx.within_budget}")
    print(f"  Chunks retrieved: {chunks}")
    if ctx.text:
        preview = ctx.text[:120].replace("\n", " ")
        print(f"  Preview:          {preview}...")
    print()
    print(f"  ✅ STEP 3 PASSED — {ctx.token_estimate} tokens, "
          f"{chunks} chunks, within budget")

    mark("M5 Memory Context", True)


# ─────────────────────────────────────────────────────────────
# Step 4 — M3: VIX gate check
# ─────────────────────────────────────────────────────────────

def step4_m3_vix_gate(market_data):
    print()
    print("=" * 58)
    print("  STEP 4 — M3: VIX Gate Check")
    print("=" * 58)

    from module3_risk_engine.engine import risk_engine
    from module5_memory.engine import memory_engine

    vix = Decimal(str(market_data.india_vix))
    tolerance = memory_engine.get_risk_tolerance()

    status = risk_engine.get_vix_gate_status(
        vix_value=vix,
        tolerance=tolerance,
    )

    assert status.gate is not None, "Gate status missing"
    assert status.vix_signal is not None, "VIX signal missing"

    print(f"  India VIX:    {status.vix_value}")
    print(f"  VIX limit:    {status.vix_limit}")
    print(f"  VIX signal:   {status.vix_signal}")
    print(f"  Gate:         {status.gate.value}")
    print(f"  Tolerance:    {status.tolerance.value}")
    print(f"  Advisor note: {status.advisor_note[:100]}...")
    print()
    print(f"  ✅ STEP 4 PASSED — VIX={status.vix_value}, "
          f"gate={status.gate.value}")

    mark("M3 VIX Gate", True)


# ─────────────────────────────────────────────────────────────
# Step 5 — M4: Stock screening
# ─────────────────────────────────────────────────────────────

def step5_m4_screener(market_data):
    print()
    print("=" * 58)
    print("  STEP 5 — M4: Stock Screener")
    print("=" * 58)

    from module4_setup_generator.technical.stock_screener import stock_screener

    candidates = stock_screener.screen(
        market_data=market_data,
        max_candidates=10,
    )

    total = len(market_data.stocks)
    filtered = len(candidates)
    skipped = total - filtered

    print(f"  Total stocks:   {total}")
    print(f"  Candidates:     {filtered}")
    print(f"  Skipped:        {skipped}")
    print()

    if candidates:
        for c in candidates:
            print(f"    ✓ {c.ticker:12s} — {c.advisor_flag.value}")
    else:
        print("    (no candidates passed screening)")
        # Show what was skipped
        for s in market_data.stocks:
            print(f"    ✗ {s.ticker:12s} — {s.advisor_flag.value} (skipped)")

    print()
    print(f"  ✅ STEP 5 PASSED — {filtered} candidates from {total} stocks")

    mark("M4 Stock Screener", True)


# ─────────────────────────────────────────────────────────────
# Step 6 — M5: Store and retrieve a test trade
# ─────────────────────────────────────────────────────────────

def step6_m5_trade():
    print()
    print("=" * 58)
    print("  STEP 6 — M5: Trade Storage")
    print("=" * 58)

    from module5_memory.database.schema import get_connection
    from module5_memory.engine import memory_engine
    from module5_memory.models import TradeRecord

    trade = TradeRecord(
        user_id="XCU700",
        ticker="HDFCBANK",
        sector="Banking",
        entry_price=Decimal("769.00"),
        stop_loss=Decimal("727.00"),
        target_price=Decimal("888.00"),
        shares=13,
        setup_source="manual",
    )

    # Save
    trade_id = memory_engine.save_trade(trade)
    assert trade_id, "Trade ID not returned"
    print(f"  Trade saved:    {trade_id}")

    # Retrieve
    loaded = memory_engine.get_trade(trade_id)
    assert loaded is not None, f"Trade {trade_id} not found after save"
    assert loaded.ticker == "HDFCBANK", f"Ticker mismatch: {loaded.ticker}"
    assert loaded.shares == 13, f"Shares mismatch: {loaded.shares}"
    assert loaded.entry_price == Decimal("769.00"), f"Entry mismatch: {loaded.entry_price}"

    print(f"  Retrieved:      {loaded.ticker}, {loaded.shares} shares @ ₹{loaded.entry_price}")
    print(f"  Status:         {loaded.status.value}")
    print(f"  Stop loss:      ₹{loaded.stop_loss}")
    print(f"  Target:         ₹{loaded.target_price}")

    # Cleanup — delete the test trade directly from SQLite
    conn = get_connection()
    cursor = conn.execute("DELETE FROM trades WHERE trade_id = ?", (trade_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()

    verify = memory_engine.get_trade(trade_id)
    assert verify is None, "Trade still exists after cleanup"
    print(f"  Cleaned up:     {trade_id} deleted ✓")
    print()
    print(f"  ✅ STEP 6 PASSED — Save → retrieve → cleanup successful")

    mark("M5 Trade Storage", True)


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

async def main():
    print()
    print("╔" + "═" * 56 + "╗")
    print("║    SWING ADVISOR BOT — END-TO-END SMOKE TEST         ║")
    print("║    M1 → M3 → M4 → M5 pipeline validation            ║")
    print("╚" + "═" * 56 + "╝")
    print()

    market_data = None

    # Step 1 — M1
    try:
        market_data = await step1_m1_data()
    except Exception as e:
        print(f"  ❌ STEP 1 FAILED — {e}")
        traceback.print_exc()
        mark("M1 Data Layer", False, str(e))

    # Step 2 — M5 profile
    try:
        step2_m5_profile()
    except Exception as e:
        print(f"  ❌ STEP 2 FAILED — {e}")
        traceback.print_exc()
        mark("M5 User Profile", False, str(e))

    # Step 3 — M5 context
    try:
        step3_m5_context()
    except Exception as e:
        print(f"  ❌ STEP 3 FAILED — {e}")
        traceback.print_exc()
        mark("M5 Memory Context", False, str(e))

    # Step 4 — M3 VIX gate (needs M1 data)
    if market_data:
        try:
            step4_m3_vix_gate(market_data)
        except Exception as e:
            print(f"  ❌ STEP 4 FAILED — {e}")
            traceback.print_exc()
            mark("M3 VIX Gate", False, str(e))
    else:
        print(f"\n  ⚠️  STEP 4 SKIPPED — M1 data not available")
        mark("M3 VIX Gate", False, "M1 data not available")

    # Step 5 — M4 screener (needs M1 data)
    if market_data:
        try:
            step5_m4_screener(market_data)
        except Exception as e:
            print(f"  ❌ STEP 5 FAILED — {e}")
            traceback.print_exc()
            mark("M4 Stock Screener", False, str(e))
    else:
        print(f"\n  ⚠️  STEP 5 SKIPPED — M1 data not available")
        mark("M4 Stock Screener", False, "M1 data not available")

    # Step 6 — M5 trade storage
    try:
        step6_m5_trade()
    except Exception as e:
        print(f"  ❌ STEP 6 FAILED — {e}")
        traceback.print_exc()
        mark("M5 Trade Storage", False, str(e))

    # ── Final Summary ──
    print()
    print("=" * 58)
    print("  SMOKE TEST COMPLETE")
    print("=" * 58)

    all_passed = True
    for step, result in results.items():
        if result is None:
            icon = "⚠️"
            all_passed = False
        elif result[0]:
            icon = "✅"
        else:
            icon = "❌"
            all_passed = False
        pad = 22 - len(step)
        print(f"  {step}:{' ' * pad}{icon}")
        if result and not result[0] and result[1]:
            print(f"    → {result[1][:80]}")

    print()
    if all_passed:
        print("  Ready to build M6:    ✅")
    else:
        print("  Ready to build M6:    ❌ (fix failures first)")

    print("=" * 58)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
