"""
SwingAdvisorBot — Module 3 Integration Tests
=============================================

Run:  python test_m3.py

Tests:
  1. APPROVED Trade          — HDFCBANK, good R/R, all checks pass
  2. REJECTED: Bad R/R       — target too close, R/R < 2.0
  3. REJECTED: VIX Gate      — VIX=27.5, gate closed for moderate
  4. REDUCE_SIZE             — requested 30 shares, only 13 allowed
  5. REJECTED: Sector Overexposure — 78% Banking already, adding more
  6. Full Pipeline (Engine)  — end-to-end via risk_engine.calculate_risk()

Requirements:
  - No .env needed (M3 is pure Python + Decimal, no API calls)
  - No market hours needed (all inputs are explicit)

All expected values verified by hand against the few-shot examples.
"""

import sys
import traceback
from decimal import Decimal


# ──────────────────────────────────────────────
# Test 1 — APPROVED Trade
# ──────────────────────────────────────────────
def test_approved_trade():
    """Few-shot 1: HDFCBANK, entry=1623, target=1900, stop=1548.

    Expected:
      verdict    = APPROVED
      shares     = 13
      position   = ₹21,099.00
      risk       = ₹975.00 (1.95% of ₹50K)
      R/R        = 1:3.69
    """
    print("\n" + "=" * 50)
    print("TEST 1 — APPROVED Trade (HDFCBANK)")
    print("=" * 50)

    from module3_risk_engine.engine import risk_engine

    report = risk_engine.calculate_risk(
        ticker="HDFCBANK",
        entry_price=Decimal("1623.00"),
        target_price=Decimal("1900.00"),
        stop_loss=Decimal("1548.00"),
        vix_value=Decimal("14.20"),
        capital=Decimal("50000.00"),
        tolerance="moderate",
        positions=[],
        display_name="Vijay",
    )

    print(f"  Verdict:  {report.verdict.value}")
    print(f"  Shares:   {report.position_size_shares}")
    print(f"  Position: ₹{report.position_size_rupees}")
    print(f"  Risk:     ₹{report.total_risk_rupees} ({report.risk_pct_of_capital}%)")
    print(f"  R/R:      {report.risk_reward_ratio}")
    print(f"  VIX:      {report.vix_value} (limit {report.vix_limit})")

    # Assertions
    assert report.verdict.value == "APPROVED", f"Expected APPROVED, got {report.verdict.value}"
    assert report.position_size_shares == 13, f"Expected 13 shares, got {report.position_size_shares}"
    assert report.position_size_rupees == Decimal("21099.00"), f"Expected ₹21099.00, got {report.position_size_rupees}"
    assert report.total_risk_rupees == Decimal("975.00"), f"Expected ₹975.00, got {report.total_risk_rupees}"
    assert report.risk_reward_ratio == "1:3.69", f"Expected 1:3.69, got {report.risk_reward_ratio}"
    assert report.risk_pct_of_capital == Decimal("1.95"), f"Expected 1.95%, got {report.risk_pct_of_capital}"
    assert report.advisor_note is not None, "advisor_note must not be None"
    assert "Vijay" in report.advisor_note, "advisor_note must mention user name"
    assert report.cot_reasoning is not None, "cot_reasoning must not be None"
    assert len(report.checks_passed) >= 5, f"Expected ≥5 checks passed, got {len(report.checks_passed)}"
    assert len(report.checks_failed) == 0, f"Expected 0 checks failed, got {len(report.checks_failed)}"

    print(f"\n  Advisor note: {report.advisor_note[:120]}...")
    print(f"  Checks passed: {report.checks_passed}")
    print("\n  ✅ TEST 1 PASSED — APPROVED trade matches few-shot 1")


# ──────────────────────────────────────────────
# Test 2 — REJECTED: Bad Risk/Reward
# ──────────────────────────────────────────────
def test_rejected_bad_rr():
    """Few-shot 2: HDFCBANK, entry=1623, target=1680, stop=1548.

    Expected:
      verdict    = REJECTED
      reason     = risk_reward_below_minimum
      R/R        = 1:0.76
      suggested  = ₹1773 (for 1:2.0 ratio)
    """
    print("\n" + "=" * 50)
    print("TEST 2 — REJECTED: Bad R/R (target too close)")
    print("=" * 50)

    from module3_risk_engine.engine import risk_engine

    report = risk_engine.calculate_risk(
        ticker="HDFCBANK",
        entry_price=Decimal("1623.00"),
        target_price=Decimal("1680.00"),
        stop_loss=Decimal("1548.00"),
        vix_value=Decimal("14.20"),
        capital=Decimal("50000.00"),
        tolerance="moderate",
        positions=[],
        display_name="Vijay",
    )

    print(f"  Verdict:  {report.verdict.value}")
    print(f"  Reason:   {report.rejection_reason}")
    print(f"  R/R:      {report.risk_reward_ratio}")
    print(f"  Suggested target: ₹{report.suggested_target}")
    print(f"  Minimum required: {report.minimum_required}")

    # Assertions
    assert report.verdict.value == "REJECTED", f"Expected REJECTED, got {report.verdict.value}"
    assert report.rejection_reason == "risk_reward_below_minimum", f"Expected risk_reward_below_minimum, got {report.rejection_reason}"
    assert report.risk_reward_ratio == "1:0.76", f"Expected 1:0.76, got {report.risk_reward_ratio}"
    assert report.suggested_target == Decimal("1773.000"), f"Expected ₹1773.000, got {report.suggested_target}"
    assert report.minimum_required == "1:2.0", f"Expected 1:2.0, got {report.minimum_required}"
    assert report.advisor_note is not None, "advisor_note must not be None"
    assert "Vijay" in report.advisor_note, "advisor_note must mention user name"
    assert report.position_size_shares == 0, f"REJECTED should have 0 shares, got {report.position_size_shares}"

    print(f"\n  Advisor note: {report.advisor_note[:120]}...")
    print("\n  ✅ TEST 2 PASSED — REJECTED for bad R/R matches few-shot 2")


# ──────────────────────────────────────────────
# Test 3 — REJECTED: VIX Gate Failed
# ──────────────────────────────────────────────
def test_rejected_vix_gate():
    """Few-shot 3: Any stock with VIX=27.5, moderate tolerance.

    Expected:
      verdict    = REJECTED
      reason     = vix_gate_failed
      VIX limit  = 20 (moderate)
      VIX signal = high_fear
    """
    print("\n" + "=" * 50)
    print("TEST 3 — REJECTED: VIX Gate (VIX=27.5)")
    print("=" * 50)

    from module3_risk_engine.engine import risk_engine

    report = risk_engine.calculate_risk(
        ticker="HDFCBANK",
        entry_price=Decimal("1623.00"),
        target_price=Decimal("1900.00"),
        stop_loss=Decimal("1548.00"),
        vix_value=Decimal("27.50"),
        capital=Decimal("50000.00"),
        tolerance="moderate",
        positions=[],
        display_name="Vijay",
    )

    print(f"  Verdict:  {report.verdict.value}")
    print(f"  Reason:   {report.rejection_reason}")
    print(f"  VIX:      {report.vix_value} (limit {report.vix_limit})")
    print(f"  Signal:   {report.vix_signal}")

    # Assertions
    assert report.verdict.value == "REJECTED", f"Expected REJECTED, got {report.verdict.value}"
    assert report.rejection_reason == "vix_gate_failed", f"Expected vix_gate_failed, got {report.rejection_reason}"
    assert report.vix_value == Decimal("27.50"), f"Expected VIX 27.50, got {report.vix_value}"
    assert report.vix_limit == Decimal("20"), f"Expected VIX limit 20, got {report.vix_limit}"
    assert report.vix_signal == "high_fear", f"Expected high_fear, got {report.vix_signal}"
    assert report.advisor_note is not None, "advisor_note must not be None"
    assert report.position_size_shares == 0, f"REJECTED should have 0 shares, got {report.position_size_shares}"

    # Also test VIX gate standalone
    vix_status = risk_engine.get_vix_gate_status(
        vix_value=Decimal("27.50"),
        tolerance="moderate",
    )
    assert vix_status.gate.value == "closed", f"Expected gate closed, got {vix_status.gate.value}"
    print(f"  Standalone gate: {vix_status.gate.value}")

    print(f"\n  Advisor note: {report.advisor_note[:120]}...")
    print("\n  ✅ TEST 3 PASSED — REJECTED for VIX gate matches few-shot 3")


# ──────────────────────────────────────────────
# Test 4 — REDUCE_SIZE
# ──────────────────────────────────────────────
def test_reduce_size():
    """Few-shot 4: HDFCBANK, same setup but requested=30 shares.

    Expected:
      verdict    = REDUCE_SIZE
      requested  = 30
      approved   = 13
      R/R        = 1:3.69 (same trade, safer size)
    """
    print("\n" + "=" * 50)
    print("TEST 4 — REDUCE_SIZE (requested 30, approved 13)")
    print("=" * 50)

    from module3_risk_engine.engine import risk_engine

    report = risk_engine.calculate_risk(
        ticker="HDFCBANK",
        entry_price=Decimal("1623.00"),
        target_price=Decimal("1900.00"),
        stop_loss=Decimal("1548.00"),
        vix_value=Decimal("14.20"),
        requested_shares=30,
        capital=Decimal("50000.00"),
        tolerance="moderate",
        positions=[],
        display_name="Vijay",
    )

    print(f"  Verdict:    {report.verdict.value}")
    print(f"  Requested:  {report.requested_shares}")
    print(f"  Approved:   {report.approved_shares}")
    print(f"  Shares:     {report.position_size_shares}")
    print(f"  Risk (req): ₹{report.requested_risk_rupees} ({report.risk_pct_at_requested}%)")
    print(f"  Risk (appr):₹{report.approved_risk_rupees} ({report.risk_pct_at_approved}%)")
    print(f"  R/R:        {report.risk_reward_ratio}")

    # Assertions
    assert report.verdict.value == "REDUCE_SIZE", f"Expected REDUCE_SIZE, got {report.verdict.value}"
    assert report.requested_shares == 30, f"Expected requested=30, got {report.requested_shares}"
    assert report.approved_shares == 13, f"Expected approved=13, got {report.approved_shares}"
    assert report.position_size_shares == 13, f"Expected 13 shares, got {report.position_size_shares}"
    assert report.risk_reward_ratio == "1:3.69", f"Expected 1:3.69, got {report.risk_reward_ratio}"
    assert report.requested_risk_rupees == Decimal("2250.00"), f"Expected ₹2250.00, got {report.requested_risk_rupees}"
    assert report.approved_risk_rupees == Decimal("975.00"), f"Expected ₹975.00, got {report.approved_risk_rupees}"
    assert report.advisor_note is not None, "advisor_note must not be None"
    assert "Vijay" in report.advisor_note, "advisor_note must mention user name"
    assert "30" in report.advisor_note, "advisor_note must mention requested 30"

    print(f"\n  Advisor note: {report.advisor_note[:120]}...")
    print("\n  ✅ TEST 4 PASSED — REDUCE_SIZE matches few-shot 4")


# ──────────────────────────────────────────────
# Test 5 — REJECTED: Sector Overexposure
# ──────────────────────────────────────────────
def test_rejected_sector():
    """Few-shot 5: ICICIBANK when already 78% in Banking.

    Setup: 2 existing Banking positions eating 78% of capital.
      HDFCBANK: 13 shares @ ₹1623 = ₹21,099
      KOTAKBANK: 10 shares @ ₹1800 = ₹18,000
      Total Banking = ₹39,099 = 78.2% of ₹50K
    Adding ICICIBANK would push Banking well over 25%.

    Expected:
      verdict   = REJECTED
      reason    = sector_overexposure
      sector    = Banking
    """
    print("\n" + "=" * 50)
    print("TEST 5 — REJECTED: Sector Overexposure (Banking)")
    print("=" * 50)

    from module3_risk_engine.engine import risk_engine
    from module3_risk_engine.models import OpenPosition

    existing_positions = [
        OpenPosition(
            ticker="HDFCBANK",
            sector="Banking",
            entry_price=Decimal("1623.00"),
            quantity=13,
            stop_loss=Decimal("1548.00"),
            target=Decimal("1900.00"),
        ),
        OpenPosition(
            ticker="KOTAKBANK",
            sector="Banking",
            entry_price=Decimal("1800.00"),
            quantity=10,
            stop_loss=Decimal("1720.00"),
            target=Decimal("2050.00"),
        ),
    ]

    report = risk_engine.calculate_risk(
        ticker="ICICIBANK",
        entry_price=Decimal("1050.00"),
        target_price=Decimal("1250.00"),
        stop_loss=Decimal("990.00"),
        vix_value=Decimal("14.20"),
        capital=Decimal("50000.00"),
        tolerance="moderate",
        positions=existing_positions,
        display_name="Vijay",
    )

    print(f"  Verdict:    {report.verdict.value}")
    print(f"  Reason:     {report.rejection_reason}")
    print(f"  Sector:     {report.sector}")
    print(f"  Current:    {report.current_exposure_pct}%")
    print(f"  Limit:      {report.max_exposure_pct}%")
    if report.suggested_alternatives:
        print(f"  Alternatives: {report.suggested_alternatives}")

    # Assertions
    assert report.verdict.value == "REJECTED", f"Expected REJECTED, got {report.verdict.value}"
    assert report.rejection_reason == "sector_overexposure", f"Expected sector_overexposure, got {report.rejection_reason}"
    assert report.sector == "Banking", f"Expected Banking, got {report.sector}"
    assert report.current_exposure_pct > Decimal("25"), f"Expected current > 25%, got {report.current_exposure_pct}%"
    assert report.max_exposure_pct == Decimal("50.00"), f"Expected 50.00%, got {report.max_exposure_pct}"
    assert report.advisor_note is not None, "advisor_note must not be None"
    assert "Banking" in report.advisor_note, "advisor_note must mention Banking"

    print(f"\n  Advisor note: {report.advisor_note[:120]}...")
    print("\n  ✅ TEST 5 PASSED — REJECTED for sector overexposure matches few-shot 5")


# ──────────────────────────────────────────────
# Test 6 — Full Pipeline via Engine
# ──────────────────────────────────────────────
def test_full_pipeline():
    """End-to-end test using risk_engine public API.

    Tests all 4 engine methods:
      1. calculate_risk()       — full validation
      2. get_position_size()    — quick sizing
      3. check_portfolio_risk() — portfolio health
      4. get_vix_gate_status()  — VIX gate
    """
    print("\n" + "=" * 50)
    print("TEST 6 — Full Pipeline (all 4 engine methods)")
    print("=" * 50)

    from module3_risk_engine.engine import risk_engine
    from module3_risk_engine.models import OpenPosition

    # 6a: calculate_risk — should APPROVE
    print("\n  6a: calculate_risk (TCS)")
    report = risk_engine.calculate_risk(
        ticker="TCS",
        entry_price=Decimal("3800.00"),
        target_price=Decimal("4200.00"),
        stop_loss=Decimal("3650.00"),
        vix_value=Decimal("12.50"),
        capital=Decimal("100000.00"),
        tolerance="moderate",
        positions=[],
        display_name="Vijay",
    )
    print(f"      Verdict: {report.verdict.value}, Shares: {report.position_size_shares}")
    print(f"      R/R: {report.risk_reward_ratio}, Risk: ₹{report.total_risk_rupees}")
    assert report.verdict.value == "APPROVED", f"Expected APPROVED for TCS, got {report.verdict.value}"
    assert report.position_size_shares > 0, "Should have positive shares"
    print("      ✅ calculate_risk passed")

    # 6b: get_position_size — quick sizing
    print("\n  6b: get_position_size (RELIANCE)")
    result = risk_engine.get_position_size(
        entry_price=Decimal("2500.00"),
        stop_loss=Decimal("2400.00"),
        capital=Decimal("100000.00"),
        tolerance="moderate",
    )
    print(f"      Shares: {result['shares']}, Risk: ₹{result['total_risk_rupees']}")
    assert result["shares"] > 0, "Should have positive shares"
    assert result["total_risk_rupees"] <= Decimal("2000.00"), "Risk should be ≤ 2% of 100K"
    print("      ✅ get_position_size passed")

    # 6c: check_portfolio_risk — health check
    print("\n  6c: check_portfolio_risk (with 1 position)")
    positions = [
        OpenPosition(
            ticker="TCS",
            sector="IT",
            entry_price=Decimal("3800.00"),
            quantity=5,
            stop_loss=Decimal("3650.00"),
            target=Decimal("4200.00"),
        ),
    ]
    result = risk_engine.check_portfolio_risk(
        capital=Decimal("100000.00"),
        positions=positions,
        tolerance="moderate",
        display_name="Vijay",
    )
    print(f"      Health: {result['health_grade']}")
    print(f"      Can add: {result['can_add_trade']} ({result['trades_remaining']} remaining)")
    print(f"      Risk: ₹{result['portfolio_report'].total_risk_rupees} ({result['portfolio_report'].total_risk_pct}%)")
    assert result["can_add_trade"] is True, "Should be able to add trades"
    assert result["health_grade"] in ("EXCELLENT", "GOOD"), f"Expected good health, got {result['health_grade']}"
    print("      ✅ check_portfolio_risk passed")

    # 6d: get_vix_gate_status — gate check
    print("\n  6d: get_vix_gate_status")
    # Open gate
    status = risk_engine.get_vix_gate_status(
        vix_value=Decimal("12.50"),
        tolerance="moderate",
    )
    print(f"      VIX=12.50: gate={status.gate.value}, signal={status.vix_signal}")
    assert status.gate.value == "open", f"Expected open, got {status.gate.value}"
    assert status.vix_signal == "low_fear", f"Expected low_fear, got {status.vix_signal}"

    # Closed gate
    status = risk_engine.get_vix_gate_status(
        vix_value=Decimal("22.00"),
        tolerance="moderate",
    )
    print(f"      VIX=22.00: gate={status.gate.value}, signal={status.vix_signal}")
    assert status.gate.value == "closed", f"Expected closed, got {status.gate.value}"
    assert status.vix_signal == "high_fear", f"Expected high_fear, got {status.vix_signal}"

    # Conservative (tighter gate)
    status = risk_engine.get_vix_gate_status(
        vix_value=Decimal("16.00"),
        tolerance="conservative",
    )
    print(f"      VIX=16.00 (conservative): gate={status.gate.value}")
    assert status.gate.value == "closed", f"Expected closed for conservative at 16, got {status.gate.value}"

    # Aggressive (looser gate)
    status = risk_engine.get_vix_gate_status(
        vix_value=Decimal("23.00"),
        tolerance="aggressive",
    )
    print(f"      VIX=23.00 (aggressive): gate={status.gate.value}")
    assert status.gate.value == "open", f"Expected open for aggressive at 23, got {status.gate.value}"

    print("      ✅ get_vix_gate_status passed")

    print("\n  ✅ TEST 6 PASSED — All 4 engine methods work correctly")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────
def main():
    tests = [
        ("Test 1: APPROVED Trade", test_approved_trade),
        ("Test 2: REJECTED Bad R/R", test_rejected_bad_rr),
        ("Test 3: REJECTED VIX Gate", test_rejected_vix_gate),
        ("Test 4: REDUCE_SIZE", test_reduce_size),
        ("Test 5: REJECTED Sector", test_rejected_sector),
        ("Test 6: Full Pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"\n  ❌ {name} FAILED: {e}")
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 50)
    print("MODULE 3 TEST SUMMARY")
    print("=" * 50)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if errors:
        print("\n  Failed tests:")
        for name, err in errors:
            print(f"    ❌ {name}: {err}")

    if failed == 0:
        print("\n  🎉 ALL MODULE 3 TESTS PASSED!")
        print("  Risk engine is operational. No trade passes without approval.")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed. Fix before proceeding to M4.")
        sys.exit(1)


if __name__ == "__main__":
    main()
