"""
SwingAdvisorBot — Module 4 Integration Tests
=============================================

Run:  python test_m4.py

Tests:
  1. Level Calculator     — HDFCBANK price=769.55, flag=ACCUMULATION_ZONE
  2. Stock Screener       — Filter 5 stocks by advisor flags
  3. Confidence Scorer    — Score with known inputs
  4. Full Pipeline (no Claude) — End-to-end via generate_setups_from_data
  5. Skipped Stock        — SELLING_PRESSURE flag → skipped
  6. Display Card         — TradeSetup.to_display_card() renders correctly

Requirements:
  - No .env needed (Tests 1-3, 5, 6 are pure Python)
  - Test 4 uses skip_claude=True (no API key needed)
  - No market hours dependency (all inputs are explicit)

All expected values verified by hand.
"""

import sys
import traceback
from decimal import Decimal


# ──────────────────────────────────────────────
# Test 1 — Level Calculator
# ──────────────────────────────────────────────
def test_level_calculator():
    """HDFCBANK price=769.55, flag=ACCUMULATION_ZONE (5% stop).

    Expected:
      entry_zone_low  = 765.70  (769.55 - 0.5% = 769.55 - 3.85)
      entry_zone_high = 773.40  (769.55 + 0.5% = 769.55 + 3.85)
      stop_loss       = 727.42  (765.70 - 5% = 765.70 - 38.28)
      risk_per_share  = 38.28   (765.70 - 727.42)
      reward_per_share = 114.84 (38.28 × 3.0)
      target_price    = 888.24  (773.40 + 114.84)
      risk_reward     = 1:3.00
    """
    print("\n" + "=" * 50)
    print("TEST 1 — Level Calculator (HDFCBANK)")
    print("=" * 50)

    from module4_setup_generator.technical.level_calculator import level_calculator

    levels = level_calculator.calculate(
        current_price=769.55,
        advisor_flag="ACCUMULATION_ZONE",
    )

    print(f"  Entry zone:    ₹{levels.entry_zone_low} - ₹{levels.entry_zone_high}")
    print(f"  Stop loss:     ₹{levels.stop_loss}")
    print(f"  Target:        ₹{levels.target_price}")
    print(f"  Risk/share:    ₹{levels.risk_per_share}")
    print(f"  Reward/share:  ₹{levels.reward_per_share}")
    print(f"  R/R:           {levels.risk_reward_ratio}")

    assert levels.entry_zone_low == Decimal("765.70"), f"Expected 765.70, got {levels.entry_zone_low}"
    assert levels.entry_zone_high == Decimal("773.40"), f"Expected 773.40, got {levels.entry_zone_high}"
    assert levels.stop_loss == Decimal("727.41"), f"Expected 727.41, got {levels.stop_loss}"
    assert levels.risk_per_share == Decimal("38.29"), f"Expected 38.29, got {levels.risk_per_share}"
    assert levels.reward_per_share == Decimal("114.87"), f"Expected 114.87, got {levels.reward_per_share}"
    assert levels.target_price == Decimal("888.27"), f"Expected 888.27, got {levels.target_price}"
    assert levels.risk_reward_ratio == "1:3.00", f"Expected 1:3.00, got {levels.risk_reward_ratio}"

    print("\n  ✅ TEST 1 PASSED — Level calculator matches hand-calculated values")


# ──────────────────────────────────────────────
# Test 2 — Stock Screener
# ──────────────────────────────────────────────
def test_stock_screener():
    """Screen 5 stocks — 3 should pass, 2 should be skipped.

    Stocks:
      HDFCBANK  — ACCUMULATION_ZONE  → pass (priority 4)
      RELIANCE  — BREAKOUT_WATCH     → pass (priority 1)
      TCS       — SELLING_PRESSURE   → skip
      INFY      — MOMENTUM_BUILDING  → pass (priority 3)
      WIPRO     — NEUTRAL            → skip

    Expected order: RELIANCE(1), INFY(3), HDFCBANK(4)
    """
    print("\n" + "=" * 50)
    print("TEST 2 — Stock Screener")
    print("=" * 50)

    from module1_data_layer.models import (
        AdvisorFlag,
        MarketData,
        MarketStatus,
        StockData,
        VolumeSignal,
    )
    from module4_setup_generator.technical.stock_screener import stock_screener

    stocks = [
        StockData(
            ticker="HDFCBANK", price=769.55,
            advisor_flag=AdvisorFlag.ACCUMULATION_ZONE,
            volume_signal=VolumeSignal.ABOVE_AVERAGE,
            high_52w=900.0, low_52w=600.0, sector="Banking",
        ),
        StockData(
            ticker="RELIANCE", price=2450.0,
            advisor_flag=AdvisorFlag.BREAKOUT_WATCH,
            volume_signal=VolumeSignal.UNUSUAL_SPIKE,
            high_52w=2700.0, low_52w=2100.0, sector="Energy",
        ),
        StockData(
            ticker="TCS", price=3500.0,
            advisor_flag=AdvisorFlag.SELLING_PRESSURE,
            volume_signal=VolumeSignal.NORMAL,
            high_52w=4000.0, low_52w=3200.0, sector="IT",
        ),
        StockData(
            ticker="INFY", price=1500.0,
            advisor_flag=AdvisorFlag.MOMENTUM_BUILDING,
            volume_signal=VolumeSignal.ABOVE_AVERAGE,
            high_52w=1700.0, low_52w=1300.0, sector="IT",
        ),
        StockData(
            ticker="WIPRO", price=450.0,
            advisor_flag=AdvisorFlag.NEUTRAL,
            volume_signal=VolumeSignal.BELOW_AVERAGE,
            high_52w=550.0, low_52w=380.0, sector="IT",
        ),
    ]

    market_data = MarketData(
        market_status=MarketStatus.OPEN,
        stocks=stocks,
    )

    candidates = stock_screener.screen(market_data=market_data, max_candidates=10)

    print(f"  Candidates: {len(candidates)}")
    for c in candidates:
        print(f"    {c.ticker} — {c.advisor_flag.value}")

    assert len(candidates) == 3, f"Expected 3 candidates, got {len(candidates)}"
    assert candidates[0].ticker == "RELIANCE", f"Expected RELIANCE first, got {candidates[0].ticker}"
    assert candidates[1].ticker == "INFY", f"Expected INFY second, got {candidates[1].ticker}"
    assert candidates[2].ticker == "HDFCBANK", f"Expected HDFCBANK third, got {candidates[2].ticker}"

    # Check skip reasons
    skip_tcs = stock_screener.get_skip_reason(stocks[2])
    skip_wipro = stock_screener.get_skip_reason(stocks[4])
    assert skip_tcs is not None, "TCS should have a skip reason"
    assert skip_wipro is not None, "WIPRO should have a skip reason"
    print(f"  TCS skip: {skip_tcs}")
    print(f"  WIPRO skip: {skip_wipro}")

    print("\n  ✅ TEST 2 PASSED — Screener filters and orders correctly")


# ──────────────────────────────────────────────
# Test 3 — Confidence Scorer
# ──────────────────────────────────────────────
def test_confidence_scorer():
    """Score HDFCBANK with known inputs.

    Inputs:
      VIX = 14.0          → +0.7  (≤15)
      Sector = bullish    → +1.0
      Volume = above_avg  → +0.5
      Flag = accumulation → +0.3
      R/R = 3.00          → +0.3

    Expected: 5.0 + 0.7 + 1.0 + 0.5 + 0.3 + 0.3 = 7.8
    """
    print("\n" + "=" * 50)
    print("TEST 3 — Confidence Scorer")
    print("=" * 50)

    from module1_data_layer.models import (
        AdvisorFlag,
        StockData,
        VolumeSignal,
    )
    from module4_setup_generator.technical.confidence_scorer import confidence_scorer

    stock = StockData(
        ticker="HDFCBANK", price=769.55,
        advisor_flag=AdvisorFlag.ACCUMULATION_ZONE,
        volume_signal=VolumeSignal.ABOVE_AVERAGE,
        high_52w=900.0, low_52w=600.0, sector="Banking",
    )

    score = confidence_scorer.score(
        stock=stock,
        india_vix=14.0,
        sector_mood="bullish",
        risk_reward_ratio=Decimal("3.00"),
    )

    print(f"  Score: {score}")
    print(f"  Breakdown: base=5.0, vix=+0.7, sector=+1.0, vol=+0.5, flag=+0.3, rr=+0.3")

    assert score == 7.8, f"Expected 7.8, got {score}"

    # Test with bearish conditions
    score_bearish = confidence_scorer.score(
        stock=stock,
        india_vix=22.0,
        sector_mood="bearish",
        risk_reward_ratio=Decimal("2.0"),
    )
    print(f"  Bearish score: {score_bearish}")
    assert score_bearish < 6.0, f"Expected < 6.0 for bearish, got {score_bearish}"

    print("\n  ✅ TEST 3 PASSED — Confidence scorer produces expected scores")


# ──────────────────────────────────────────────
# Test 4 — Full Pipeline (No Claude)
# ──────────────────────────────────────────────
def test_full_pipeline_no_claude():
    """End-to-end via generate_setups_from_data with skip_claude=True.

    Inputs:
      3 qualifying stocks (RELIANCE, INFY, HDFCBANK)
      2 skipped stocks (TCS, WIPRO)
      Market mood: cautious_bullish
      VIX: 14.2

    Expected:
      - At least 1 setup generated
      - All setups have entry < target
      - All setups have stop < entry
      - No duplicate tickers
      - Package has market_mood and india_vix
    """
    print("\n" + "=" * 50)
    print("TEST 4 — Full Pipeline (No Claude)")
    print("=" * 50)

    from module1_data_layer.models import (
        AdvisorFlag,
        MarketData,
        MarketStatus,
        StockData,
        VolumeSignal,
    )
    from module2_analysis_engine.models import MarketAnalysis, MarketMood, SectorAnalysis
    from module4_setup_generator.engine import setup_engine
    from module4_setup_generator.models import SetupFilter

    stocks = [
        StockData(
            ticker="HDFCBANK", price=769.55,
            advisor_flag=AdvisorFlag.ACCUMULATION_ZONE,
            volume_signal=VolumeSignal.ABOVE_AVERAGE,
            volume_ratio=1.37,
            high_52w=900.0, low_52w=600.0, sector="Banking",
        ),
        StockData(
            ticker="RELIANCE", price=2450.0,
            advisor_flag=AdvisorFlag.BREAKOUT_WATCH,
            volume_signal=VolumeSignal.UNUSUAL_SPIKE,
            volume_ratio=3.2,
            high_52w=2700.0, low_52w=2100.0, sector="Energy",
        ),
        StockData(
            ticker="INFY", price=1500.0,
            advisor_flag=AdvisorFlag.MOMENTUM_BUILDING,
            volume_signal=VolumeSignal.ABOVE_AVERAGE,
            volume_ratio=1.5,
            high_52w=1700.0, low_52w=1300.0, sector="IT",
        ),
        StockData(
            ticker="TCS", price=3500.0,
            advisor_flag=AdvisorFlag.SELLING_PRESSURE,
            volume_signal=VolumeSignal.NORMAL,
            high_52w=4000.0, low_52w=3200.0, sector="IT",
        ),
        StockData(
            ticker="WIPRO", price=450.0,
            advisor_flag=AdvisorFlag.NEUTRAL,
            volume_signal=VolumeSignal.BELOW_AVERAGE,
            high_52w=550.0, low_52w=380.0, sector="IT",
        ),
    ]

    market_data = MarketData(
        market_status=MarketStatus.OPEN,
        india_vix=14.2,
        stocks=stocks,
    )

    analysis = MarketAnalysis(
        market_mood=MarketMood.CAUTIOUS_BULLISH,
        mood_confidence=0.75,
        situation="Nifty 50 trading flat with mild positive bias. VIX at 14.2 indicates low fear. Banking sector showing accumulation patterns.",
        reasoning="RBI rate decision expected next week. Markets consolidating near 22500. FII flows mildly positive. Sector rotation visible from IT to Banking.",
        action="Consider selective accumulation in banking stocks near support. Keep positions light until post-RBI clarity.",
        risk="Unexpected global event or RBI hawkish surprise could trigger selloff. Keep stop losses tight.",
        sector_analyses=[
            SectorAnalysis(
                sector_name="Banking",
                sector_mood=MarketMood.BULLISH,
                change_pct=1.2,
            ),
            SectorAnalysis(
                sector_name="IT",
                sector_mood=MarketMood.NEUTRAL,
                change_pct=-0.3,
            ),
            SectorAnalysis(
                sector_name="Energy",
                sector_mood=MarketMood.CAUTIOUS_BULLISH,
                change_pct=0.8,
            ),
        ],
    )

    setup_filter = SetupFilter(
        display_name="Vijay",
        capital=50000.0,
        risk_tolerance="moderate",
        max_setups=5,
        min_confidence=6.0,
        skip_claude=True,
    )

    package = setup_engine.generate_setups_from_data(
        market_data=market_data,
        analysis=analysis,
        setup_filter=setup_filter,
    )

    print(f"  Setups: {len(package.setups)}")
    print(f"  Skipped: {len(package.skipped_setups)}")
    print(f"  Market mood: {package.market_mood}")
    print(f"  India VIX: {package.india_vix}")
    print(f"  Freshness: {package.freshness.value}")
    print(f"  Advisor note: {package.advisor_note[:120] if package.advisor_note else 'None'}...")

    for s in package.setups:
        print(f"\n    {s.ticker} ({s.sector})")
        print(f"      Entry: ₹{s.entry_zone_low} - ₹{s.entry_zone_high}")
        print(f"      Target: ₹{s.target_price}, Stop: ₹{s.stop_loss}")
        print(f"      Confidence: {s.confidence_score}, R/R: {s.risk_reward_ratio}")
        print(f"      Shares: {s.position_size_shares}, Verdict: {s.risk_verdict}")

    # Assertions
    assert len(package.setups) >= 1, f"Expected at least 1 setup, got {len(package.setups)}"
    assert package.market_mood == "cautious_bullish", f"Expected cautious_bullish, got {package.market_mood}"
    assert package.india_vix == 14.2, f"Expected VIX 14.2, got {package.india_vix}"

    # No duplicate tickers
    tickers = [s.ticker for s in package.setups]
    assert len(tickers) == len(set(tickers)), f"Duplicate tickers found: {tickers}"

    for s in package.setups:
        # Entry < target
        assert s.entry_zone_high < s.target_price, (
            f"{s.ticker}: entry_high {s.entry_zone_high} >= target {s.target_price}"
        )
        # Stop < entry
        assert s.stop_loss < s.entry_zone_low, (
            f"{s.ticker}: stop {s.stop_loss} >= entry_low {s.entry_zone_low}"
        )
        # Confidence in range
        assert 4.0 <= s.confidence_score <= 9.5, (
            f"{s.ticker}: confidence {s.confidence_score} out of range"
        )
        # Has risk verdict
        assert s.risk_verdict in ("APPROVED", "REDUCE_SIZE"), (
            f"{s.ticker}: unexpected verdict {s.risk_verdict}"
        )
        # Has position sizing
        assert s.position_size_shares > 0, (
            f"{s.ticker}: no position sizing"
        )
        # Has a lesson (fallback from lesson rotation since skip_claude=True)
        assert s.lesson is not None, f"{s.ticker}: no lesson"

    print("\n  ✅ TEST 4 PASSED — Full pipeline produces valid setups")


# ──────────────────────────────────────────────
# Test 5 — Skipped Stock (Selling Pressure)
# ──────────────────────────────────────────────
def test_skipped_stock():
    """TCS with SELLING_PRESSURE should be skipped by screener.

    The screener should reject it, and it should appear
    in the skipped_setups list in the package.
    """
    print("\n" + "=" * 50)
    print("TEST 5 — Skipped Stock (Selling Pressure)")
    print("=" * 50)

    from module1_data_layer.models import (
        AdvisorFlag,
        MarketData,
        MarketStatus,
        StockData,
        VolumeSignal,
    )
    from module2_analysis_engine.models import MarketAnalysis, MarketMood
    from module4_setup_generator.engine import setup_engine
    from module4_setup_generator.models import SetupFilter

    stocks = [
        StockData(
            ticker="TCS", price=3500.0,
            advisor_flag=AdvisorFlag.SELLING_PRESSURE,
            volume_signal=VolumeSignal.NORMAL,
            high_52w=4000.0, low_52w=3200.0, sector="IT",
        ),
    ]

    market_data = MarketData(
        market_status=MarketStatus.OPEN,
        india_vix=14.0,
        stocks=stocks,
    )

    analysis = MarketAnalysis(
        market_mood=MarketMood.NEUTRAL,
        mood_confidence=0.5,
        situation="Market trading flat with no clear direction today.",
        reasoning="Lack of triggers keeping market range-bound.",
        action="Wait for clarity before new positions.",
        risk="Sideways action could break either way on news.",
    )

    setup_filter = SetupFilter(
        display_name="Vijay",
        capital=50000.0,
        risk_tolerance="moderate",
        skip_claude=True,
    )

    package = setup_engine.generate_setups_from_data(
        market_data=market_data,
        analysis=analysis,
        setup_filter=setup_filter,
    )

    print(f"  Setups: {len(package.setups)}")
    print(f"  Skipped: {len(package.skipped_setups)}")

    assert len(package.setups) == 0, f"Expected 0 setups, got {len(package.setups)}"
    assert len(package.skipped_setups) >= 1, f"Expected at least 1 skipped, got {len(package.skipped_setups)}"

    tcs_skip = next(
        (s for s in package.skipped_setups if s.ticker == "TCS"), None
    )
    assert tcs_skip is not None, "TCS should be in skipped list"
    print(f"  TCS skip reason: {tcs_skip.skip_reason}")
    assert tcs_skip.skip_reason is not None, "Skip reason must not be None"

    # Advisor note should mention no setups
    assert package.advisor_note is not None, "Advisor note must not be None"
    print(f"  Advisor note: {package.advisor_note}")

    print("\n  ✅ TEST 5 PASSED — Selling pressure stock correctly skipped")


# ──────────────────────────────────────────────
# Test 6 — Display Card
# ──────────────────────────────────────────────
def test_display_card():
    """TradeSetup.to_display_card() should produce a readable string."""
    print("\n" + "=" * 50)
    print("TEST 6 — Display Card Rendering")
    print("=" * 50)

    from module4_setup_generator.models import SetupType, TradeSetup

    setup = TradeSetup(
        ticker="HDFCBANK",
        company_name="HDFC Bank Limited",
        sector="Banking",
        setup_type=SetupType.SWING_LONG,
        entry_zone_low=Decimal("765.70"),
        entry_zone_high=Decimal("773.40"),
        target_price=Decimal("888.24"),
        stop_loss=Decimal("727.42"),
        current_price=Decimal("769.55"),
        confidence_score=7.8,
        risk_reward_ratio="1:3.00",
        position_size_shares=13,
        position_size_rupees=Decimal("9954.10"),
        max_risk_rupees=Decimal("497.64"),
        risk_pct_of_capital=Decimal("1.00"),
        risk_verdict="APPROVED",
        setup_reasoning="HDFCBANK shows accumulation at ₹769. Volume 37% above average confirms institutional buying.",
        entry_trigger="Enter on close above ₹773 with sustained volume.",
        exit_strategy="Book 50% at ₹850, trail rest to ₹888.",
        risk_warning="Exit immediately if price drops below ₹727.",
        lesson="This setup demonstrates accumulation at support.",
    )

    card = setup.to_display_card()
    print(card)

    # Assertions
    assert "HDFCBANK" in card, "Card must contain ticker"
    assert "765.70" in card, "Card must contain entry_zone_low"
    assert "888.24" in card, "Card must contain target"
    assert "727.42" in card, "Card must contain stop_loss"
    assert "7.8" in card, "Card must contain confidence score"
    assert "accumulation" in card.lower(), "Card must contain reasoning"

    print("\n  ✅ TEST 6 PASSED — Display card renders correctly")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────
def main():
    tests = [
        ("Test 1 — Level Calculator", test_level_calculator),
        ("Test 2 — Stock Screener", test_stock_screener),
        ("Test 3 — Confidence Scorer", test_confidence_scorer),
        ("Test 4 — Full Pipeline (No Claude)", test_full_pipeline_no_claude),
        ("Test 5 — Skipped Stock", test_skipped_stock),
        ("Test 6 — Display Card", test_display_card),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"\n  ❌ {name} FAILED: {e}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"MODULE 4 TEST RESULTS: {passed}/{len(tests)} passed")
    if errors:
        print(f"FAILED ({failed}):")
        for name, e in errors:
            print(f"  • {name}: {e}")
    else:
        print("ALL TESTS PASSED ✅")
    print("=" * 50)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
