"""
SwingAdvisorBot — Module 5 Integration Tests
=============================================

Run:  python test_m5.py

Tests:
  1. SQLite Schema         — Create tables, verify they exist
  2. User Profile CRUD     — Upsert, get, update capital
  3. Trade Lifecycle        — Insert trade, close trade, P&L
  4. Learning Progress      — Upsert, retrieve by concept
  5. Chunker               — Trade and conversation chunking
  6. Memory Context Budget  — Profile-only context ≤300 tokens
  7. M3 Compatibility       — get_user_context, get_open_positions
  8. Watchlist CRUD         — Add, list, remove
  9. Engine Integration     — Full pipeline via memory_engine

Requirements:
  - No .env needed (all tests use in-memory SQLite, no Pinecone, no Claude)
  - No market hours dependency
  - Tests 1-9 are all offline — no API keys needed
"""

import os
import sys
import tempfile
import traceback
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


# ──────────────────────────────────────────────
# Test 1 — SQLite Schema
# ──────────────────────────────────────────────
def test_sqlite_schema():
    """Create all tables in a temp DB and verify they exist."""
    print("\n" + "=" * 50)
    print("TEST 1 — SQLite Schema")
    print("=" * 50)

    import sqlite3

    from module5_memory.database.schema import (
        initialize_database,
        get_connection,
        drop_all_tables,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        # Create tables
        initialize_database(db_path)
        print("  ✓ initialize_database() completed")

        # Verify tables exist
        conn = get_connection(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        conn.close()

        expected = ["daily_stats", "learning_progress", "trades", "user_profiles", "watchlist"]
        assert tables == expected, f"Expected {expected}, got {tables}"
        print(f"  ✓ All 5 tables created: {tables}")

        # Idempotent — run again without error
        initialize_database(db_path)
        print("  ✓ Idempotent — second call succeeded")

        # Drop all
        drop_all_tables(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        remaining = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert len(remaining) == 0, f"Expected 0 tables after drop, got {remaining}"
        print("  ✓ drop_all_tables() cleared everything")

    print("  ✅ TEST 1 PASSED")


# ──────────────────────────────────────────────
# Test 2 — User Profile CRUD
# ──────────────────────────────────────────────
def test_user_profile_crud():
    """Upsert, get, update capital on a user profile."""
    print("\n" + "=" * 50)
    print("TEST 2 — User Profile CRUD")
    print("=" * 50)

    from module5_memory.database.sqlite_manager import SQLiteManager
    from module5_memory.models import UserProfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        mgr = SQLiteManager(db_path)

        # Create default profile
        profile = UserProfile(
            user_id="XCU700",
            name="Vijay",
            capital=Decimal("50000.00"),
            risk_tolerance="moderate",
        )
        mgr.upsert_user_profile(profile)
        print("  ✓ Profile inserted")

        # Retrieve
        loaded = mgr.get_user_profile("XCU700")
        assert loaded is not None, "Profile not found"
        assert loaded.name == "Vijay"
        assert loaded.capital == Decimal("50000.00")
        assert loaded.risk_tolerance == "moderate"
        print(f"  ✓ Retrieved: {loaded.name}, ₹{loaded.capital}, {loaded.risk_tolerance}")

        # Update capital
        mgr.update_user_capital("XCU700", Decimal("75000.00"))
        updated = mgr.get_user_profile("XCU700")
        assert updated.capital == Decimal("75000.00"), f"Expected 75000, got {updated.capital}"
        print(f"  ✓ Capital updated to ₹{updated.capital}")

        # Win rate (computed field)
        profile.total_trades = 10
        profile.winning_trades = 6
        profile.total_pnl = Decimal("5000.00")
        mgr.upsert_user_profile(profile)
        loaded2 = mgr.get_user_profile("XCU700")
        assert loaded2.win_rate == 60.0, f"Expected 60.0%, got {loaded2.win_rate}%"
        print(f"  ✓ Win rate computed: {loaded2.win_rate}%")

        # Profile not found
        missing = mgr.get_user_profile("NONEXISTENT")
        assert missing is None, "Expected None for missing profile"
        print("  ✓ Missing profile returns None")

        # Context summary
        summary = loaded2.to_context_summary()
        assert "Vijay" in summary
        assert "moderate" in summary
        print(f"  ✓ Context summary: {summary[:80]}...")

    print("  ✅ TEST 2 PASSED")


# ──────────────────────────────────────────────
# Test 3 — Trade Lifecycle
# ──────────────────────────────────────────────
def test_trade_lifecycle():
    """Insert a trade, close it, verify P&L calculation."""
    print("\n" + "=" * 50)
    print("TEST 3 — Trade Lifecycle")
    print("=" * 50)

    from module5_memory.database.sqlite_manager import SQLiteManager
    from module5_memory.models import (
        ExitReason,
        TradeRecord,
        TradeStatus,
        UserProfile,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        mgr = SQLiteManager(db_path)

        # Need a user first (FK)
        mgr.upsert_user_profile(UserProfile(user_id="XCU700"))

        # Insert open trade
        trade = TradeRecord(
            user_id="XCU700",
            ticker="HDFCBANK",
            sector="Banking",
            entry_price=Decimal("769.55"),
            stop_loss=Decimal("727.42"),
            target_price=Decimal("888.24"),
            shares=13,
            market_mood="cautiously_bullish",
            vix_at_entry=Decimal("14.2"),
        )
        trade_id = mgr.insert_trade(trade)
        print(f"  ✓ Trade inserted: {trade_id}")

        # Verify it's open
        loaded = mgr.get_trade(trade_id)
        assert loaded is not None
        assert loaded.status == TradeStatus.OPEN
        assert loaded.ticker == "HDFCBANK"
        assert loaded.shares == 13
        print(f"  ✓ Trade loaded: {loaded.ticker}, {loaded.shares} shares, status={loaded.status.value}")

        # Close trade — target hit
        closed = mgr.close_trade(
            trade_id=trade_id,
            exit_price=Decimal("888.24"),
            exit_reason=ExitReason.TARGET_HIT,
        )
        assert closed is not None
        assert closed.status == TradeStatus.CLOSED
        assert closed.exit_price == Decimal("888.24")
        assert closed.exit_reason == ExitReason.TARGET_HIT

        # P&L: (888.24 - 769.55) × 13 = 118.69 × 13 = 1542.97
        expected_pnl = (Decimal("888.24") - Decimal("769.55")) * 13
        assert closed.pnl_rupees == expected_pnl, f"Expected {expected_pnl}, got {closed.pnl_rupees}"
        print(f"  ✓ Trade closed: P&L = ₹{closed.pnl_rupees}")

        # Get open trades (should be empty now)
        open_trades = mgr.get_open_trades("XCU700")
        assert len(open_trades) == 0, f"Expected 0 open trades, got {len(open_trades)}"
        print("  ✓ Open trades: 0 (all closed)")

        # Get by ticker
        ticker_trades = mgr.get_trades_by_ticker("XCU700", "HDFCBANK")
        assert len(ticker_trades) == 1
        print(f"  ✓ Trades by ticker HDFCBANK: {len(ticker_trades)}")

        # Sector exposure (should be empty — trade is closed)
        exposure = mgr.get_sector_exposure("XCU700")
        assert len(exposure) == 0
        print("  ✓ Sector exposure: empty (no open trades)")

        # Embedding text
        emb_text = closed.to_embedding_text()
        assert "HDFCBANK" in emb_text
        assert "769.55" in emb_text
        print(f"  ✓ Embedding text: {emb_text[:80]}...")

    print("  ✅ TEST 3 PASSED")


# ──────────────────────────────────────────────
# Test 4 — Learning Progress
# ──────────────────────────────────────────────
def test_learning_progress():
    """Upsert learning progress, retrieve by concept."""
    print("\n" + "=" * 50)
    print("TEST 4 — Learning Progress")
    print("=" * 50)

    from module5_memory.database.sqlite_manager import SQLiteManager
    from module5_memory.models import LearningProgress, UserProfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        mgr = SQLiteManager(db_path)

        mgr.upsert_user_profile(UserProfile(user_id="XCU700"))

        # Insert learning record
        progress = LearningProgress(
            user_id="XCU700",
            concept="stop_loss",
            quiz_score=85,
            times_taught=2,
        )
        mgr.upsert_learning(progress)
        print("  ✓ Learning record inserted: stop_loss")

        # Retrieve by concept
        loaded = mgr.get_learning_by_concept("XCU700", "stop_loss")
        assert loaded is not None
        assert loaded.concept == "stop_loss"
        assert loaded.quiz_score == 85
        assert loaded.times_taught == 2
        print(f"  ✓ Retrieved: concept={loaded.concept}, score={loaded.quiz_score}, taught={loaded.times_taught}x")

        # Add another concept
        progress2 = LearningProgress(
            user_id="XCU700",
            concept="position_sizing",
            quiz_score=70,
        )
        mgr.upsert_learning(progress2)

        # Get all
        all_learning = mgr.get_all_learning("XCU700")
        assert len(all_learning) == 2
        concepts = {l.concept for l in all_learning}
        assert concepts == {"stop_loss", "position_sizing"}
        print(f"  ✓ All learning: {concepts}")

        # Missing concept
        missing = mgr.get_learning_by_concept("XCU700", "nonexistent")
        assert missing is None
        print("  ✓ Missing concept returns None")

    print("  ✅ TEST 4 PASSED")


# ──────────────────────────────────────────────
# Test 5 — Chunker
# ──────────────────────────────────────────────
def test_chunker():
    """Test trade and conversation chunking."""
    print("\n" + "=" * 50)
    print("TEST 5 — Chunker")
    print("=" * 50)

    from module5_memory.embeddings.chunker import (
        Chunk,
        chunk_trade,
        chunk_learning,
        chunk_conversation,
        chunk_market_pattern,
        chunk_knowledge,
    )
    from module5_memory.models import LearningProgress, TradeRecord

    # Trade chunk
    trade = TradeRecord(
        ticker="HDFCBANK",
        sector="Banking",
        entry_price=Decimal("769.55"),
        stop_loss=Decimal("727.42"),
        target_price=Decimal("888.24"),
        shares=13,
    )
    chunk = chunk_trade(trade)
    assert chunk.namespace == "trade_memory"
    assert "HDFCBANK" in chunk.text
    assert chunk.token_estimate > 0
    assert "trade_id" in chunk.metadata
    print(f"  ✓ Trade chunk: ns={chunk.namespace}, tokens≈{chunk.token_estimate}")

    # Learning chunk
    progress = LearningProgress(
        concept="stop_loss",
        quiz_score=85,
        times_taught=3,
    )
    lchunk = chunk_learning(progress)
    assert lchunk.namespace == "lessons"
    assert "stop_loss" in lchunk.text
    assert "85%" in lchunk.text
    print(f"  ✓ Learning chunk: ns={lchunk.namespace}, text='{lchunk.text[:60]}...'")

    # Conversation chunk (sliding window)
    long_text = "This is a test conversation. " * 100  # ~2900 chars
    conv_chunks = chunk_conversation(long_text, "conv_001")
    assert len(conv_chunks) >= 2, f"Expected ≥2 chunks, got {len(conv_chunks)}"
    assert all(c.namespace == "conversations" for c in conv_chunks)
    print(f"  ✓ Conversation chunks: {len(conv_chunks)} chunks from {len(long_text)} chars")

    # Market pattern chunk
    pattern = chunk_market_pattern(
        "NIFTY forming higher lows with declining VIX",
        "pat_001",
        ticker="NIFTY",
        date="2026-05-15",
    )
    assert pattern.namespace == "market_patterns"
    assert pattern.metadata["ticker"] == "NIFTY"
    print(f"  ✓ Market pattern chunk: ns={pattern.namespace}")

    # Knowledge base chunks
    kb_text = "Stop loss is essential.\n\nPosition sizing protects capital.\n\nRisk-reward ratio matters."
    kb_chunks = chunk_knowledge(kb_text, "manual", topic="risk_management")
    assert len(kb_chunks) == 3
    assert all(c.namespace == "knowledge_base" for c in kb_chunks)
    print(f"  ✓ Knowledge chunks: {len(kb_chunks)} paragraphs")

    # Token estimation
    assert Chunk.estimate_tokens("hello world") >= 1
    print(f"  ✓ Token estimate('hello world') = {Chunk.estimate_tokens('hello world')}")

    print("  ✅ TEST 5 PASSED")


# ──────────────────────────────────────────────
# Test 6 — Memory Context Budget
# ──────────────────────────────────────────────
def test_memory_context_budget():
    """Build profile-only context and verify it's within 300 token budget."""
    print("\n" + "=" * 50)
    print("TEST 6 — Memory Context Budget")
    print("=" * 50)

    from module5_memory.database.sqlite_manager import SQLiteManager
    from module5_memory.models import UserProfile
    from module5_memory.retrieval.context_builder import ContextBuilder
    from module5_memory.retrieval.rag_retriever import RAGRetriever

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        mgr = SQLiteManager(db_path)

        # Create profile
        profile = UserProfile(
            user_id="XCU700",
            name="Vijay",
            capital=Decimal("50000.00"),
            risk_tolerance="moderate",
            total_trades=15,
            winning_trades=9,
            total_pnl=Decimal("3200.00"),
            open_positions_count=2,
        )
        mgr.upsert_user_profile(profile)

        # Build context (no Pinecone — profile only)
        retriever = RAGRetriever()
        builder = ContextBuilder(mgr, retriever)
        ctx = builder.build_context(user_id="XCU700", query="")

        assert not ctx.is_empty, "Context should not be empty"
        assert ctx.within_budget, f"Context exceeds 300 tokens: {ctx.token_estimate}"
        assert ctx.token_estimate <= 300, f"Token estimate {ctx.token_estimate} > 300"
        assert "Vijay" in ctx.text
        assert "50,000" in ctx.text
        assert "moderate" in ctx.text
        print(f"  ✓ Context text: {ctx.text[:100]}...")
        print(f"  ✓ Token estimate: {ctx.token_estimate} (budget: 300)")
        print(f"  ✓ Within budget: {ctx.within_budget}")
        print(f"  ✓ Chunks used: {ctx.chunks_used}")

        # Profile-only shortcut
        ctx2 = builder.build_context_profile_only("XCU700")
        assert ctx2.within_budget
        print(f"  ✓ Profile-only context: {ctx2.token_estimate} tokens")

        # Missing user → empty context
        ctx_empty = builder.build_context(user_id="NOBODY")
        assert ctx_empty.is_empty
        assert ctx_empty.token_estimate == 0
        print("  ✓ Missing user → empty context")

    print("  ✅ TEST 6 PASSED")


# ──────────────────────────────────────────────
# Test 7 — M3 Compatibility
# ──────────────────────────────────────────────
def test_m3_compatibility():
    """Verify MemoryProvider returns M3-compatible data."""
    print("\n" + "=" * 50)
    print("TEST 7 — M3 Compatibility")
    print("=" * 50)

    from module5_memory.memory_provider import MemoryProvider
    from module5_memory.models import TradeRecord, TradeStatus, UserProfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        provider = MemoryProvider(db_path)

        # Create profile
        profile = UserProfile(
            user_id="XCU700",
            name="Vijay",
            capital=Decimal("50000.00"),
            risk_tolerance="moderate",
        )
        provider.update_profile(profile)

        # Add an open trade
        trade = TradeRecord(
            user_id="XCU700",
            ticker="HDFCBANK",
            sector="Banking",
            entry_price=Decimal("1623.00"),
            stop_loss=Decimal("1548.00"),
            target_price=Decimal("1900.00"),
            shares=13,
        )
        provider.save_trade(trade)

        # get_user_context — M3 compatible
        ctx = provider.get_user_context("XCU700")
        assert ctx["user_id"] == "XCU700"
        assert ctx["name"] == "Vijay"
        assert ctx["capital"] == Decimal("50000.00")
        assert len(ctx["open_positions"]) == 1
        print(f"  ✓ get_user_context: {ctx['name']}, ₹{ctx['capital']}, {len(ctx['open_positions'])} positions")

        # Check M3 OpenPosition type
        pos = ctx["open_positions"][0]
        from module3_risk_engine.models import OpenPosition as M3OpenPosition
        assert isinstance(pos, M3OpenPosition), f"Expected M3 OpenPosition, got {type(pos)}"
        assert pos.ticker == "HDFCBANK"
        assert pos.quantity == 13
        assert pos.entry_price == Decimal("1623.00")
        print(f"  ✓ OpenPosition: {pos.ticker}, qty={pos.quantity}, entry=₹{pos.entry_price}")

        # get_capital
        capital = provider.get_capital("XCU700")
        assert capital == Decimal("50000.00")
        print(f"  ✓ get_capital: ₹{capital}")

        # get_risk_tolerance
        from module3_risk_engine.models import RiskTolerance
        tolerance = provider.get_risk_tolerance("XCU700")
        assert tolerance == RiskTolerance.MODERATE
        print(f"  ✓ get_risk_tolerance: {tolerance.value}")

        # get_sector_exposure
        exposure = provider.get_sector_exposure("XCU700")
        assert "Banking" in exposure
        assert exposure["Banking"] == Decimal("1")
        print(f"  ✓ get_sector_exposure: {dict(exposure)}")

        # get_display_name
        name = provider.get_display_name("XCU700")
        assert name == "Vijay"
        print(f"  ✓ get_display_name: {name}")

        # get_open_positions
        positions = provider.get_open_positions("XCU700")
        assert len(positions) == 1
        assert isinstance(positions[0], M3OpenPosition)
        print(f"  ✓ get_open_positions: {len(positions)} position(s)")

        # available_capital
        total_invested = ctx["total_invested"]
        available = ctx["available_capital"]
        expected_invested = Decimal("1623.00") * 13  # = 21099.00
        assert total_invested == expected_invested, f"Expected {expected_invested}, got {total_invested}"
        expected_available = Decimal("50000.00") - expected_invested
        assert available == expected_available
        print(f"  ✓ Invested: ₹{total_invested}, Available: ₹{available}")

    print("  ✅ TEST 7 PASSED")


# ──────────────────────────────────────────────
# Test 8 — Watchlist CRUD
# ──────────────────────────────────────────────
def test_watchlist_crud():
    """Add, list, remove watchlist items."""
    print("\n" + "=" * 50)
    print("TEST 8 — Watchlist CRUD")
    print("=" * 50)

    from module5_memory.database.sqlite_manager import SQLiteManager
    from module5_memory.models import UserProfile, WatchlistItem

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        mgr = SQLiteManager(db_path)

        mgr.upsert_user_profile(UserProfile(user_id="XCU700"))

        # Add items
        item1 = WatchlistItem(
            user_id="XCU700",
            ticker="RELIANCE",
            alert_price=Decimal("2500.00"),
            notes="Watching for breakout",
        )
        item2 = WatchlistItem(
            user_id="XCU700",
            ticker="TCS",
            alert_price=Decimal("3800.00"),
        )
        wid1 = mgr.add_watchlist_item(item1)
        wid2 = mgr.add_watchlist_item(item2)
        print(f"  ✓ Added: RELIANCE ({wid1}), TCS ({wid2})")

        # List
        watchlist = mgr.get_watchlist("XCU700")
        assert len(watchlist) == 2
        tickers = {w.ticker for w in watchlist}
        assert tickers == {"RELIANCE", "TCS"}
        print(f"  ✓ Watchlist: {tickers}")

        # Remove one
        removed = mgr.remove_watchlist_item(wid1)
        assert removed is True
        remaining = mgr.get_watchlist("XCU700")
        assert len(remaining) == 1
        assert remaining[0].ticker == "TCS"
        print(f"  ✓ Removed RELIANCE, remaining: {remaining[0].ticker}")

        # Remove non-existent
        removed2 = mgr.remove_watchlist_item("nonexistent_id")
        assert removed2 is False
        print("  ✓ Remove non-existent returns False")

    print("  ✅ TEST 8 PASSED")


# ──────────────────────────────────────────────
# Test 9 — Engine Integration
# ──────────────────────────────────────────────
def test_engine_integration():
    """Full pipeline via MemoryEngine."""
    print("\n" + "=" * 50)
    print("TEST 9 — Engine Integration")
    print("=" * 50)

    from module5_memory.engine import MemoryEngine
    from module5_memory.models import (
        LearningProgress,
        TradeRecord,
        UserProfile,
        WatchlistItem,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        engine = MemoryEngine(db_path)

        # Create profile via engine
        profile = engine.get_or_create_profile("XCU700")
        assert profile.user_id == "XCU700"
        assert profile.name == "Vijay"
        print(f"  ✓ Profile: {profile.name}")

        # Save trade via engine
        trade = TradeRecord(
            user_id="XCU700",
            ticker="INFY",
            sector="IT",
            entry_price=Decimal("1500.00"),
            stop_loss=Decimal("1425.00"),
            target_price=Decimal("1725.00"),
            shares=10,
        )
        trade_id = engine.save_trade(trade)
        assert trade_id
        print(f"  ✓ Trade saved: {trade_id}")

        # Get trade history
        history = engine.get_trade_history("XCU700")
        assert len(history) == 1
        assert history[0].ticker == "INFY"
        print(f"  ✓ Trade history: {len(history)} trade(s)")

        # Close trade via engine
        closed = engine.close_trade(
            trade_id=trade_id,
            exit_price=Decimal("1725.00"),
            exit_reason="target_hit",
        )
        assert closed is not None
        expected_pnl = (Decimal("1725.00") - Decimal("1500.00")) * 10
        assert closed.pnl_rupees == expected_pnl
        print(f"  ✓ Trade closed: P&L = ₹{closed.pnl_rupees}")

        # Memory context
        ctx = engine.get_memory_context("XCU700")
        assert not ctx.is_empty
        assert ctx.within_budget
        print(f"  ✓ Memory context: {ctx.token_estimate} tokens, within_budget={ctx.within_budget}")

        # Learning via engine
        progress = LearningProgress(
            user_id="XCU700",
            concept="risk_reward_ratio",
            quiz_score=90,
        )
        engine.update_learning(progress)
        loaded = engine.get_learning("XCU700", "risk_reward_ratio")
        assert loaded is not None
        assert loaded.quiz_score == 90
        print(f"  ✓ Learning: {loaded.concept}, score={loaded.quiz_score}")

        # Watchlist via engine
        item = WatchlistItem(user_id="XCU700", ticker="SBIN")
        wid = engine.add_to_watchlist(item)
        wl = engine.get_watchlist("XCU700")
        assert len(wl) == 1
        engine.remove_from_watchlist(wid)
        wl2 = engine.get_watchlist("XCU700")
        assert len(wl2) == 0
        print("  ✓ Watchlist: add → list → remove")

        # M3 compat via engine
        capital = engine.get_capital("XCU700")
        assert capital == Decimal("50000.00")
        name = engine.get_display_name("XCU700")
        assert name == "Vijay"
        print(f"  ✓ M3 compat: capital=₹{capital}, name={name}")

    print("  ✅ TEST 9 PASSED")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────
def main():
    tests = [
        ("Test 1: SQLite Schema", test_sqlite_schema),
        ("Test 2: User Profile CRUD", test_user_profile_crud),
        ("Test 3: Trade Lifecycle", test_trade_lifecycle),
        ("Test 4: Learning Progress", test_learning_progress),
        ("Test 5: Chunker", test_chunker),
        ("Test 6: Memory Context Budget", test_memory_context_budget),
        ("Test 7: M3 Compatibility", test_m3_compatibility),
        ("Test 8: Watchlist CRUD", test_watchlist_crud),
        ("Test 9: Engine Integration", test_engine_integration),
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
    print("MODULE 5 TEST SUMMARY")
    print("=" * 50)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if errors:
        print("\n  Failed tests:")
        for name, err in errors:
            print(f"    ❌ {name}: {err}")

    if failed == 0:
        print("\n  🎉 ALL MODULE 5 TESTS PASSED!")
        print("  Memory engine is operational. The advisor remembers everything.")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed. Fix before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
