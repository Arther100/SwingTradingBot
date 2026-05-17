"""
SwingAdvisorBot — Module 5: Memory & Personalization
engine.py — Public API entry point

This is the single import other modules need:
  from module5_memory.engine import memory_engine

The engine exposes public methods matching the 5 MCP tools:
  1. get_user_profile()      → User profile from SQLite
  2. save_trade()            → Save trade to SQLite + Pinecone
  3. get_trade_history()     → Trade list from SQLite
  4. get_memory_context()    → ≤300 token context for Claude
  5. update_learning()       → Learning progress to SQLite + Pinecone

Plus M3-compatible methods:
  6. get_user_context()      → Dict for risk calculations
  7. get_open_positions()    → M3 OpenPosition list
  8. get_capital()           → Decimal capital
  9. get_risk_tolerance()    → M3 RiskTolerance enum
  10. get_sector_exposure()  → Dict of sector counts
  11. get_display_name()     → User name string

And verification:
  12. verify_advice()        → 2-round verification via Claude

These methods are the Python API. The MCP tools in mcp_tools.py
are the HTTP API — both use MemoryProvider underneath.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from module5_memory.memory_provider import MemoryProvider
from module5_memory.models import (
    DailyStats,
    LearningProgress,
    MemoryContext,
    TradeRecord,
    UserProfile,
    VerificationResult,
    WatchlistItem,
)

logger = logging.getLogger("swing_advisor.memory_engine")


class MemoryEngine:
    """Public API for Module 5 — Memory & Personalization.

    Usage:
        from module5_memory.engine import memory_engine

        # Get user profile
        profile = memory_engine.get_user_profile("XCU700")

        # Get memory context for Claude
        ctx = memory_engine.get_memory_context("XCU700", "HDFCBANK setup")

        # Save a trade
        trade_id = memory_engine.save_trade(trade_record)

        # M3 compatibility
        context = memory_engine.get_user_context("XCU700")
        positions = memory_engine.get_open_positions("XCU700")
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._provider = MemoryProvider(db_path)

    # ═══════════════════════════════════════════════════════
    # USER PROFILE
    # ═══════════════════════════════════════════════════════

    def get_user_profile(self, user_id: str = "XCU700") -> UserProfile | None:
        """Get user profile from SQLite."""
        return self._provider.get_user_profile(user_id)

    def get_or_create_profile(self, user_id: str = "XCU700") -> UserProfile:
        """Get profile, creating default if not found."""
        return self._provider.get_or_create_profile(user_id)

    def update_profile(self, profile: UserProfile) -> None:
        """Update user profile."""
        self._provider.update_profile(profile)

    # ═══════════════════════════════════════════════════════
    # MEMORY CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_memory_context(
        self,
        user_id: str = "XCU700",
        query: str = "",
        agent_name: str | None = None,
    ) -> MemoryContext:
        """Build ≤300 token memory context for Claude prompt.

        This is the main method M2 calls before every Claude request.
        """
        return self._provider.get_memory_context(user_id, query, agent_name)

    # ═══════════════════════════════════════════════════════
    # TRADES
    # ═══════════════════════════════════════════════════════

    def save_trade(self, trade: TradeRecord) -> str:
        """Save trade to SQLite + Pinecone. Returns trade_id."""
        return self._provider.save_trade(trade)

    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_reason: str,
    ) -> TradeRecord | None:
        """Close a trade — auto-computes P&L."""
        return self._provider.close_trade(trade_id, exit_price, exit_reason)

    def get_trade(self, trade_id: str) -> TradeRecord | None:
        """Get a single trade by ID."""
        return self._provider.get_trade(trade_id)

    def get_trade_history(
        self,
        user_id: str = "XCU700",
        limit: int = 50,
    ) -> list[TradeRecord]:
        """Get trade history for a user."""
        return self._provider.get_trade_history(user_id, limit)

    def get_trades_by_ticker(
        self,
        ticker: str,
        user_id: str = "XCU700",
    ) -> list[TradeRecord]:
        """Get trades for a specific ticker."""
        return self._provider.get_trades_by_ticker(ticker, user_id)

    # ═══════════════════════════════════════════════════════
    # LEARNING
    # ═══════════════════════════════════════════════════════

    def update_learning(self, progress: LearningProgress) -> None:
        """Save learning progress to SQLite + Pinecone."""
        self._provider.update_learning(progress)

    def get_learning(
        self,
        user_id: str,
        concept: str,
    ) -> LearningProgress | None:
        """Get learning progress for a concept."""
        return self._provider.get_learning(user_id, concept)

    def get_all_learning(self, user_id: str = "XCU700") -> list[LearningProgress]:
        """Get all learning records."""
        return self._provider.get_all_learning(user_id)

    # ═══════════════════════════════════════════════════════
    # WATCHLIST
    # ═══════════════════════════════════════════════════════

    def add_to_watchlist(self, item: WatchlistItem) -> str:
        """Add ticker to watchlist. Returns watchlist_id."""
        return self._provider.add_to_watchlist(item)

    def remove_from_watchlist(self, watchlist_id: str) -> bool:
        """Remove ticker from watchlist."""
        return self._provider.remove_from_watchlist(watchlist_id)

    def get_watchlist(self, user_id: str = "XCU700") -> list[WatchlistItem]:
        """Get user's watchlist."""
        return self._provider.get_watchlist(user_id)

    # ═══════════════════════════════════════════════════════
    # DAILY STATS
    # ═══════════════════════════════════════════════════════

    def save_daily_stats(self, stats: DailyStats) -> None:
        """Save daily performance stats."""
        self._provider.save_daily_stats(stats)

    def get_daily_stats(self, user_id: str, date: str) -> DailyStats | None:
        """Get stats for a specific date."""
        return self._provider.get_daily_stats(user_id, date)

    # ═══════════════════════════════════════════════════════
    # VERIFICATION
    # ═══════════════════════════════════════════════════════

    async def verify_advice(
        self,
        advice: str,
        user_id: str = "XCU700",
        query: str = "",
    ) -> VerificationResult:
        """Verify generated advice against user history (2-round)."""
        return await self._provider.verify_advice(advice, user_id, query)

    # ═══════════════════════════════════════════════════════
    # M3 COMPATIBILITY — replaces UserContextStub
    # ═══════════════════════════════════════════════════════

    def get_user_context(self, user_id: str = "XCU700") -> dict:
        """M3-compatible user context for risk calculations."""
        return self._provider.get_user_context(user_id)

    def get_open_positions(self, user_id: str = "XCU700") -> list:
        """M3-compatible open positions list."""
        return self._provider.get_open_positions(user_id)

    def get_capital(self, user_id: str = "XCU700") -> Decimal:
        """M3-compatible trading capital."""
        return self._provider.get_capital(user_id)

    def get_risk_tolerance(self, user_id: str = "XCU700"):
        """M3-compatible risk tolerance."""
        return self._provider.get_risk_tolerance(user_id)

    def get_sector_exposure(self, user_id: str = "XCU700") -> dict[str, Decimal]:
        """M3-compatible sector exposure."""
        return self._provider.get_sector_exposure(user_id)

    def get_display_name(self, user_id: str = "XCU700") -> str:
        """M3-compatible display name."""
        return self._provider.get_display_name(user_id)


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

memory_engine = MemoryEngine()
