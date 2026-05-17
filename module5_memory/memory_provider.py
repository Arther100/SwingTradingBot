"""
SwingAdvisorBot — Module 5: Memory & Personalization
memory_provider.py — Unified memory interface for all modules

This is THE class other modules call. It replaces all TODO-M5 stubs
in M3/M4 with real SQLite + Pinecone backed data.

MemoryProvider wraps:
  - SQLiteManager  → structured CRUD (profile, trades, learning, watchlist)
  - ContextBuilder → ≤300 token context for Claude prompts
  - NamespaceManager → Pinecone store pipeline
  - VerificationEngine → 2-round advice verification

Also provides M3-compatible methods that match UserContextStub interface:
  get_user_context, get_open_positions, get_capital,
  get_risk_tolerance, get_sector_exposure, get_display_name

Usage:
    provider = MemoryProvider()
    profile = provider.get_user_profile("XCU700")
    context = provider.get_memory_context("XCU700", "HDFCBANK setup")
    provider.save_trade(trade_record)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from module5_memory.database.sqlite_manager import SQLiteManager
from module5_memory.models import (
    DailyStats,
    LearningProgress,
    MemoryContext,
    TradeRecord,
    TradeStatus,
    UserProfile,
    WatchlistItem,
)
from module5_memory.retrieval.context_builder import ContextBuilder
from module5_memory.retrieval.rag_retriever import RAGRetriever
from module5_memory.vector_store.namespace_manager import NamespaceManager
from module5_memory.verification.verification_engine import VerificationEngine

if TYPE_CHECKING:
    from module3_risk_engine.models import OpenPosition, RiskTolerance

logger = logging.getLogger("swing_advisor.m5_provider")


class MemoryProvider:
    """Unified memory interface for SwingAdvisorBot.

    Single entry point for all memory operations.
    Other modules import only this class.

    Graceful degradation at every layer:
      - Pinecone down → SQLite only
      - SQLite missing → empty defaults
      - Never blocks, never crashes

    Usage:
        provider = MemoryProvider()
        profile = provider.get_user_profile("XCU700")
        ctx = provider.get_memory_context("XCU700", "HDFCBANK setup")
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._sqlite = SQLiteManager(db_path)
        self._retriever = RAGRetriever()
        self._ns_manager = NamespaceManager()
        self._context_builder = ContextBuilder(self._sqlite, self._retriever)
        self._verification = VerificationEngine()

    # ═══════════════════════════════════════════════════════
    # USER PROFILE
    # ═══════════════════════════════════════════════════════

    def get_user_profile(self, user_id: str = "XCU700") -> UserProfile | None:
        """Get user profile from SQLite."""
        return self._sqlite.get_user_profile(user_id)

    def get_or_create_profile(self, user_id: str = "XCU700") -> UserProfile:
        """Get profile, creating default if not found."""
        profile = self._sqlite.get_user_profile(user_id)
        if profile:
            return profile

        # Create default profile
        profile = UserProfile(user_id=user_id)
        self._sqlite.upsert_user_profile(profile)
        logger.info(f"[Provider] Created default profile for {user_id}")
        return profile

    def update_profile(self, profile: UserProfile) -> None:
        """Update user profile in SQLite."""
        self._sqlite.upsert_user_profile(profile)

    # ═══════════════════════════════════════════════════════
    # MEMORY CONTEXT (for Claude prompts)
    # ═══════════════════════════════════════════════════════

    def get_memory_context(
        self,
        user_id: str = "XCU700",
        query: str = "",
        agent_name: str | None = None,
    ) -> MemoryContext:
        """Build ≤300 token memory context for Claude.

        This is the main method M2 calls before every Claude request.

        Args:
            user_id: Zerodha client ID.
            query: Query text for semantic search.
            agent_name: Agent name for namespace scoping.

        Returns:
            MemoryContext with text ready for prompt injection.
        """
        return self._context_builder.build_context(user_id, query, agent_name)

    # ═══════════════════════════════════════════════════════
    # TRADES
    # ═══════════════════════════════════════════════════════

    def save_trade(self, trade: TradeRecord) -> str:
        """Save a new trade to SQLite + Pinecone.

        Returns trade_id.
        """
        trade_id = self._sqlite.insert_trade(trade)
        self._ns_manager.store_trade(trade)
        logger.debug(f"[Provider] Saved trade {trade_id} ({trade.ticker})")
        return trade_id

    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_reason: str,
    ) -> TradeRecord | None:
        """Close an open trade — compute P&L, update SQLite + Pinecone.

        Returns updated TradeRecord or None.
        """
        from module5_memory.models import ExitReason

        reason = ExitReason(exit_reason)
        trade = self._sqlite.close_trade(trade_id, exit_price, reason)
        if trade:
            self._ns_manager.store_trade(trade)  # Re-embed with exit data
            self._update_profile_stats(trade.user_id)
        return trade

    def get_trade(self, trade_id: str) -> TradeRecord | None:
        """Get a single trade by ID."""
        return self._sqlite.get_trade(trade_id)

    def get_trade_history(
        self,
        user_id: str = "XCU700",
        limit: int = 50,
    ) -> list[TradeRecord]:
        """Get all trades for a user."""
        return self._sqlite.get_trades_by_user(user_id, limit=limit)

    def get_trades_by_ticker(
        self,
        ticker: str,
        user_id: str = "XCU700",
    ) -> list[TradeRecord]:
        """Get trades for a specific ticker."""
        return self._sqlite.get_trades_by_ticker(user_id, ticker)

    # ═══════════════════════════════════════════════════════
    # LEARNING PROGRESS
    # ═══════════════════════════════════════════════════════

    def update_learning(self, progress: LearningProgress) -> None:
        """Save learning progress to SQLite + Pinecone."""
        self._sqlite.upsert_learning(progress)
        self._ns_manager.store_learning(progress)

    def get_learning(
        self,
        user_id: str,
        concept: str,
    ) -> LearningProgress | None:
        """Get learning progress for a concept."""
        return self._sqlite.get_learning_by_concept(user_id, concept)

    def get_all_learning(self, user_id: str = "XCU700") -> list[LearningProgress]:
        """Get all learning records."""
        return self._sqlite.get_all_learning(user_id)

    # ═══════════════════════════════════════════════════════
    # WATCHLIST
    # ═══════════════════════════════════════════════════════

    def add_to_watchlist(self, item: WatchlistItem) -> str:
        """Add ticker to watchlist. Returns watchlist_id."""
        return self._sqlite.add_watchlist_item(item)

    def remove_from_watchlist(self, watchlist_id: str) -> bool:
        """Remove ticker from watchlist."""
        return self._sqlite.remove_watchlist_item(watchlist_id)

    def get_watchlist(self, user_id: str = "XCU700") -> list[WatchlistItem]:
        """Get user's watchlist."""
        return self._sqlite.get_watchlist(user_id)

    # ═══════════════════════════════════════════════════════
    # DAILY STATS
    # ═══════════════════════════════════════════════════════

    def save_daily_stats(self, stats: DailyStats) -> None:
        """Save daily stats."""
        self._sqlite.upsert_daily_stats(stats)

    def get_daily_stats(self, user_id: str, date: str) -> DailyStats | None:
        """Get stats for a specific date."""
        return self._sqlite.get_daily_stats(user_id, date)

    # ═══════════════════════════════════════════════════════
    # VERIFICATION
    # ═══════════════════════════════════════════════════════

    async def verify_advice(
        self,
        advice: str,
        user_id: str = "XCU700",
        query: str = "",
    ):
        """Verify generated advice against user history.

        Returns VerificationResult.
        """
        profile = self.get_or_create_profile(user_id)
        chunks = self._retriever.retrieve(query) if query else []

        return await self._verification.verify(
            round1_advice=advice,
            user_profile=profile,
            relevant_chunks=chunks,
        )

    # ═══════════════════════════════════════════════════════
    # M3 COMPATIBILITY — replaces UserContextStub
    # ═══════════════════════════════════════════════════════

    def get_user_context(self, user_id: str = "XCU700") -> dict:
        """M3-compatible: Return user context for risk calculations.

        Replaces UserContextStub.get_user_context().
        """
        from module3_risk_engine.models import OpenPosition as M3OpenPosition, RiskTolerance

        profile = self.get_or_create_profile(user_id)
        open_trades = self._sqlite.get_trades_by_user(
            user_id, status=TradeStatus.OPEN
        )
        sector_exposure = self._sqlite.get_sector_exposure(user_id)

        # Convert M5 trades to M3 OpenPosition
        positions = [
            M3OpenPosition(
                ticker=t.ticker,
                sector=t.sector,
                entry_price=t.entry_price,
                quantity=t.shares,
                stop_loss=t.stop_loss,
                target=t.target_price,
                entry_date=t.entry_date,
            )
            for t in open_trades
        ]

        total_invested = sum(
            t.entry_price * t.shares for t in open_trades
        )

        tolerance_map = {
            "conservative": RiskTolerance.CONSERVATIVE,
            "moderate": RiskTolerance.MODERATE,
            "aggressive": RiskTolerance.AGGRESSIVE,
        }

        return {
            "user_id": profile.user_id,
            "name": profile.name,
            "capital": profile.capital,
            "risk_tolerance": tolerance_map.get(
                profile.risk_tolerance, RiskTolerance.MODERATE
            ),
            "open_positions": positions,
            "sector_exposure": {
                sector: Decimal(str(count)) for sector, count in sector_exposure.items()
            },
            "total_invested": total_invested,
            "available_capital": profile.capital - total_invested,
        }

    def get_open_positions(self, user_id: str = "XCU700") -> list:
        """M3-compatible: Return open positions as M3 OpenPosition list.

        Replaces UserContextStub.get_open_positions().
        """
        from module3_risk_engine.models import OpenPosition as M3OpenPosition

        open_trades = self._sqlite.get_trades_by_user(
            user_id, status=TradeStatus.OPEN
        )
        return [
            M3OpenPosition(
                ticker=t.ticker,
                sector=t.sector,
                entry_price=t.entry_price,
                quantity=t.shares,
                stop_loss=t.stop_loss,
                target=t.target_price,
                entry_date=t.entry_date,
            )
            for t in open_trades
        ]

    def get_capital(self, user_id: str = "XCU700") -> Decimal:
        """M3-compatible: Return trading capital.

        Replaces UserContextStub.get_capital().
        """
        profile = self.get_or_create_profile(user_id)
        return profile.capital

    def get_risk_tolerance(self, user_id: str = "XCU700"):
        """M3-compatible: Return risk tolerance.

        Replaces UserContextStub.get_risk_tolerance().
        """
        from module3_risk_engine.models import RiskTolerance

        profile = self.get_or_create_profile(user_id)
        tolerance_map = {
            "conservative": RiskTolerance.CONSERVATIVE,
            "moderate": RiskTolerance.MODERATE,
            "aggressive": RiskTolerance.AGGRESSIVE,
        }
        return tolerance_map.get(profile.risk_tolerance, RiskTolerance.MODERATE)

    def get_sector_exposure(self, user_id: str = "XCU700") -> dict[str, Decimal]:
        """M3-compatible: Return sector exposure.

        Replaces UserContextStub.get_sector_exposure().
        """
        exposure = self._sqlite.get_sector_exposure(user_id)
        return {sector: Decimal(str(count)) for sector, count in exposure.items()}

    def get_display_name(self, user_id: str = "XCU700") -> str:
        """M3-compatible: Return display name.

        Replaces UserContextStub.get_display_name().
        """
        profile = self.get_or_create_profile(user_id)
        return profile.name

    # ═══════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════

    def _update_profile_stats(self, user_id: str) -> None:
        """Recompute aggregate stats from trades and update profile."""
        all_trades = self._sqlite.get_trades_by_user(user_id, limit=10000)
        open_trades = [t for t in all_trades if t.status == TradeStatus.OPEN]
        closed_trades = [t for t in all_trades if t.status != TradeStatus.OPEN]

        total_trades = len(all_trades)
        winning = sum(
            1 for t in closed_trades if t.pnl_rupees and t.pnl_rupees > 0
        )
        total_pnl = sum(
            (t.pnl_rupees or Decimal("0")) for t in closed_trades
        )

        self._sqlite.update_user_stats(
            user_id=user_id,
            total_trades=total_trades,
            winning_trades=winning,
            total_pnl=total_pnl,
            open_positions_count=len(open_trades),
        )
