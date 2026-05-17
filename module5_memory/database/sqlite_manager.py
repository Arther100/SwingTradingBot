"""
SwingAdvisorBot — Module 5: Memory & Personalization
database/sqlite_manager.py — CRUD operations for all SQLite tables

Provides SQLiteManager class with methods for every table:
  UserProfile  → upsert, get, update capital/stats
  TradeRecord  → insert, update (close), get by id/ticker/status, list
  Learning     → upsert, get by concept, list
  Watchlist    → add, remove, list
  DailyStats   → upsert, get by date, recent

All Decimal fields stored as TEXT, loaded back as Decimal.
All datetime fields stored as ISO string, loaded back as datetime.
Parameterized queries only — no string interpolation (SQL injection safe).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from decimal import Decimal
from typing import Optional

from zoneinfo import ZoneInfo

from module5_memory.database.schema import get_connection, initialize_database
from module5_memory.models import (
    DailyStats,
    ExitReason,
    LearningProgress,
    SetupSource,
    TradeRecord,
    TradeStatus,
    UserProfile,
    WatchlistItem,
)

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.m5_sqlite")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _dt_to_str(dt: datetime | None) -> str | None:
    """Convert datetime to ISO string for SQLite storage."""
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    """Convert ISO string from SQLite back to datetime."""
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt


def _dec_to_str(d: Decimal | None) -> str | None:
    """Convert Decimal to string for SQLite storage."""
    if d is None:
        return None
    return str(d)


def _str_to_dec(s: str | None) -> Decimal | None:
    """Convert string from SQLite back to Decimal."""
    if not s:
        return None
    return Decimal(s)


# ─────────────────────────────────────────────────────────────
# SQLiteManager
# ─────────────────────────────────────────────────────────────


class SQLiteManager:
    """CRUD operations for Module 5 SQLite tables.

    Usage:
        mgr = SQLiteManager()                 # uses default db
        mgr = SQLiteManager(db_path=":memory:") # for tests

    All methods use parameterized queries.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        initialize_database(db_path)

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db_path)

    # ═══════════════════════════════════════════════════════
    # USER PROFILE
    # ═══════════════════════════════════════════════════════

    def upsert_user_profile(self, profile: UserProfile) -> None:
        """Insert or update user profile."""
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, name, capital, risk_tolerance,
                    total_trades, winning_trades, total_pnl, open_positions_count,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,
                    capital = excluded.capital,
                    risk_tolerance = excluded.risk_tolerance,
                    total_trades = excluded.total_trades,
                    winning_trades = excluded.winning_trades,
                    total_pnl = excluded.total_pnl,
                    open_positions_count = excluded.open_positions_count,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.user_id,
                    profile.name,
                    _dec_to_str(profile.capital),
                    profile.risk_tolerance,
                    profile.total_trades,
                    profile.winning_trades,
                    _dec_to_str(profile.total_pnl),
                    profile.open_positions_count,
                    _dt_to_str(profile.created_at),
                    _dt_to_str(profile.updated_at),
                ),
            )
            conn.commit()
            logger.debug(f"[SQLite] Upserted profile: {profile.user_id}")
        finally:
            conn.close()

    def get_user_profile(self, user_id: str) -> UserProfile | None:
        """Get user profile by ID. Returns None if not found."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if not row:
                return None

            return UserProfile(
                user_id=row["user_id"],
                name=row["name"],
                capital=Decimal(row["capital"]),
                risk_tolerance=row["risk_tolerance"],
                total_trades=row["total_trades"] or 0,
                winning_trades=row["winning_trades"] or 0,
                total_pnl=Decimal(row["total_pnl"]) if row["total_pnl"] else Decimal("0.00"),
                open_positions_count=row["open_positions_count"] or 0,
                created_at=_str_to_dt(row["created_at"]),
                updated_at=_str_to_dt(row["updated_at"]),
            )
        finally:
            conn.close()

    def update_user_capital(self, user_id: str, capital: Decimal) -> None:
        """Update only the capital field."""
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE user_profiles SET capital = ?, updated_at = ? WHERE user_id = ?",
                (_dec_to_str(capital), _dt_to_str(datetime.now(IST)), user_id),
            )
            conn.commit()
        finally:
            conn.close()

    def update_user_stats(
        self,
        user_id: str,
        total_trades: int,
        winning_trades: int,
        total_pnl: Decimal,
        open_positions_count: int,
    ) -> None:
        """Update aggregated trade stats on profile.

        Note: user_profiles table doesn't store these stats directly —
        they are computed fields on the UserProfile model.
        We store them by updating the profile's updated_at timestamp
        and the caller should re-upsert the full profile.
        """
        profile = self.get_user_profile(user_id)
        if not profile:
            return

        profile.total_trades = total_trades
        profile.winning_trades = winning_trades
        profile.total_pnl = total_pnl
        profile.open_positions_count = open_positions_count
        profile.updated_at = datetime.now(IST)
        self.upsert_user_profile(profile)

    # ═══════════════════════════════════════════════════════
    # TRADES
    # ═══════════════════════════════════════════════════════

    def insert_trade(self, trade: TradeRecord) -> str:
        """Insert a new trade record. Returns trade_id."""
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO trades (
                    trade_id, user_id, ticker, sector,
                    entry_price, exit_price, stop_loss, target_price,
                    shares, entry_date, exit_date,
                    status, pnl_rupees, pnl_pct, exit_reason,
                    market_mood, vix_at_entry, setup_source, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.trade_id,
                    trade.user_id,
                    trade.ticker,
                    trade.sector,
                    _dec_to_str(trade.entry_price),
                    _dec_to_str(trade.exit_price),
                    _dec_to_str(trade.stop_loss),
                    _dec_to_str(trade.target_price),
                    trade.shares,
                    _dt_to_str(trade.entry_date),
                    _dt_to_str(trade.exit_date),
                    trade.status.value,
                    _dec_to_str(trade.pnl_rupees),
                    _dec_to_str(trade.pnl_pct),
                    trade.exit_reason.value if trade.exit_reason else None,
                    trade.market_mood,
                    _dec_to_str(trade.vix_at_entry),
                    trade.setup_source.value,
                    trade.notes,
                ),
            )
            conn.commit()
            logger.debug(f"[SQLite] Inserted trade: {trade.trade_id} ({trade.ticker})")
            return trade.trade_id
        finally:
            conn.close()

    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        """Convert a SQLite Row to TradeRecord."""
        return TradeRecord(
            trade_id=row["trade_id"],
            user_id=row["user_id"],
            ticker=row["ticker"],
            sector=row["sector"],
            entry_price=Decimal(row["entry_price"]),
            exit_price=_str_to_dec(row["exit_price"]),
            stop_loss=Decimal(row["stop_loss"]),
            target_price=Decimal(row["target_price"]),
            shares=row["shares"],
            entry_date=_str_to_dt(row["entry_date"]),
            exit_date=_str_to_dt(row["exit_date"]),
            status=TradeStatus(row["status"]),
            pnl_rupees=_str_to_dec(row["pnl_rupees"]),
            pnl_pct=_str_to_dec(row["pnl_pct"]),
            exit_reason=ExitReason(row["exit_reason"]) if row["exit_reason"] else None,
            market_mood=row["market_mood"],
            vix_at_entry=_str_to_dec(row["vix_at_entry"]),
            setup_source=SetupSource(row["setup_source"]) if row["setup_source"] else SetupSource.M4_GENERATED,
            notes=row["notes"],
        )

    def get_trade(self, trade_id: str) -> TradeRecord | None:
        """Get a single trade by ID."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
            return self._row_to_trade(row) if row else None
        finally:
            conn.close()

    def get_trades_by_user(
        self,
        user_id: str,
        status: TradeStatus | None = None,
        limit: int = 50,
    ) -> list[TradeRecord]:
        """Get trades for a user, optionally filtered by status."""
        conn = self._conn()
        try:
            if status:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE user_id = ? AND status = ? ORDER BY entry_date DESC LIMIT ?",
                    (user_id, status.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE user_id = ? ORDER BY entry_date DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            return [self._row_to_trade(r) for r in rows]
        finally:
            conn.close()

    def get_trades_by_ticker(
        self,
        user_id: str,
        ticker: str,
        limit: int = 10,
    ) -> list[TradeRecord]:
        """Get trades for a specific ticker."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM trades WHERE user_id = ? AND ticker = ? ORDER BY entry_date DESC LIMIT ?",
                (user_id, ticker, limit),
            ).fetchall()
            return [self._row_to_trade(r) for r in rows]
        finally:
            conn.close()

    def get_open_trades(self, user_id: str) -> list[TradeRecord]:
        """Get all open trades for a user."""
        return self.get_trades_by_user(user_id, status=TradeStatus.OPEN)

    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_reason: ExitReason,
        exit_date: datetime | None = None,
    ) -> TradeRecord | None:
        """Close an open trade — compute P&L and update status.

        Returns updated TradeRecord or None if trade not found.
        """
        trade = self.get_trade(trade_id)
        if not trade:
            return None

        exit_dt = exit_date or datetime.now(IST)
        pnl_rupees = (exit_price - trade.entry_price) * trade.shares
        pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100

        new_status = (
            TradeStatus.STOPPED_OUT
            if exit_reason == ExitReason.STOP_HIT
            else TradeStatus.CLOSED
        )

        conn = self._conn()
        try:
            conn.execute(
                """
                UPDATE trades SET
                    exit_price = ?, exit_date = ?, exit_reason = ?,
                    status = ?, pnl_rupees = ?, pnl_pct = ?
                WHERE trade_id = ?
                """,
                (
                    _dec_to_str(exit_price),
                    _dt_to_str(exit_dt),
                    exit_reason.value,
                    new_status.value,
                    _dec_to_str(pnl_rupees),
                    _dec_to_str(pnl_pct.quantize(Decimal("0.01"))),
                    trade_id,
                ),
            )
            conn.commit()
            logger.debug(f"[SQLite] Closed trade {trade_id}: P&L ₹{pnl_rupees}")
        finally:
            conn.close()

        return self.get_trade(trade_id)

    def get_sector_exposure(self, user_id: str) -> dict[str, int]:
        """Get count of open positions per sector.

        Returns: {"IT": 2, "Banking": 1, ...}
        """
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT sector, COUNT(*) as cnt FROM trades WHERE user_id = ? AND status = ? GROUP BY sector",
                (user_id, TradeStatus.OPEN.value),
            ).fetchall()
            return {row["sector"]: row["cnt"] for row in rows}
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════
    # LEARNING PROGRESS
    # ═══════════════════════════════════════════════════════

    def upsert_learning(self, progress: LearningProgress) -> None:
        """Insert or update learning progress for a concept."""
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO learning_progress (progress_id, user_id, concept, taught_at, quiz_score, times_taught, last_taught)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(progress_id) DO UPDATE SET
                    quiz_score = excluded.quiz_score,
                    times_taught = excluded.times_taught,
                    last_taught = excluded.last_taught
                """,
                (
                    progress.progress_id,
                    progress.user_id,
                    progress.concept,
                    _dt_to_str(progress.taught_at),
                    progress.quiz_score,
                    progress.times_taught,
                    _dt_to_str(progress.last_taught),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_learning_by_concept(
        self,
        user_id: str,
        concept: str,
    ) -> LearningProgress | None:
        """Get learning progress for a specific concept."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM learning_progress WHERE user_id = ? AND concept = ?",
                (user_id, concept),
            ).fetchone()

            if not row:
                return None

            return LearningProgress(
                progress_id=row["progress_id"],
                user_id=row["user_id"],
                concept=row["concept"],
                taught_at=_str_to_dt(row["taught_at"]),
                quiz_score=row["quiz_score"],
                times_taught=row["times_taught"],
                last_taught=_str_to_dt(row["last_taught"]),
            )
        finally:
            conn.close()

    def get_all_learning(self, user_id: str) -> list[LearningProgress]:
        """Get all learning progress records for a user."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM learning_progress WHERE user_id = ? ORDER BY last_taught DESC",
                (user_id,),
            ).fetchall()
            return [
                LearningProgress(
                    progress_id=r["progress_id"],
                    user_id=r["user_id"],
                    concept=r["concept"],
                    taught_at=_str_to_dt(r["taught_at"]),
                    quiz_score=r["quiz_score"],
                    times_taught=r["times_taught"],
                    last_taught=_str_to_dt(r["last_taught"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════
    # WATCHLIST
    # ═══════════════════════════════════════════════════════

    def add_watchlist_item(self, item: WatchlistItem) -> str:
        """Add item to watchlist. Returns watchlist_id."""
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO watchlist (watchlist_id, user_id, ticker, added_at, alert_price, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.watchlist_id,
                    item.user_id,
                    item.ticker,
                    _dt_to_str(item.added_at),
                    _dec_to_str(item.alert_price),
                    item.notes,
                ),
            )
            conn.commit()
            return item.watchlist_id
        finally:
            conn.close()

    def remove_watchlist_item(self, watchlist_id: str) -> bool:
        """Remove item from watchlist. Returns True if deleted."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "DELETE FROM watchlist WHERE watchlist_id = ?",
                (watchlist_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_watchlist(self, user_id: str) -> list[WatchlistItem]:
        """Get all watchlist items for a user."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM watchlist WHERE user_id = ? ORDER BY added_at DESC",
                (user_id,),
            ).fetchall()
            return [
                WatchlistItem(
                    watchlist_id=r["watchlist_id"],
                    user_id=r["user_id"],
                    ticker=r["ticker"],
                    added_at=_str_to_dt(r["added_at"]),
                    alert_price=_str_to_dec(r["alert_price"]),
                    notes=r["notes"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════
    # DAILY STATS
    # ═══════════════════════════════════════════════════════

    def upsert_daily_stats(self, stats: DailyStats) -> None:
        """Insert or update daily stats."""
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO daily_stats (stat_id, user_id, date, total_trades, winning_trades, total_pnl, win_rate, best_trade, worst_trade)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stat_id) DO UPDATE SET
                    total_trades = excluded.total_trades,
                    winning_trades = excluded.winning_trades,
                    total_pnl = excluded.total_pnl,
                    win_rate = excluded.win_rate,
                    best_trade = excluded.best_trade,
                    worst_trade = excluded.worst_trade
                """,
                (
                    stats.stat_id,
                    stats.user_id,
                    stats.date,
                    stats.total_trades,
                    stats.winning_trades,
                    _dec_to_str(stats.total_pnl),
                    _dec_to_str(stats.win_rate),
                    stats.best_trade,
                    stats.worst_trade,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_daily_stats(self, user_id: str, date: str) -> DailyStats | None:
        """Get stats for a specific date (YYYY-MM-DD)."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM daily_stats WHERE user_id = ? AND date = ?",
                (user_id, date),
            ).fetchone()

            if not row:
                return None

            return DailyStats(
                stat_id=row["stat_id"],
                user_id=row["user_id"],
                date=row["date"],
                total_trades=row["total_trades"],
                winning_trades=row["winning_trades"],
                total_pnl=Decimal(row["total_pnl"]),
                win_rate=Decimal(row["win_rate"]),
                best_trade=row["best_trade"],
                worst_trade=row["worst_trade"],
            )
        finally:
            conn.close()

    def get_recent_stats(
        self,
        user_id: str,
        limit: int = 7,
    ) -> list[DailyStats]:
        """Get most recent daily stats."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM daily_stats WHERE user_id = ? ORDER BY date DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [
                DailyStats(
                    stat_id=r["stat_id"],
                    user_id=r["user_id"],
                    date=r["date"],
                    total_trades=r["total_trades"],
                    winning_trades=r["winning_trades"],
                    total_pnl=Decimal(r["total_pnl"]),
                    win_rate=Decimal(r["win_rate"]),
                    best_trade=r["best_trade"],
                    worst_trade=r["worst_trade"],
                )
                for r in rows
            ]
        finally:
            conn.close()
