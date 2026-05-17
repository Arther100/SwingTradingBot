"""
SwingAdvisorBot — Module 5: Memory & Personalization
database/schema.py — SQLite table creation and migration

Creates all tables in memory.db on first run.
Idempotent — safe to call multiple times (IF NOT EXISTS).

Tables:
  user_profiles       → User identity, capital, risk tolerance
  trades              → Every trade with full lifecycle
  learning_progress   → Concepts taught and quiz scores
  watchlist           → User's watchlist entries
  daily_stats         → Daily performance snapshots

All financial amounts stored as TEXT (Decimal as string).
All timestamps stored as TEXT (ISO format, IST).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from module5_memory.config import SQLITE_DB_PATH, SQLITE_JOURNAL_MODE

logger = logging.getLogger("swing_advisor.m5_schema")


# ─────────────────────────────────────────────────────────────
# SQL Table Definitions
# ─────────────────────────────────────────────────────────────

CREATE_USER_PROFILES = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id              TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    capital              TEXT NOT NULL DEFAULT '50000.00',
    risk_tolerance       TEXT NOT NULL DEFAULT 'moderate',
    total_trades         INTEGER DEFAULT 0,
    winning_trades       INTEGER DEFAULT 0,
    total_pnl            TEXT DEFAULT '0.00',
    open_positions_count INTEGER DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
"""

CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id       TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    sector         TEXT DEFAULT 'Other',
    entry_price    TEXT NOT NULL,
    exit_price     TEXT,
    stop_loss      TEXT NOT NULL,
    target_price   TEXT NOT NULL,
    shares         INTEGER NOT NULL,
    entry_date     TEXT NOT NULL,
    exit_date      TEXT,
    status         TEXT NOT NULL DEFAULT 'open',
    pnl_rupees     TEXT,
    pnl_pct        TEXT,
    exit_reason    TEXT,
    market_mood    TEXT,
    vix_at_entry   TEXT,
    setup_source   TEXT DEFAULT 'm4_generated',
    notes          TEXT,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);
"""

CREATE_LEARNING_PROGRESS = """
CREATE TABLE IF NOT EXISTS learning_progress (
    progress_id    TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    concept        TEXT NOT NULL,
    taught_at      TEXT NOT NULL,
    quiz_score     INTEGER,
    times_taught   INTEGER DEFAULT 1,
    last_taught    TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);
"""

CREATE_WATCHLIST = """
CREATE TABLE IF NOT EXISTS watchlist (
    watchlist_id   TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    added_at       TEXT NOT NULL,
    alert_price    TEXT,
    notes          TEXT,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);
"""

CREATE_DAILY_STATS = """
CREATE TABLE IF NOT EXISTS daily_stats (
    stat_id        TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    date           TEXT NOT NULL,
    total_trades   INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    total_pnl      TEXT DEFAULT '0.00',
    win_rate       TEXT DEFAULT '0.00',
    best_trade     TEXT,
    worst_trade    TEXT,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);
"""

# Indexes for fast queries
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);",
    "CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON trades(entry_date);",
    "CREATE INDEX IF NOT EXISTS idx_learning_user ON learning_progress(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_learning_concept ON learning_progress(concept);",
    "CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_daily_stats_user_date ON daily_stats(user_id, date);",
]

ALL_TABLES = [
    CREATE_USER_PROFILES,
    CREATE_TRADES,
    CREATE_LEARNING_PROGRESS,
    CREATE_WATCHLIST,
    CREATE_DAILY_STATS,
]


# ─────────────────────────────────────────────────────────────
# Schema Creation
# ─────────────────────────────────────────────────────────────


def initialize_database(db_path: str | None = None) -> None:
    """Create all tables and indexes in SQLite.

    Idempotent — safe to call on every startup.
    Creates the data directory if it doesn't exist.

    Args:
        db_path: Override database path (for testing).
                 Defaults to SQLITE_DB_PATH from config.
    """
    path = db_path or SQLITE_DB_PATH

    # Ensure directory exists
    db_dir = Path(path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        # Enable WAL mode for concurrent reads
        conn.execute(f"PRAGMA journal_mode={SQLITE_JOURNAL_MODE};")
        conn.execute("PRAGMA foreign_keys=ON;")

        # Create tables
        for sql in ALL_TABLES:
            conn.execute(sql)

        # Create indexes
        for sql in CREATE_INDEXES:
            conn.execute(sql)

        conn.commit()
        logger.info(f"[Schema] Database initialized at {path}")

    finally:
        conn.close()


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with proper settings.

    Args:
        db_path: Override database path.
                 Defaults to SQLITE_DB_PATH from config.

    Returns:
        sqlite3.Connection with WAL mode and foreign keys enabled.
    """
    path = db_path or SQLITE_DB_PATH

    # Ensure DB exists
    if not Path(path).exists():
        initialize_database(path)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def drop_all_tables(db_path: str | None = None) -> None:
    """Drop all tables — for testing only.

    Args:
        db_path: Override database path.
    """
    path = db_path or SQLITE_DB_PATH

    if not Path(path).exists():
        return

    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP TABLE IF EXISTS daily_stats;")
        conn.execute("DROP TABLE IF EXISTS watchlist;")
        conn.execute("DROP TABLE IF EXISTS learning_progress;")
        conn.execute("DROP TABLE IF EXISTS trades;")
        conn.execute("DROP TABLE IF EXISTS user_profiles;")
        conn.commit()
        logger.info(f"[Schema] All tables dropped at {path}")
    finally:
        conn.close()
