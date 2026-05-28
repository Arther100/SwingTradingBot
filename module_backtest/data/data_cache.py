"""
SwingAdvisorBot — Module Backtest
data/data_cache.py — SQLite cache for historical OHLCV + backtest results

Three tables:
  historical_ohlcv   — Daily price bars per ticker (never re-fetch from Kite)
  backtest_results   — Serialised BacktestResult / PortfolioBacktestResult JSON
  signal_weights     — Current M4 signal weights after backtest adjustment

Design:
  - Thread-safe via WAL mode + per-connection context managers
  - Parameterized queries only (SQL injection safe)
  - UNIQUE(ticker, date) on ohlcv prevents duplicate rows
  - INSERT OR IGNORE for idempotent batch inserts
  - Incremental — only new bars are inserted, existing bars are never updated
  - Data dir created automatically if it doesn't exist

Usage:
    from module_backtest.data.data_cache import historical_cache

    # Store OHLCV bars
    historical_cache.store_bars("HDFCBANK", bars)

    # Load OHLCV bars for a date range
    bars = historical_cache.get_bars("HDFCBANK", from_date, to_date)

    # Store / retrieve a BacktestResult
    historical_cache.store_backtest_result(result)
    result = historical_cache.get_backtest_result("HDFCBANK", "breakout_watch")

    # Signal weights
    historical_cache.store_signal_weight(weight)
    weights = historical_cache.get_all_signal_weights()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from zoneinfo import ZoneInfo

from module_backtest.config import (
    BACKTEST_RESULTS_DB_PATH,
    HISTORICAL_DB_PATH,
    SIGNAL_WEIGHTS_DB_PATH,
)
from module_backtest.models import (
    BacktestResult,
    OHLCVBar,
    PortfolioBacktestResult,
    SignalType,
    SignalWeight,
)

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.backtest.data_cache")


# ═══════════════════════════════════════════════════════════
# SCHEMA INITIALIZATION
# ═══════════════════════════════════════════════════════════

_OHLCV_SCHEMA = """
CREATE TABLE IF NOT EXISTS historical_ohlcv (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,   -- ISO date YYYY-MM-DD
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      INTEGER NOT NULL DEFAULT 0,
    fetched_at  TEXT    NOT NULL,   -- ISO datetime when fetched from Kite
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date
    ON historical_ohlcv (ticker, date);
"""

_RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    signal_type     TEXT    NOT NULL,
    period_start    TEXT    NOT NULL,
    period_end      TEXT    NOT NULL,
    result_json     TEXT    NOT NULL,   -- JSON-serialised BacktestResult
    generated_at    TEXT    NOT NULL,
    UNIQUE(ticker, signal_type, period_start, period_end)
);

CREATE TABLE IF NOT EXISTS portfolio_backtest_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start    TEXT    NOT NULL,
    period_end      TEXT    NOT NULL,
    result_json     TEXT    NOT NULL,   -- JSON-serialised PortfolioBacktestResult
    generated_at    TEXT    NOT NULL,
    UNIQUE(period_start, period_end)
);
"""

_WEIGHTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_weights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type     TEXT    NOT NULL UNIQUE,
    default_weight  REAL    NOT NULL,
    current_weight  REAL    NOT NULL,
    multiplier      REAL    NOT NULL DEFAULT 1.0,
    win_rate        REAL,
    profit_factor   REAL,
    sample_size     INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT    NOT NULL,
    backtest_period TEXT
);
"""


def _ensure_dir(db_path: str) -> None:
    """Create parent directory of db_path if it doesn't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _get_conn(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode for thread safety."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(db_path: str, schema: str) -> None:
    """Initialize a SQLite database with the given schema SQL."""
    _ensure_dir(db_path)
    with _get_conn(db_path) as conn:
        conn.executescript(schema)
        conn.commit()


# ═══════════════════════════════════════════════════════════
# HISTORICAL OHLCV CACHE
# ═══════════════════════════════════════════════════════════


class HistoricalDataCache:
    """SQLite-backed cache for Kite OHLCV historical data.

    Write once, read many. Once a bar is stored, it is never
    updated — historical price data does not change.

    Thread-safe: each call opens a fresh connection from the pool.
    """

    def __init__(self, db_path: str = HISTORICAL_DB_PATH) -> None:
        self._db_path = db_path
        _init_db(db_path, _OHLCV_SCHEMA)

    def store_bars(self, ticker: str, bars: list[OHLCVBar]) -> int:
        """Insert new OHLCV bars for a ticker.

        Uses INSERT OR IGNORE — existing rows are silently skipped.

        Args:
            ticker: NSE ticker symbol
            bars:   List of OHLCVBar to store

        Returns:
            Number of rows actually inserted (skipped rows not counted).
        """
        if not bars:
            return 0

        now_iso = datetime.now(IST).isoformat()
        rows = [
            (
                ticker.upper(),
                b.date.isoformat(),
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                now_iso,
            )
            for b in bars
        ]

        with _get_conn(self._db_path) as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO historical_ohlcv
                    (ticker, date, open, high, low, close, volume, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            inserted = cursor.rowcount

        logger.debug(
            f"[DataCache] {ticker}: stored {inserted}/{len(bars)} bars "
            f"({len(bars) - inserted} already cached)"
        )
        return inserted

    def get_bars(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
    ) -> list[OHLCVBar]:
        """Load OHLCV bars for a ticker in a date range.

        Returns:
            list[OHLCVBar] sorted oldest-to-newest.
            Empty list if no data found.
        """
        with _get_conn(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM historical_ohlcv
                WHERE ticker = ?
                  AND date >= ?
                  AND date <= ?
                ORDER BY date ASC
                """,
                (
                    ticker.upper(),
                    from_date.isoformat(),
                    to_date.isoformat(),
                ),
            ).fetchall()

        bars: list[OHLCVBar] = []
        for row in rows:
            try:
                bars.append(
                    OHLCVBar(
                        date=date.fromisoformat(row["date"]),
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                    )
                )
            except Exception as exc:
                logger.warning(
                    f"[DataCache] {ticker}: skipping malformed row "
                    f"{dict(row)} — {exc}"
                )

        return bars

    def get_date_range(self, ticker: str) -> Optional[tuple[date, date]]:
        """Return (earliest_date, latest_date) of cached bars for a ticker.

        Returns None if no bars cached.
        """
        with _get_conn(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT MIN(date) as min_d, MAX(date) as max_d
                FROM historical_ohlcv
                WHERE ticker = ?
                """,
                (ticker.upper(),),
            ).fetchone()

        if not row or not row["min_d"]:
            return None

        return (
            date.fromisoformat(row["min_d"]),
            date.fromisoformat(row["max_d"]),
        )

    def get_cached_tickers(self) -> list[str]:
        """Return all tickers that have at least one bar cached."""
        with _get_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM historical_ohlcv ORDER BY ticker"
            ).fetchall()
        return [row["ticker"] for row in rows]

    def count_bars(self, ticker: str) -> int:
        """Return total number of cached bars for a ticker."""
        with _get_conn(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM historical_ohlcv WHERE ticker = ?",
                (ticker.upper(),),
            ).fetchone()
        return row["cnt"] if row else 0

    def clear_ticker(self, ticker: str) -> int:
        """Delete all cached bars for a ticker. Returns rows deleted."""
        with _get_conn(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM historical_ohlcv WHERE ticker = ?",
                (ticker.upper(),),
            )
            conn.commit()
        logger.info(f"[DataCache] Cleared {cursor.rowcount} bars for {ticker}")
        return cursor.rowcount


# ═══════════════════════════════════════════════════════════
# BACKTEST RESULTS CACHE
# ═══════════════════════════════════════════════════════════


class BacktestResultCache:
    """SQLite-backed storage for BacktestResult and PortfolioBacktestResult.

    Results are serialised as JSON using Pydantic model_dump_json().
    UNIQUE constraint on (ticker, signal_type, period_start, period_end)
    — re-running a backtest overwrites the previous result for that key.
    """

    def __init__(self, db_path: str = BACKTEST_RESULTS_DB_PATH) -> None:
        self._db_path = db_path
        _init_db(db_path, _RESULTS_SCHEMA)

    def store_backtest_result(self, result: BacktestResult) -> None:
        """Upsert a BacktestResult. Overwrites if same key exists."""
        now_iso = datetime.now(IST).isoformat()
        result_json = result.model_dump_json(exclude_none=True)

        with _get_conn(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO backtest_results
                    (ticker, signal_type, period_start, period_end,
                     result_json, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, signal_type, period_start, period_end)
                DO UPDATE SET
                    result_json  = excluded.result_json,
                    generated_at = excluded.generated_at
                """,
                (
                    result.ticker.upper(),
                    result.signal_type.value,
                    result.period_start.isoformat(),
                    result.period_end.isoformat(),
                    result_json,
                    now_iso,
                ),
            )
            conn.commit()

        logger.debug(
            f"[DataCache] Stored BacktestResult: "
            f"{result.ticker}/{result.signal_type.value}"
        )

    def get_backtest_result(
        self,
        ticker: str,
        signal_type: str | SignalType,
        period_start: Optional[date] = None,
        period_end: Optional[date] = None,
    ) -> Optional[BacktestResult]:
        """Load the most recent BacktestResult for ticker + signal_type.

        If period_start/period_end provided, matches exactly.
        Otherwise returns the latest by generated_at.
        """
        signal_val = signal_type.value if isinstance(signal_type, SignalType) else signal_type

        if period_start and period_end:
            query = """
                SELECT result_json FROM backtest_results
                WHERE ticker = ? AND signal_type = ?
                  AND period_start = ? AND period_end = ?
                LIMIT 1
            """
            params = (
                ticker.upper(), signal_val,
                period_start.isoformat(), period_end.isoformat(),
            )
        else:
            query = """
                SELECT result_json FROM backtest_results
                WHERE ticker = ? AND signal_type = ?
                ORDER BY generated_at DESC
                LIMIT 1
            """
            params = (ticker.upper(), signal_val)

        with _get_conn(self._db_path) as conn:
            row = conn.execute(query, params).fetchone()

        if not row:
            return None

        try:
            return BacktestResult.model_validate_json(row["result_json"])
        except Exception as exc:
            logger.warning(
                f"[DataCache] Failed to deserialise BacktestResult "
                f"{ticker}/{signal_val}: {exc}"
            )
            return None

    def get_all_results_for_ticker(
        self, ticker: str
    ) -> list[BacktestResult]:
        """Return all BacktestResults for a ticker (all signal types)."""
        with _get_conn(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT result_json FROM backtest_results
                WHERE ticker = ?
                ORDER BY signal_type, generated_at DESC
                """,
                (ticker.upper(),),
            ).fetchall()

        results = []
        for row in rows:
            try:
                results.append(BacktestResult.model_validate_json(row["result_json"]))
            except Exception as exc:
                logger.warning(f"[DataCache] Skipping malformed result: {exc}")
        return results

    def store_portfolio_result(self, result: PortfolioBacktestResult) -> None:
        """Upsert a PortfolioBacktestResult. Overwrites if same period exists."""
        now_iso = datetime.now(IST).isoformat()
        result_json = result.model_dump_json(exclude_none=True)

        with _get_conn(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO portfolio_backtest_results
                    (period_start, period_end, result_json, generated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(period_start, period_end)
                DO UPDATE SET
                    result_json  = excluded.result_json,
                    generated_at = excluded.generated_at
                """,
                (
                    result.period_start.isoformat(),
                    result.period_end.isoformat(),
                    result_json,
                    now_iso,
                ),
            )
            conn.commit()

        logger.info(
            f"[DataCache] Stored PortfolioBacktestResult: "
            f"{result.period_start} → {result.period_end}, "
            f"verdict={result.advisor_verdict.value}"
        )

    def get_latest_portfolio_result(self) -> Optional[PortfolioBacktestResult]:
        """Return the most recently stored PortfolioBacktestResult."""
        with _get_conn(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT result_json FROM portfolio_backtest_results
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ).fetchone()

        if not row:
            return None

        try:
            return PortfolioBacktestResult.model_validate_json(row["result_json"])
        except Exception as exc:
            logger.warning(f"[DataCache] Failed to load portfolio result: {exc}")
            return None


# ═══════════════════════════════════════════════════════════
# SIGNAL WEIGHTS CACHE
# ═══════════════════════════════════════════════════════════


class SignalWeightCache:
    """SQLite-backed storage for evidence-based M4 signal weights.

    One row per signal_type. Updated by weight_updater.py after
    each backtest run. Read by M4 confidence_scorer at startup.
    """

    def __init__(self, db_path: str = SIGNAL_WEIGHTS_DB_PATH) -> None:
        self._db_path = db_path
        _init_db(db_path, _WEIGHTS_SCHEMA)

    def store_signal_weight(self, weight: SignalWeight) -> None:
        """Upsert a SignalWeight row."""
        with _get_conn(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO signal_weights
                    (signal_type, default_weight, current_weight, multiplier,
                     win_rate, profit_factor, sample_size, updated_at,
                     backtest_period)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signal_type)
                DO UPDATE SET
                    default_weight  = excluded.default_weight,
                    current_weight  = excluded.current_weight,
                    multiplier      = excluded.multiplier,
                    win_rate        = excluded.win_rate,
                    profit_factor   = excluded.profit_factor,
                    sample_size     = excluded.sample_size,
                    updated_at      = excluded.updated_at,
                    backtest_period = excluded.backtest_period
                """,
                (
                    weight.signal_type.value,
                    weight.default_weight,
                    weight.current_weight,
                    weight.multiplier,
                    weight.win_rate,
                    weight.profit_factor,
                    weight.sample_size,
                    weight.updated_at.isoformat(),
                    weight.backtest_period,
                ),
            )
            conn.commit()

        logger.info(
            f"[DataCache] Signal weight updated: {weight.signal_type.value} "
            f"default={weight.default_weight:.1f} → "
            f"current={weight.current_weight:.1f} "
            f"(×{weight.multiplier:.2f})"
        )

    def get_signal_weight(self, signal_type: str | SignalType) -> Optional[SignalWeight]:
        """Return the current SignalWeight for a signal type, or None."""
        signal_val = signal_type.value if isinstance(signal_type, SignalType) else signal_type

        with _get_conn(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM signal_weights WHERE signal_type = ?",
                (signal_val,),
            ).fetchone()

        if not row:
            return None

        return _row_to_signal_weight(row)

    def get_all_signal_weights(self) -> list[SignalWeight]:
        """Return all signal weights, sorted by signal_type."""
        with _get_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM signal_weights ORDER BY signal_type"
            ).fetchall()

        return [_row_to_signal_weight(r) for r in rows]

    def get_weight_dict(self) -> dict[str, float]:
        """Return {signal_type_value → current_weight} for all stored weights.

        Used by M4 confidence_scorer to load backtest-adjusted weights.
        Returns empty dict if no weights have been stored yet.
        """
        weights = self.get_all_signal_weights()
        return {w.signal_type.value: w.current_weight for w in weights}


# ─────────────────────────────────────────────────────────────
# Row deserialisation helper
# ─────────────────────────────────────────────────────────────

def _row_to_signal_weight(row: sqlite3.Row) -> SignalWeight:
    """Convert a SQLite row to a SignalWeight model."""
    updated_at = datetime.fromisoformat(row["updated_at"])
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=IST)

    return SignalWeight(
        signal_type=SignalType(row["signal_type"]),
        default_weight=row["default_weight"],
        current_weight=row["current_weight"],
        multiplier=row["multiplier"],
        win_rate=row["win_rate"],
        profit_factor=row["profit_factor"],
        sample_size=row["sample_size"],
        updated_at=updated_at,
        backtest_period=row["backtest_period"],
    )


# ═══════════════════════════════════════════════════════════
# Module-level singletons
# ═══════════════════════════════════════════════════════════

historical_cache = HistoricalDataCache()
backtest_result_cache = BacktestResultCache()
signal_weight_cache = SignalWeightCache()
