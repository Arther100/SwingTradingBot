"""
SwingAdvisorBot — Module 6: Daily Reports + Alerts
alerts/alert_tracker.py — SQLite dedup tracker for sent alerts

Prevents duplicate alerts: max one alert per (ticker, alert_type, date).

Storage: SQLite file alongside M5's database — lightweight, no extra deps.

Usage:
    from module6_reports.alerts.alert_tracker import alert_tracker

    if not alert_tracker.has_alert("HDFCBANK", "entry_zone", "2026-05-15"):
        # send alert...
        alert_tracker.record_alert("HDFCBANK", "entry_zone", "2026-05-15", msg_id=42)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from typing import Optional

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.alert_tracker")

# Database path — same directory as M5 database
_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "module5_memory",
    "data",
)
_DB_PATH = os.path.join(_DB_DIR, "alert_tracker.db")


class AlertTracker:
    """SQLite-based alert dedup tracker.

    Schema:
        alerts(
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            date TEXT NOT NULL,
            sent_at TEXT NOT NULL,
            telegram_message_id INTEGER,
            UNIQUE(ticker, alert_type, date)
        )

    Thread-safe: uses check_same_thread=False.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the database and table if they don't exist."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                date TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                telegram_message_id INTEGER,
                UNIQUE(ticker, alert_type, date)
            )
        """)
        self._conn.commit()
        logger.debug(f"[AlertTracker] DB ready: {self._db_path}")

    def has_alert(
        self,
        ticker: str,
        alert_type: str,
        date: str,
    ) -> bool:
        """Check if an alert was already sent for this ticker+type+date.

        Args:
            ticker: NSE ticker symbol.
            alert_type: Alert type string (e.g. "entry_zone").
            date: Date string YYYY-MM-DD.

        Returns:
            True if alert exists (already sent today).
        """
        if not self._conn:
            self._ensure_db()

        cursor = self._conn.execute(
            "SELECT 1 FROM alerts WHERE ticker = ? AND alert_type = ? AND date = ?",
            (ticker, alert_type, date),
        )
        return cursor.fetchone() is not None

    def record_alert(
        self,
        ticker: str,
        alert_type: str,
        date: str,
        telegram_message_id: Optional[int] = None,
    ) -> None:
        """Record that an alert was sent.

        Uses INSERT OR IGNORE to handle race conditions gracefully.

        Args:
            ticker: NSE ticker symbol.
            alert_type: Alert type string.
            date: Date string YYYY-MM-DD.
            telegram_message_id: Telegram message_id for reference.
        """
        if not self._conn:
            self._ensure_db()

        sent_at = datetime.now(IST).isoformat()

        self._conn.execute(
            """INSERT OR IGNORE INTO alerts
               (ticker, alert_type, date, sent_at, telegram_message_id)
               VALUES (?, ?, ?, ?, ?)""",
            (ticker, alert_type, date, sent_at, telegram_message_id),
        )
        self._conn.commit()
        logger.debug(
            f"[AlertTracker] Recorded: {ticker}/{alert_type}/{date} "
            f"msg_id={telegram_message_id}"
        )

    def get_today_alerts(self) -> list[dict]:
        """Get all alerts sent today.

        Returns list of dicts with ticker, alert_type, sent_at, telegram_message_id.
        """
        if not self._conn:
            self._ensure_db()

        today = datetime.now(IST).strftime("%Y-%m-%d")
        cursor = self._conn.execute(
            "SELECT ticker, alert_type, sent_at, telegram_message_id "
            "FROM alerts WHERE date = ? ORDER BY sent_at",
            (today,),
        )
        return [
            {
                "ticker": row[0],
                "alert_type": row[1],
                "sent_at": row[2],
                "telegram_message_id": row[3],
            }
            for row in cursor.fetchall()
        ]

    def get_alert_count(self, date: str | None = None) -> int:
        """Get number of alerts for a given date (default: today)."""
        if not self._conn:
            self._ensure_db()

        date = date or datetime.now(IST).strftime("%Y-%m-%d")
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE date = ?",
            (date,),
        )
        return cursor.fetchone()[0]

    def cleanup_old_alerts(self, days_to_keep: int = 30) -> int:
        """Delete alerts older than N days.

        Returns number of rows deleted.
        """
        if not self._conn:
            self._ensure_db()

        from datetime import timedelta

        cutoff = (datetime.now(IST) - timedelta(days=days_to_keep)).strftime(
            "%Y-%m-%d"
        )

        cursor = self._conn.execute(
            "DELETE FROM alerts WHERE date < ?",
            (cutoff,),
        )
        self._conn.commit()

        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"[AlertTracker] Cleaned up {deleted} old alerts")
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

alert_tracker = AlertTracker()
