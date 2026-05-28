"""
SwingAdvisorBot — Module 1: Data Layer
fetchers/fii_dii_fetcher.py — NSE FII/DII institutional flow fetcher

FII (Foreign Institutional Investors) and DII (Domestic Institutional
Investors) data is the single most important macro signal for Indian
swing traders. Before analysing any stock chart, a professional trader
checks whether smart money is flowing into or out of the market.

FII/DII thresholds (daily net in ₹ crore):
  FII net > +2,000  → strong institutional conviction to buy
  FII net +500-2000 → mild institutional interest
  FII net 0 to -500 → cautious, watching risk
  FII net < -2,000  → distribution, reduce longs
  FII net < -5,000  → high-conviction selling, avoid new trades

Combined signal (FII net + DII net):
  combined > +3,000  → strong_bullish
  combined +1,000-3,000 → bullish
  combined 0-+1,000  → mild_bullish
  combined -1,000-0  → mild_bearish
  combined < -1,000  → bearish
  combined < -3,000  → strong_bearish

Data source:
  NSE India public API — https://www.nseindia.com/api/fiidiiTradeReact
  Requires browser-like session cookies via NseSessionManager.
  Updates once daily after 4:00 PM IST (end-of-day settlement data).

Caching:
  TTL: 14,400 seconds (4 hours) — data updates only once daily.
  Cache key: "fii_dii:{YYYY-MM-DD}" — one entry per trading day.

Consecutive-day tracking:
  History stored in module1_data_layer/data/fii_dii_history.json.
  Tracks last 30 trading days of FII/DII net values.
  Used to compute consecutive buying/selling streaks.

Models defined here (re-exported by models.py in File 04):
  FiiDiiSignal — 6-level enum for combined institutional flow signal
  FiiDiiData   — Complete FII/DII snapshot with advisor-quality fields
"""

from __future__ import annotations

import enum
import json
import logging
import os
import gzip
import json as _json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from module1_data_layer.cache import cache
from module1_data_layer.fetchers.nse_session_manager import nse_session

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.fetchers.fii_dii")

# NSE FII/DII data endpoint
NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

# Cache TTL: 4 hours (data is published once daily after 4 PM)
CACHE_TTL_FII_DII: int = 14_400

# History file for consecutive-day streak tracking (last 30 days)
_DATA_DIR = Path(__file__).parent.parent / "data"
_HISTORY_FILE = _DATA_DIR / "fii_dii_history.json"
_MAX_HISTORY_DAYS: int = 30

# Request timeout
_TIMEOUT_SEC: float = 15.0


# ─────────────────────────────────────────────────────────────
# Enums and Models
# These are defined here and re-exported by models.py (File 04).
# Import path after File 04: from module1_data_layer.models import FiiDiiData
# ─────────────────────────────────────────────────────────────


class FiiDiiSignal(str, enum.Enum):
    """Combined FII + DII institutional flow signal.

    Based on combined net (FII net + DII net) in ₹ crore:
      > +3,000   → strong_bullish
      +1,000-3,000 → bullish
      0-+1,000   → mild_bullish
      -1,000-0   → mild_bearish
      -3,000-1,000 → bearish
      < -3,000   → strong_bearish
    """

    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    MILD_BULLISH = "mild_bullish"
    NEUTRAL = "neutral"
    MILD_BEARISH = "mild_bearish"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class FiiDiiData(BaseModel):
    """Complete FII/DII institutional flow snapshot for one trading day.

    Carries all the context needed for M2 (analysis), M4 (setups),
    and M6 (morning brief) to act on institutional flow intelligence.

    FII/DII data is optional in the pipeline — if NSE is unreachable,
    the bot continues without it (logged as degraded, not failed).
    """

    date: str = Field(..., description="Trading date YYYY-MM-DD (IST)")

    # ── FII (Foreign Institutional Investors) ──
    fii_buy: float = Field(..., description="FII gross buy value in ₹ crore")
    fii_sell: float = Field(..., description="FII gross sell value in ₹ crore")
    fii_net: float = Field(
        ..., description="FII net (buy - sell) in ₹ crore. Positive = buying."
    )

    # ── DII (Domestic Institutional Investors) ──
    dii_buy: float = Field(..., description="DII gross buy value in ₹ crore")
    dii_sell: float = Field(..., description="DII gross sell value in ₹ crore")
    dii_net: float = Field(
        ..., description="DII net (buy - sell) in ₹ crore. Positive = buying."
    )

    # ── Combined signal ──
    combined_net: float = Field(
        ..., description="FII net + DII net in ₹ crore"
    )
    fii_signal: FiiDiiSignal = Field(
        ..., description="Signal based on FII net alone"
    )
    dii_signal: FiiDiiSignal = Field(
        ..., description="Signal based on DII net alone"
    )
    combined_signal: FiiDiiSignal = Field(
        ..., description="Primary signal — based on FII + DII combined net"
    )

    # ── Trend context ──
    consecutive_fii_buying_days: int = Field(
        default=1,
        description="Number of consecutive days FII has been net buying. "
                    "Negative = consecutive selling days.",
    )

    # ── Advisor interpretation ──
    advisor_note: str = Field(
        ...,
        description="2-sentence advisor-quality interpretation of today's flow. "
                    "References actual ₹ amounts and gives actionable guidance.",
    )
    market_impact: str = Field(
        default="medium",
        description="Expected impact on market: high / medium / low",
    )

    # ── Metadata ──
    fetched_at: str = Field(
        default_factory=lambda: datetime.now(IST).isoformat(),
        description="IST timestamp when this data was fetched",
    )
    is_real_data: bool = Field(
        default=True,
        description="False if NSE was unreachable and this is a cached/fallback value",
    )


# ─────────────────────────────────────────────────────────────
# Fetcher
# ─────────────────────────────────────────────────────────────


class FiiDiiFetcher:
    """Fetches and processes FII/DII institutional flow data from NSE.

    Pipeline:
      1. Check cache (TTL 4 hours) → return if fresh
      2. Obtain NSE session cookies via NseSessionManager
      3. GET https://www.nseindia.com/api/fiidiiTradeReact
      4. Parse response → extract FII/DII buy/sell/net
      5. Calculate per-entity and combined signals
      6. Load history → compute consecutive-day streak
      7. Generate advisor note (template-based)
      8. Store in cache + update history file
      9. Return FiiDiiData

    On any failure: return None (pipeline marks M1 as degraded, not failed).
    """

    async def fetch(self) -> Optional[FiiDiiData]:
        """Fetch today's FII/DII data from NSE, or return cached value.

        Returns:
            FiiDiiData if successful, None if NSE is unreachable.
        """
        today = datetime.now(IST).strftime("%Y-%m-%d")
        cache_key = f"fii_dii:{today}"

        # ── Step 1: Cache hit ──────────────────────────────────
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug(f"[FII/DII] Cache hit for {today}")
            return cached

        # ── Step 2–4: Fetch from NSE ───────────────────────────
        raw = await self._call_nse_api()
        if raw is None:
            logger.warning("[FII/DII] NSE API unreachable — no FII/DII data today.")
            return None

        parsed = self._parse_response(raw, today)
        if parsed is None:
            logger.warning("[FII/DII] Could not parse NSE response.")
            return None

        fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net = parsed

        # ── Step 5: Signals ────────────────────────────────────
        combined_net = fii_net + dii_net
        fii_signal = _net_to_signal(fii_net)
        dii_signal = _net_to_signal(dii_net)
        combined_signal = _combined_net_to_signal(combined_net)

        # ── Step 6: Consecutive days ───────────────────────────
        history = _load_history()
        consecutive = _compute_consecutive_days(history, fii_net)

        # ── Step 7: Advisor note ───────────────────────────────
        advisor_note = _build_advisor_note(
            fii_net=fii_net,
            dii_net=dii_net,
            combined_signal=combined_signal,
            consecutive=consecutive,
        )
        market_impact = _assess_market_impact(abs(combined_net))

        # ── Build result ───────────────────────────────────────
        result = FiiDiiData(
            date=today,
            fii_buy=round(fii_buy, 2),
            fii_sell=round(fii_sell, 2),
            fii_net=round(fii_net, 2),
            dii_buy=round(dii_buy, 2),
            dii_sell=round(dii_sell, 2),
            dii_net=round(dii_net, 2),
            combined_net=round(combined_net, 2),
            fii_signal=fii_signal,
            dii_signal=dii_signal,
            combined_signal=combined_signal,
            consecutive_fii_buying_days=consecutive,
            advisor_note=advisor_note,
            market_impact=market_impact,
        )

        # ── Step 8: Cache + persist history ───────────────────
        cache.set(cache_key, result, CACHE_TTL_FII_DII)
        _save_history(history, today, fii_net, dii_net)

        logger.info(
            f"[FII/DII] {today} — FII net: ₹{fii_net:+,.0f} cr | "
            f"DII net: ₹{dii_net:+,.0f} cr | "
            f"Combined: {combined_signal.value} | "
            f"Streak: {consecutive:+d} days"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    async def _call_nse_api(self) -> Optional[list[dict]]:
        """GET the NSE FII/DII endpoint with session cookies.

        Retries once after cookie invalidation if NSE returns 401/403.

        Returns:
            Parsed JSON list on success, None on any failure.
        """
        for attempt in range(2):
            headers, cookies = await nse_session.get_session_context()
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=_TIMEOUT_SEC,
                ) as client:
                    resp = await client.get(
                        NSE_FII_DII_URL,
                        headers=headers,
                        cookies=cookies,
                    )

                    if resp.status_code in (401, 403):
                        logger.warning(
                            f"[FII/DII] NSE returned {resp.status_code} "
                            f"(attempt {attempt + 1}) — invalidating session."
                        )
                        nse_session.invalidate()
                        continue  # Retry with fresh cookies

                    resp.raise_for_status()

                    # NSE sends gzip-compressed JSON when we pass Accept-Encoding
                    # manually. httpx only auto-decompresses when it controls that
                    # header itself, so we must decompress here if needed.
                    try:
                        data = resp.json()
                    except (ValueError, UnicodeDecodeError):
                        try:
                            raw = gzip.decompress(resp.content)
                            data = _json.loads(raw)
                        except Exception as decomp_err:
                            logger.warning(
                                f"[FII/DII] Failed to decode NSE response: {decomp_err}"
                            )
                            return None

                    if isinstance(data, list) and data:
                        return data

                    logger.warning(
                        f"[FII/DII] Unexpected NSE response format: "
                        f"{type(data).__name__}"
                    )
                    return None

            except httpx.TimeoutException:
                logger.warning(
                    f"[FII/DII] NSE API timed out after {_TIMEOUT_SEC}s "
                    f"(attempt {attempt + 1})."
                )
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    f"[FII/DII] NSE HTTP error {exc.response.status_code}"
                )
                return None
            except Exception as exc:
                logger.warning(f"[FII/DII] Unexpected error: {exc}")
                return None

        return None

    def _parse_response(
        self, data: list[dict], today: str
    ) -> Optional[tuple[float, float, float, float, float, float]]:
        """Parse NSE FII/DII JSON response into (fii_buy, fii_sell, fii_net,
        dii_buy, dii_sell, dii_net).

        NSE response is a list of category objects. We look for the
        'FII/FPI' and 'DII' categories.

        NSE sometimes returns values as strings with commas (e.g. "15,234.56")
        or as plain floats. Both are handled.

        Returns:
            6-tuple of floats on success, None if required categories missing.
        """
        fii: Optional[dict] = None
        dii: Optional[dict] = None

        for item in data:
            category = str(item.get("category", "")).upper()
            if "FII" in category or "FPI" in category:
                fii = item
            elif "DII" in category:
                dii = item

        if fii is None or dii is None:
            logger.warning(
                f"[FII/DII] Could not find FII/DII categories in response. "
                f"Categories found: {[d.get('category') for d in data]}"
            )
            return None

        try:
            fii_buy = _parse_crore(fii.get("buyValue") or fii.get("buy_value", 0))
            fii_sell = _parse_crore(fii.get("sellValue") or fii.get("sell_value", 0))
            fii_net = _parse_crore(fii.get("netValue") or fii.get("net_value", 0))

            dii_buy = _parse_crore(dii.get("buyValue") or dii.get("buy_value", 0))
            dii_sell = _parse_crore(dii.get("sellValue") or dii.get("sell_value", 0))
            dii_net = _parse_crore(dii.get("netValue") or dii.get("net_value", 0))

            # Verify net is consistent (buy - sell ≈ net); recalculate if not
            if abs((fii_buy - fii_sell) - fii_net) > 1.0:
                fii_net = fii_buy - fii_sell
            if abs((dii_buy - dii_sell) - dii_net) > 1.0:
                dii_net = dii_buy - dii_sell

            return fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net

        except (ValueError, TypeError) as exc:
            logger.warning(f"[FII/DII] Value parsing error: {exc}")
            return None


# ─────────────────────────────────────────────────────────────
# Signal calculators — pure functions, easy to test
# ─────────────────────────────────────────────────────────────


def _parse_crore(value: object) -> float:
    """Parse a crore value that may be a float, int, or comma-formatted string.

    Examples:
        "15,234.56" → 15234.56
        15234.56    → 15234.56
        "-2,100.00" → -2100.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(",", "").strip()
    return float(cleaned) if cleaned else 0.0


def _net_to_signal(net: float) -> FiiDiiSignal:
    """Map a single entity's net value (₹ crore) to a FiiDiiSignal.

    Thresholds (calibrated for typical FII/DII daily flow ranges):
      > +2,000  → strong_bullish
      +500-2000 → bullish
      0-+500    → mild_bullish
      -500-0    → mild_bearish
      -2000-500 → bearish
      < -2,000  → strong_bearish
    """
    if net > 2_000:
        return FiiDiiSignal.STRONG_BULLISH
    elif net > 500:
        return FiiDiiSignal.BULLISH
    elif net > 0:
        return FiiDiiSignal.MILD_BULLISH
    elif net > -500:
        return FiiDiiSignal.MILD_BEARISH
    elif net > -2_000:
        return FiiDiiSignal.BEARISH
    else:
        return FiiDiiSignal.STRONG_BEARISH


def _combined_net_to_signal(combined_net: float) -> FiiDiiSignal:
    """Map the combined FII + DII net to a FiiDiiSignal.

    Combined thresholds (higher bar since two entities contribute):
      > +3,000  → strong_bullish
      +1,000-3,000 → bullish
      0-+1,000  → mild_bullish
      -1,000-0  → mild_bearish
      -3,000-1,000 → bearish
      < -3,000  → strong_bearish
    """
    if combined_net > 3_000:
        return FiiDiiSignal.STRONG_BULLISH
    elif combined_net > 1_000:
        return FiiDiiSignal.BULLISH
    elif combined_net > 0:
        return FiiDiiSignal.MILD_BULLISH
    elif combined_net > -1_000:
        return FiiDiiSignal.MILD_BEARISH
    elif combined_net > -3_000:
        return FiiDiiSignal.BEARISH
    else:
        return FiiDiiSignal.STRONG_BEARISH


def _assess_market_impact(abs_combined: float) -> str:
    """Return market impact string based on absolute combined net flow."""
    if abs_combined > 3_000:
        return "high"
    elif abs_combined > 1_000:
        return "medium"
    else:
        return "low"


# ─────────────────────────────────────────────────────────────
# Consecutive-day streak helpers
# ─────────────────────────────────────────────────────────────


def _load_history() -> dict[str, dict[str, float]]:
    """Load FII/DII history from JSON file.

    Returns dict: {"YYYY-MM-DD": {"fii_net": float, "dii_net": float}, ...}
    Returns empty dict if file doesn't exist or is corrupt.
    """
    try:
        if _HISTORY_FILE.exists():
            return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"[FII/DII] Could not load history: {exc}")
    return {}


def _save_history(
    history: dict[str, dict[str, float]],
    today: str,
    fii_net: float,
    dii_net: float,
) -> None:
    """Append today's entry to history file, pruning to last 30 days."""
    history[today] = {"fii_net": round(fii_net, 2), "dii_net": round(dii_net, 2)}

    # Prune to last N days
    if len(history) > _MAX_HISTORY_DAYS:
        sorted_keys = sorted(history.keys(), reverse=True)
        history = {k: history[k] for k in sorted_keys[:_MAX_HISTORY_DAYS]}

    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _HISTORY_FILE.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning(f"[FII/DII] Could not save history: {exc}")


def _compute_consecutive_days(
    history: dict[str, dict[str, float]], today_fii_net: float
) -> int:
    """Compute consecutive FII buying or selling days including today.

    Returns:
        Positive integer = consecutive buying days (FII net > 0).
        Negative integer = consecutive selling days (FII net < 0).
        Example: 3 = 3 consecutive buying days, -2 = 2 consecutive selling.
    """
    if not history:
        return 1 if today_fii_net >= 0 else -1

    # Sort past days descending (most recent first), exclude today
    today = datetime.now(IST).strftime("%Y-%m-%d")
    past_days = sorted(
        [(k, v["fii_net"]) for k, v in history.items() if k < today],
        reverse=True,
    )

    today_buying = today_fii_net >= 0
    streak = 1 if today_buying else -1

    for _, past_fii_net in past_days:
        past_buying = past_fii_net >= 0
        if past_buying == today_buying:
            streak = streak + 1 if today_buying else streak - 1
        else:
            break

    return streak


# ─────────────────────────────────────────────────────────────
# Advisor note builder — template-based (M2 will add deeper analysis)
# ─────────────────────────────────────────────────────────────


def _build_advisor_note(
    fii_net: float,
    dii_net: float,
    combined_signal: FiiDiiSignal,
    consecutive: int,
) -> str:
    """Generate a 2-sentence advisor-quality note about today's FII/DII flow.

    Template-based (no Claude call — M1 is a data layer, not analysis).
    M2's MarketAnalysisAgent enriches this further in the morning brief.
    """
    fii_dir = "buying" if fii_net >= 0 else "selling"
    dii_dir = "buying" if dii_net >= 0 else "selling"
    streak_abs = abs(consecutive)
    streak_dir = "buying" if consecutive > 0 else "selling"

    # Sentence 1: What happened
    if combined_signal in (FiiDiiSignal.STRONG_BULLISH, FiiDiiSignal.BULLISH):
        sentence1 = (
            f"Institutional money is flowing strongly into the market — "
            f"FIIs net {fii_dir} ₹{abs(fii_net):,.0f} cr and "
            f"DIIs net {dii_dir} ₹{abs(dii_net):,.0f} cr today."
        )
    elif combined_signal == FiiDiiSignal.MILD_BULLISH:
        sentence1 = (
            f"Mild institutional support today — "
            f"FIIs net {fii_dir} ₹{abs(fii_net):,.0f} cr, "
            f"DIIs net {dii_dir} ₹{abs(dii_net):,.0f} cr."
        )
    elif combined_signal == FiiDiiSignal.MILD_BEARISH:
        sentence1 = (
            f"Slight institutional outflow today — "
            f"FIIs net {fii_dir} ₹{abs(fii_net):,.0f} cr, "
            f"DIIs net {dii_dir} ₹{abs(dii_net):,.0f} cr."
        )
    elif combined_signal in (FiiDiiSignal.BEARISH, FiiDiiSignal.STRONG_BEARISH):
        sentence1 = (
            f"Significant institutional selling — "
            f"FIIs net {fii_dir} ₹{abs(fii_net):,.0f} cr, "
            f"DIIs net {dii_dir} ₹{abs(dii_net):,.0f} cr today."
        )
    else:
        sentence1 = (
            f"Mixed institutional activity — "
            f"FIIs ₹{abs(fii_net):,.0f} cr, DIIs ₹{abs(dii_net):,.0f} cr."
        )

    # Sentence 2: Streak context + guidance
    if streak_abs >= 3:
        sentence2 = (
            f"This marks {streak_abs} consecutive days of FII {streak_dir} — "
            f"{'conviction is building, favour long setups.' if streak_dir == 'buying' else 'sustained distribution, avoid new longs until flow reverses.'}"
        )
    elif streak_abs == 2:
        sentence2 = (
            f"FIIs have been {streak_dir} for 2 consecutive days — "
            f"{'early trend, monitor for continuation.' if streak_dir == 'buying' else 'watch for support levels before entering.'}"
        )
    else:
        if combined_signal in (FiiDiiSignal.STRONG_BULLISH, FiiDiiSignal.BULLISH):
            sentence2 = "Favourable environment for swing long setups today."
        elif combined_signal in (FiiDiiSignal.BEARISH, FiiDiiSignal.STRONG_BEARISH):
            sentence2 = "Caution advised — avoid new long positions until institutional flow improves."
        else:
            sentence2 = "Market environment is mixed — apply selective stock-specific filters."

    return f"{sentence1} {sentence2}"


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

fii_dii_fetcher: FiiDiiFetcher = FiiDiiFetcher()
"""Shared FII/DII fetcher singleton.

Usage:
    from module1_data_layer.fetchers.fii_dii_fetcher import fii_dii_fetcher

    data = await fii_dii_fetcher.fetch()
    if data:
        print(data.combined_signal, data.advisor_note)
"""
