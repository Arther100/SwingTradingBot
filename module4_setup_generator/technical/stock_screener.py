"""
SwingAdvisorBot — Module 4: Trade Setup Generator
technical/stock_screener.py — Filter stocks by advisor signals + earnings risk

The screener takes M1 MarketData and selects candidate stocks
for setup generation. Only stocks with actionable signals pass.

Selection logic:
  1. Include stocks with priority flags (breakout, momentum, etc.)
  2. Exclude stocks with skip flags (selling_pressure, neutral, etc.)
  3. Block stocks with HIGH earnings risk (results in ≤ 2 days)
  4. Warn on MEDIUM earnings risk (results in 3-5 days)
  5. Sort by flag priority (breakout > unusual > momentum > accumulation)
  6. Limit to max_candidates (default 10)

FII/DII signal modifier (applied to confidence scores in ConfidenceScorer):
  combined_signal = "strong_bullish"  → +0.5 confidence bonus
  combined_signal = "strong_bearish"  → -1.5 confidence penalty
  Other signals                        → no adjustment

This is a FAST filter — no API calls, no heavy computation.
Runs in < 5ms on 50 stocks.
"""

from __future__ import annotations

import logging
from typing import Optional

from module1_data_layer.models import (
    EarningsEvent,
    EarningsRisk,
    EarningsRiskLevel,
    FiiDiiData,
    FiiDiiSignal,
    MarketData,
    StockData,
)
from module4_setup_generator.config import FLAG_PRIORITY, SKIP_FLAGS

logger = logging.getLogger("swing_advisor.stock_screener")

# FII/DII signal → confidence score adjustment applied by ConfidenceScorer
FII_DII_CONFIDENCE_ADJUSTMENTS: dict[str, float] = {
    FiiDiiSignal.STRONG_BULLISH.value: +0.5,
    FiiDiiSignal.STRONG_BEARISH.value: -1.5,
}


class StockScreener:
    """Filter M1 stocks for setup candidacy.

    Usage:
        screener = StockScreener()
        candidates = screener.screen(
            market_data=market_data,
            max_candidates=10,
        )
        # Returns list of StockData sorted by signal priority

        # With earnings risk blocking:
        candidates = screener.screen(
            market_data=market_data,
            max_candidates=10,
            earnings_calendar=earnings_calendar,   # dict[ticker → EarningsEvent]
        )

        # Get earnings risk map for setup cards:
        risk_map = screener.build_earnings_risk_map(candidates, earnings_calendar)

        # Get FII/DII confidence adjustment:
        adj = screener.get_fii_dii_adjustment(market_data.fii_dii)
    """

    def screen(
        self,
        market_data: MarketData,
        max_candidates: int = 10,
        specific_tickers: Optional[list[str]] = None,
        earnings_calendar: Optional[dict[str, EarningsEvent]] = None,
    ) -> list[StockData]:
        """Screen stocks for setup generation.

        Args:
            market_data: M1 MarketData with stocks list.
            max_candidates: Maximum stocks to return.
            specific_tickers: If provided, only evaluate these tickers.
            earnings_calendar: Dict of ticker → EarningsEvent from M1.
                               Stocks with HIGH earnings risk are blocked.
                               Pass None to skip earnings risk check (backward-compatible).

        Returns:
            List of StockData sorted by flag priority, limited to max_candidates.
            Stocks with HIGH earnings risk (≤2 days) are excluded.
        """
        stocks = market_data.stocks

        if not stocks:
            logger.warning("[StockScreener] No stocks in MarketData")
            return []

        # Filter to specific tickers if requested
        if specific_tickers:
            tickers_upper = {t.upper() for t in specific_tickers}
            stocks = [s for s in stocks if s.ticker.upper() in tickers_upper]
            logger.info(
                f"[StockScreener] Filtered to {len(stocks)} specific tickers"
            )

        # Build earnings lookup for O(1) access
        _earnings: dict[str, EarningsEvent] = earnings_calendar or {}

        # Separate into qualifying and skipped
        candidates = []
        skipped = []
        earnings_blocked = []

        for stock in stocks:
            flag = self._get_flag_value(stock)

            if flag in SKIP_FLAGS:
                skipped.append((stock.ticker, flag))
                continue

            if flag in FLAG_PRIORITY:
                # ── Earnings risk check ──────────────────────────────
                earnings_block = self._check_earnings_block(
                    stock.ticker, _earnings
                )
                if earnings_block:
                    earnings_blocked.append((stock.ticker, earnings_block))
                    logger.info(
                        f"[StockScreener] BLOCKED {stock.ticker} — "
                        f"earnings in {_earnings[stock.ticker.upper()].days_to_result} days "
                        f"(HIGH risk)"
                    )
                    continue
                candidates.append(stock)
            elif flag is None:
                skipped.append((stock.ticker, "no_flag"))
            else:
                # Unknown flag — include with low priority
                earnings_block = self._check_earnings_block(
                    stock.ticker, _earnings
                )
                if not earnings_block:
                    candidates.append(stock)

        # Sort by flag priority (lower number = higher priority)
        candidates.sort(key=lambda s: self._get_priority(s))

        # Limit to max_candidates
        result = candidates[:max_candidates]

        logger.info(
            f"[StockScreener] Evaluated {len(stocks)} stocks. "
            f"Candidates: {len(result)}, "
            f"Skipped: {len(skipped)}, "
            f"Earnings-blocked: {len(earnings_blocked)} "
            f"({', '.join(t for t, _ in earnings_blocked[:3])})"
        )

        # Log MEDIUM earnings warnings for candidates that made it through
        for stock in result:
            event = _earnings.get(stock.ticker.upper())
            if event and event.risk_level == EarningsRiskLevel.MEDIUM:
                logger.warning(
                    f"[StockScreener] EARNINGS WARN: {stock.ticker} — "
                    f"results in {event.days_to_result} days. "
                    f"Confidence reduced in scorer."
                )

        return result

    def build_earnings_risk_map(
        self,
        candidates: list[StockData],
        earnings_calendar: Optional[dict[str, EarningsEvent]] = None,
    ) -> dict[str, EarningsRisk]:
        """Build a ticker → EarningsRisk map for a list of screened candidates.

        Called by TradeSetupAgent after screening to attach EarningsRisk
        to each TradeSetup card. Only candidates that passed screening
        (i.e., no HIGH risk) are included — returns MEDIUM/LOW/NONE.

        Args:
            candidates: Stocks that passed the screener.
            earnings_calendar: Dict of ticker → EarningsEvent.

        Returns:
            Dict mapping ticker → EarningsRisk (compact 4-field model).
        """
        from module1_data_layer.fetchers.earnings_fetcher import earnings_fetcher

        _earnings = earnings_calendar or {}
        risk_map: dict[str, EarningsRisk] = {}

        for stock in candidates:
            event = _earnings.get(stock.ticker.upper())
            risk_map[stock.ticker] = earnings_fetcher.build_earnings_risk(event)

        return risk_map

    def get_fii_dii_adjustment(
        self, fii_dii: Optional[FiiDiiData]
    ) -> float:
        """Return the confidence score adjustment from FII/DII signal.

        Applied by ConfidenceScorer to every setup in the current run.

        Rules:
          strong_bullish  → +0.5  (institutional conviction supports longs)
          strong_bearish  → -1.5  (heavy selling — all setups penalised)
          All other signals → 0.0  (no adjustment)

        Args:
            fii_dii: FiiDiiData from MarketData, or None if unavailable.

        Returns:
            Float confidence adjustment (positive = bonus, negative = penalty).
        """
        if fii_dii is None:
            return 0.0
        return FII_DII_CONFIDENCE_ADJUSTMENTS.get(
            fii_dii.combined_signal.value, 0.0
        )

    def _get_flag_value(self, stock: StockData) -> Optional[str]:
        """Get the string value of a stock's advisor flag."""
        if stock.advisor_flag is None:
            return None
        if hasattr(stock.advisor_flag, "value"):
            return stock.advisor_flag.value.upper()
        return str(stock.advisor_flag).upper()

    def _get_priority(self, stock: StockData) -> int:
        """Get sort priority for a stock (lower = better)."""
        flag = self._get_flag_value(stock)
        if flag is None:
            return 99
        return FLAG_PRIORITY.get(flag, 50)

    def _check_earnings_block(
        self,
        ticker: str,
        earnings_calendar: Optional[dict[str, EarningsEvent]],
    ) -> bool:
        """Return True if the ticker should be BLOCKED due to HIGH earnings risk.

        HIGH risk = results in ≤ 2 days. No swing trade should be entered.

        Args:
            ticker: NSE ticker to check.
            earnings_calendar: Dict of ticker → EarningsEvent, or None.

        Returns:
            True if setup should be blocked entirely.
        """
        if not earnings_calendar:
            return False
        event = earnings_calendar.get(ticker.upper())
        if event is None:
            return False
        return event.risk_level == EarningsRiskLevel.HIGH

    def get_skip_reason(self, stock: StockData) -> Optional[str]:
        """Get the reason a stock would be skipped by signal filter.

        Returns None if the stock qualifies on signal alone.
        Note: earnings-based blocks are handled separately in screen().
        """
        flag = self._get_flag_value(stock)

        if flag is None:
            return "No advisor flag assigned"
        if flag in SKIP_FLAGS:
            flag_lower = flag.lower().replace("_", " ")
            return f"Advisor flag: {flag_lower} — not suitable for setup"
        return None


# Module-level singleton
stock_screener = StockScreener()
