"""
SwingAdvisorBot — Module 4: Trade Setup Generator
technical/stock_screener.py — Filter stocks by advisor signals

The screener takes M1 MarketData and selects candidate stocks
for setup generation. Only stocks with actionable signals pass.

Selection logic:
  1. Include stocks with priority flags (breakout, momentum, etc.)
  2. Exclude stocks with skip flags (selling_pressure, neutral, etc.)
  3. Sort by flag priority (breakout > unusual > momentum > accumulation)
  4. Limit to max_candidates (default 10)

This is a FAST filter — no API calls, no heavy computation.
Runs in < 1ms on 50 stocks.
"""

from __future__ import annotations

import logging
from typing import Optional

from module1_data_layer.models import MarketData, StockData
from module4_setup_generator.config import FLAG_PRIORITY, SKIP_FLAGS

logger = logging.getLogger("swing_advisor.stock_screener")


class StockScreener:
    """Filter M1 stocks for setup candidacy.

    Usage:
        screener = StockScreener()
        candidates = screener.screen(
            market_data=market_data,
            max_candidates=10,
        )
        # Returns list of StockData sorted by signal priority
    """

    def screen(
        self,
        market_data: MarketData,
        max_candidates: int = 10,
        specific_tickers: Optional[list[str]] = None,
    ) -> list[StockData]:
        """Screen stocks for setup generation.

        Args:
            market_data: M1 MarketData with stocks list.
            max_candidates: Maximum stocks to return.
            specific_tickers: If provided, only evaluate these tickers.

        Returns:
            List of StockData sorted by flag priority, limited to max_candidates.
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

        # Separate into qualifying and skipped
        candidates = []
        skipped = []

        for stock in stocks:
            flag = self._get_flag_value(stock)

            if flag in SKIP_FLAGS:
                skipped.append((stock.ticker, flag))
                continue

            if flag in FLAG_PRIORITY:
                candidates.append(stock)
            elif flag is None:
                # No flag at all — skip
                skipped.append((stock.ticker, "no_flag"))
            else:
                # Unknown flag — include with low priority
                candidates.append(stock)

        # Sort by flag priority (lower number = higher priority)
        candidates.sort(key=lambda s: self._get_priority(s))

        # Limit to max_candidates
        result = candidates[:max_candidates]

        logger.info(
            f"[StockScreener] Evaluated {len(stocks)} stocks. "
            f"Candidates: {len(result)}, "
            f"Skipped: {len(skipped)} "
            f"({', '.join(f'{t}({f})' for t, f in skipped[:5])})"
        )

        return result

    def _get_flag_value(self, stock: StockData) -> Optional[str]:
        """Get the string value of a stock's advisor flag."""
        if stock.advisor_flag is None:
            return None
        # Handle both enum and string
        if hasattr(stock.advisor_flag, "value"):
            return stock.advisor_flag.value.upper()
        return str(stock.advisor_flag).upper()

    def _get_priority(self, stock: StockData) -> int:
        """Get sort priority for a stock (lower = better)."""
        flag = self._get_flag_value(stock)
        if flag is None:
            return 99
        return FLAG_PRIORITY.get(flag, 50)

    def get_skip_reason(self, stock: StockData) -> Optional[str]:
        """Get the reason a stock would be skipped.

        Returns None if the stock qualifies.
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
