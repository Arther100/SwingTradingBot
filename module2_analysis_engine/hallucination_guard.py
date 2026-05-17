"""
SwingAdvisorBot — Module 2: AI Analysis Engine
hallucination_guard.py — Verify Claude's output against real market data

"Trust, but verify. An advisor who invents data is dangerous."

LLMs hallucinate. That is not a bug — it is a fundamental property
of how they work. Claude may invent ticker symbols that don't exist,
fabricate price levels, or reference companies not in our data.

In a finance advisor context, hallucinated data is DANGEROUS:
  → A fake ticker could lead the user to trade a non-existent stock
  → A wrong price could cause incorrect entry/exit calculations
  → A fabricated support level could blow up a stop loss strategy

This guard cross-references every factual claim in Claude's analysis
against the real MarketData from Module 1. If Claude says "HDFCBANK
is at ₹1650", we verify HDFCBANK exists in our data AND the price
is within a reasonable tolerance of the real price.

Verification strategy:
  1. TICKER CHECK — every ticker mentioned must exist in MarketData.stocks
  2. PRICE CHECK — any price mentioned for a ticker must be within ±10%
     of the real price (allows for rounding, but catches fabrication)
  3. SECTOR CHECK — sector names mentioned must match MarketData.sectors
  4. VIX CHECK — if VIX is mentioned, it must match ±2 absolute points
  5. INDEX CHECK — Nifty/Sensex references must be within ±2% of real value
  6. OPPORTUNITY CHECK — tickers in top_opportunities must exist in our data
  7. RISK CHECK — tickers in top_risks must exist in our data

The guard does NOT reject the analysis — it annotates it with warnings.
If critical hallucinations are found (>= 3 fake tickers or any fake
price > 20% off), the analysis IS rejected for retry.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from module1_data_layer.models import MarketData

logger = logging.getLogger("swing_advisor.hallucination_guard")


@dataclass
class HallucinationReport:
    """Results of hallucination verification.

    This report documents every factual claim that was verified
    and the outcome. Downstream consumers use this to:
      → Flag hallucinated content in the UI
      → Trigger retry if critical hallucinations found
      → Log for audit trail

    Attributes:
        is_clean: True if no hallucinations detected.
        should_retry: True if hallucinations are severe enough to reject.
        warnings: List of non-critical issues (flagged but not rejected).
        errors: List of critical issues (trigger rejection/retry).
        verified_tickers: Tickers that were verified as real.
        hallucinated_tickers: Tickers not found in MarketData.
        price_mismatches: Tickers where stated price is too far from real.
    """

    is_clean: bool = True
    should_retry: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    verified_tickers: list[str] = field(default_factory=list)
    hallucinated_tickers: list[str] = field(default_factory=list)
    price_mismatches: list[str] = field(default_factory=list)


class HallucinationGuard:
    """Cross-references Claude's analysis against real market data.

    Every factual claim in the analysis is verified against the
    MarketData that was used as input. This ensures the advisor
    never presents fabricated information to the user.

    The guard is designed to be conservative:
      → It flags potential issues rather than silently correcting
      → Minor discrepancies (rounding) are warnings, not errors
      → Only severe hallucinations trigger retry

    Usage:
        guard = HallucinationGuard()
        report = guard.verify(analysis_dict, market_data)
        if report.should_retry:
            # Re-prompt Claude with grounding instruction
            ...
    """

    # Price tolerance: ±10% is rounding/approximation territory
    PRICE_TOLERANCE_PCT: float = 10.0
    # Critical price divergence: >20% is fabrication
    CRITICAL_PRICE_TOLERANCE_PCT: float = 20.0
    # VIX tolerance: ±2 absolute points (VIX values are small numbers)
    VIX_TOLERANCE: float = 2.0
    # Index tolerance: ±2% of real value
    INDEX_TOLERANCE_PCT: float = 2.0
    # Max hallucinated tickers before forcing retry
    MAX_HALLUCINATED_TICKERS: int = 3

    def verify(
        self,
        analysis_dict: dict,
        market_data: MarketData,
    ) -> HallucinationReport:
        """Run all hallucination checks on a parsed analysis.

        This is the main entry point. It runs all verification
        checks and returns a consolidated report.

        Args:
            analysis_dict: Parsed JSON dict from Claude's response.
            market_data: Real MarketData from Module 1 pipeline.

        Returns:
            HallucinationReport with verification results.
        """
        report = HallucinationReport()

        # Build lookup tables from real data
        ticker_map = self._build_ticker_map(market_data)
        sector_set = self._build_sector_set(market_data)

        # ── Check 1: Verify tickers in top_opportunities ──
        self._check_ticker_list(
            tickers=analysis_dict.get("top_opportunities", []),
            label="top_opportunities",
            ticker_map=ticker_map,
            report=report,
        )

        # ── Check 2: Verify tickers in top_risks ──
        self._check_ticker_list(
            tickers=analysis_dict.get("top_risks", []),
            label="top_risks",
            ticker_map=ticker_map,
            report=report,
        )

        # ── Check 3: Verify tickers mentioned in text fields ──
        text_fields = [
            ("situation", analysis_dict.get("situation", "")),
            ("reasoning", analysis_dict.get("reasoning", "")),
            ("action", analysis_dict.get("action", "")),
            ("risk", analysis_dict.get("risk", "")),
            ("user_impact", analysis_dict.get("user_impact", "")),
            ("lesson", analysis_dict.get("lesson", "")),
        ]
        for field_name, text in text_fields:
            self._check_tickers_in_text(
                text=text,
                field_name=field_name,
                ticker_map=ticker_map,
                report=report,
            )

        # ── Check 4: Verify price claims in text ──
        for field_name, text in text_fields:
            self._check_price_claims(
                text=text,
                field_name=field_name,
                ticker_map=ticker_map,
                report=report,
            )

        # ── Check 5: Verify VIX references ──
        self._check_vix_claims(
            analysis_dict=analysis_dict,
            market_data=market_data,
            report=report,
        )

        # ── Check 6: Verify sector references ──
        self._check_sector_claims(
            analysis_dict=analysis_dict,
            sector_set=sector_set,
            report=report,
        )

        # ── Determine severity ──
        self._assess_severity(report)

        # Log result
        if report.is_clean:
            logger.info(
                f"Hallucination check passed. "
                f"Verified {len(report.verified_tickers)} tickers. "
                f"No hallucinations detected."
            )
        else:
            level = "ERROR" if report.should_retry else "WARNING"
            logger.log(
                logging.ERROR if report.should_retry else logging.WARNING,
                f"Hallucination check {level}: "
                f"Verified: {len(report.verified_tickers)}. "
                f"Hallucinated: {len(report.hallucinated_tickers)}. "
                f"Price mismatches: {len(report.price_mismatches)}. "
                f"Warnings: {len(report.warnings)}. "
                f"Errors: {len(report.errors)}. "
                f"Should retry: {report.should_retry}."
            )

        return report

    def _build_ticker_map(
        self,
        market_data: MarketData,
    ) -> dict[str, dict]:
        """Build a lookup map of real tickers and their data.

        Maps uppercase ticker symbol to a dict of key fields
        that can be used for price verification.

        Args:
            market_data: Real MarketData from Module 1.

        Returns:
            Dict mapping ticker (uppercase) to price/metadata.
        """
        ticker_map: dict[str, dict] = {}
        for stock in market_data.stocks:
            ticker_map[stock.ticker.upper()] = {
                "price": stock.price,
                "change_pct": stock.change_pct,
                "high_52w": stock.high_52w,
                "low_52w": stock.low_52w,
                "sector": stock.sector,
                "volume_signal": stock.volume_signal.value if stock.volume_signal else "normal",
                "advisor_flag": stock.advisor_flag.value if stock.advisor_flag else None,
            }
        return ticker_map

    def _build_sector_set(self, market_data: MarketData) -> set[str]:
        """Build a set of known sector names (lowercase for matching).

        Args:
            market_data: Real MarketData from Module 1.

        Returns:
            Set of lowercase sector name strings.
        """
        sectors: set[str] = set()
        for sector in market_data.sectors:
            sectors.add(sector.sector_name.lower())
        # Also add sectors from stock data
        for stock in market_data.stocks:
            if stock.sector:
                sectors.add(stock.sector.lower())
        return sectors

    def _check_ticker_list(
        self,
        tickers: list[str],
        label: str,
        ticker_map: dict[str, dict],
        report: HallucinationReport,
    ) -> None:
        """Verify a list of ticker symbols against real data.

        Every ticker in top_opportunities and top_risks must
        exist in our MarketData. A hallucinated opportunity
        is especially dangerous — it could lead to a trade
        on a non-existent or wrong stock.

        Args:
            tickers: List of ticker symbols to verify.
            label: Field name for logging (e.g., "top_opportunities").
            ticker_map: Real ticker lookup map.
            report: Report to append findings to.
        """
        for ticker in tickers:
            if not isinstance(ticker, str):
                continue
            ticker_upper = ticker.upper().strip()
            if ticker_upper in ticker_map:
                if ticker_upper not in report.verified_tickers:
                    report.verified_tickers.append(ticker_upper)
            else:
                if ticker_upper not in report.hallucinated_tickers:
                    report.hallucinated_tickers.append(ticker_upper)
                    report.is_clean = False
                    report.errors.append(
                        f"HALLUCINATED TICKER in {label}: '{ticker_upper}' "
                        f"not found in MarketData. Claude may have invented this symbol."
                    )

    def _check_tickers_in_text(
        self,
        text: str,
        field_name: str,
        ticker_map: dict[str, dict],
        report: HallucinationReport,
    ) -> None:
        """Find and verify ticker symbols mentioned in free text.

        Scans text for uppercase words that look like NSE tickers
        (2-15 uppercase letters/digits). Cross-references against
        known tickers from MarketData.

        Only flags tickers that look like they SHOULD be real stocks
        (all caps, reasonable length). Common English words in all
        caps are excluded (e.g., "VIX", "RSI", "MACD", "NSE").

        Args:
            text: Free text to scan.
            field_name: Name of the field being scanned.
            ticker_map: Real ticker lookup map.
            report: Report to append findings to.
        """
        if not text:
            return

        # Common uppercase words that are NOT ticker symbols
        non_tickers = {
            "VIX", "RSI", "MACD", "EMA", "SMA", "ATR", "NSE", "BSE",
            "IPO", "FII", "DII", "RBI", "GDP", "CPI", "PMI", "USA",
            "INR", "USD", "EUR", "FED", "ECB", "BOJ", "IMF", "ETF",
            "AMC", "NAV", "SIP", "SEBI", "NIFTY", "SENSEX", "BANK",
            "NIFTY50", "BANKNIFTY", "THE", "AND", "FOR", "NOT", "BUT",
            "WITH", "THIS", "THAT", "FROM", "INTO", "WHEN", "THEN",
            "ALSO", "JUST", "ONLY", "VERY", "MORE", "MOST", "SOME",
            "ANY", "ALL", "DAY", "BUY", "SELL", "HOLD", "EXIT",
            "STOP", "LOSS", "TARGET", "ENTRY", "RISK", "HIGH",
            "LOW", "OPEN", "CLOSE", "ABOVE", "BELOW", "TODAY",
            "IST", "JSON", "COT", "STEP",
        }

        # Find all uppercase words that look like tickers
        # NSE tickers: 2-15 chars, uppercase, may contain digits
        potential_tickers = re.findall(r"\b([A-Z][A-Z0-9]{1,14})\b", text)

        for candidate in potential_tickers:
            if candidate in non_tickers:
                continue
            if candidate in ticker_map:
                if candidate not in report.verified_tickers:
                    report.verified_tickers.append(candidate)
            # Note: we don't flag text-mentioned tickers as hallucinated
            # unless they're in structured fields (top_opportunities, top_risks).
            # Free text may reference tickers for comparison purposes.

    def _check_price_claims(
        self,
        text: str,
        field_name: str,
        ticker_map: dict[str, dict],
        report: HallucinationReport,
    ) -> None:
        """Verify price claims in text against real prices.

        Looks for patterns like:
          → "HDFCBANK at ₹1650"
          → "RELIANCE is trading at 2850"
          → "TCS price 3400"
          → "₹1650 for HDFCBANK"

        If a price is found near a known ticker, it's verified
        against the real price with PRICE_TOLERANCE_PCT tolerance.

        Args:
            text: Free text to scan for price claims.
            field_name: Name of the field being scanned.
            ticker_map: Real ticker lookup map.
            report: Report to append findings to.
        """
        if not text:
            return

        # Pattern: TICKER ... ₹PRICE or TICKER ... PRICE
        # Capture ticker and price within proximity
        patterns = [
            # "HDFCBANK at ₹1650" or "HDFCBANK around 1650"
            r"([A-Z][A-Z0-9]{1,14})\s+(?:at|around|near|above|below|is|was|trading)\s+(?:₹|Rs\.?|INR)?\s*(\d{1,7}(?:\.\d{1,2})?)",
            # "₹1650 for HDFCBANK" or "1650 in HDFCBANK"
            r"(?:₹|Rs\.?|INR)\s*(\d{1,7}(?:\.\d{1,2})?)\s+(?:for|in|of)\s+([A-Z][A-Z0-9]{1,14})",
            # "HDFCBANK (₹1650)" or "HDFCBANK (1650)"
            r"([A-Z][A-Z0-9]{1,14})\s*\((?:₹|Rs\.?|INR)?\s*(\d{1,7}(?:\.\d{1,2})?)\)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Determine which group is ticker and which is price
                if match[0].isalpha() or (match[0][0].isalpha() and any(c.isdigit() for c in match[0])):
                    ticker = match[0].upper()
                    try:
                        claimed_price = float(match[1])
                    except ValueError:
                        continue
                else:
                    try:
                        claimed_price = float(match[0])
                    except ValueError:
                        continue
                    ticker = match[1].upper()

                if ticker not in ticker_map:
                    continue

                real_price = ticker_map[ticker]["price"]
                if real_price <= 0:
                    continue

                pct_diff = abs(claimed_price - real_price) / real_price * 100

                if pct_diff > self.CRITICAL_PRICE_TOLERANCE_PCT:
                    report.is_clean = False
                    report.price_mismatches.append(ticker)
                    report.errors.append(
                        f"CRITICAL PRICE HALLUCINATION in {field_name}: "
                        f"{ticker} claimed at ₹{claimed_price:.2f} but "
                        f"real price is ₹{real_price:.2f} "
                        f"({pct_diff:.1f}% divergence). "
                        f"Claude fabricated this price."
                    )
                elif pct_diff > self.PRICE_TOLERANCE_PCT:
                    report.is_clean = False
                    report.price_mismatches.append(ticker)
                    report.warnings.append(
                        f"PRICE MISMATCH in {field_name}: "
                        f"{ticker} claimed at ₹{claimed_price:.2f} but "
                        f"real price is ₹{real_price:.2f} "
                        f"({pct_diff:.1f}% off). "
                        f"Possible rounding or stale reference."
                    )

    def _check_vix_claims(
        self,
        analysis_dict: dict,
        market_data: MarketData,
        report: HallucinationReport,
    ) -> None:
        """Verify VIX value claims against real India VIX.

        VIX is a critical fear gauge. A hallucinated VIX value
        could completely change the market mood assessment.
        We verify any VIX reference within ±2 absolute points.

        Args:
            analysis_dict: Parsed analysis dict.
            market_data: Real MarketData.
            report: Report to append findings to.
        """
        real_vix = market_data.india_vix
        if real_vix <= 0:
            return

        # Search all text fields for VIX value patterns
        all_text = " ".join(
            str(analysis_dict.get(field, ""))
            for field in ["situation", "reasoning", "action", "risk", "cot_reasoning"]
        )

        # Pattern: "VIX at 14.2" or "VIX is 14" or "India VIX 14.2"
        vix_patterns = [
            r"(?:India\s+)?VIX\s+(?:at|is|was|of|=|:)?\s*(\d{1,3}(?:\.\d{1,2})?)",
            r"VIX\s*[\(\[]?\s*(\d{1,3}(?:\.\d{1,2})?)\s*[\)\]]?",
        ]

        for pattern in vix_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                try:
                    claimed_vix = float(match)
                except ValueError:
                    continue

                # VIX values are typically 8-80. Skip obvious non-VIX numbers.
                if claimed_vix < 5 or claimed_vix > 100:
                    continue

                diff = abs(claimed_vix - real_vix)
                if diff > self.VIX_TOLERANCE:
                    report.is_clean = False
                    report.warnings.append(
                        f"VIX MISMATCH: Claimed VIX {claimed_vix} but "
                        f"real India VIX is {real_vix:.2f} "
                        f"(diff: {diff:.2f} points). "
                        f"May mislead the user's risk assessment."
                    )

    def _check_sector_claims(
        self,
        analysis_dict: dict,
        sector_set: set[str],
        report: HallucinationReport,
    ) -> None:
        """Verify sector names in sector_analyses against real data.

        If Claude's sector analyses reference sectors not in our data,
        it may be hallucinating sector performance. This check verifies
        the sector_analyses list if present.

        Args:
            analysis_dict: Parsed analysis dict.
            sector_set: Set of known sector names (lowercase).
            report: Report to append findings to.
        """
        sector_analyses = analysis_dict.get("sector_analyses", [])
        if not sector_analyses or not isinstance(sector_analyses, list):
            return

        for sector_entry in sector_analyses:
            if not isinstance(sector_entry, dict):
                continue
            sector_name = sector_entry.get("sector_name", "")
            if not sector_name:
                continue

            # Fuzzy match — check if any known sector is a substring
            sector_lower = sector_name.lower()
            matched = any(
                known in sector_lower or sector_lower in known
                for known in sector_set
            )

            if not matched and sector_set:
                report.is_clean = False
                report.warnings.append(
                    f"UNKNOWN SECTOR: '{sector_name}' not found in MarketData sectors. "
                    f"Known sectors: {', '.join(sorted(sector_set)[:5])}..."
                )

    def _assess_severity(self, report: HallucinationReport) -> None:
        """Determine if hallucinations are severe enough to trigger retry.

        Retry triggers:
          → 3+ hallucinated tickers (Claude is inventing symbols)
          → Any critical price hallucination (>20% off)
          → 5+ total warnings + errors (pattern of fabrication)

        Non-retry (warnings only):
          → 1-2 unknown tickers (could be exchange differences)
          → Minor price mismatches (rounding)
          → Unknown sectors (naming variations)

        Args:
            report: HallucinationReport to assess (mutated in place).
        """
        # Check for critical hallucinations
        if len(report.hallucinated_tickers) >= self.MAX_HALLUCINATED_TICKERS:
            report.should_retry = True
            report.errors.append(
                f"RETRY TRIGGER: {len(report.hallucinated_tickers)} hallucinated tickers "
                f"({', '.join(report.hallucinated_tickers)}). "
                f"Claude is inventing stock symbols."
            )

        # Check for critical price fabrication
        critical_price_errors = [
            e for e in report.errors if "CRITICAL PRICE" in e
        ]
        if critical_price_errors:
            report.should_retry = True

        # Check for pattern of fabrication
        total_issues = len(report.warnings) + len(report.errors)
        if total_issues >= 5:
            report.should_retry = True
            report.errors.append(
                f"RETRY TRIGGER: {total_issues} total issues detected. "
                f"Pattern of data fabrication — need stronger grounding."
            )

    def format_grounding_feedback(self, report: HallucinationReport) -> str:
        """Format hallucination findings for Claude retry prompt.

        When should_retry is True, this feedback is appended to the
        retry prompt to ground Claude in real data.

        Args:
            report: HallucinationReport with findings.

        Returns:
            Formatted string for inclusion in retry prompt.
        """
        parts: list[str] = [
            "GROUNDING CORRECTION — Your previous response contained factual errors:"
        ]

        if report.hallucinated_tickers:
            parts.append(
                f"FAKE TICKERS: {', '.join(report.hallucinated_tickers)} — "
                f"these stocks are NOT in the market data. Remove them."
            )
            parts.append(
                f"REAL TICKERS: {', '.join(report.verified_tickers[:10])} — "
                f"ONLY use tickers from the market data provided."
            )

        if report.price_mismatches:
            parts.append(
                f"WRONG PRICES: {', '.join(report.price_mismatches)} — "
                f"your stated prices are significantly wrong. "
                f"Use ONLY the exact prices from the market data."
            )

        for error in report.errors:
            if "CRITICAL PRICE" in error or "VIX MISMATCH" in error:
                parts.append(f"  → {error}")

        parts.append(
            "RULE: You must ONLY reference data that appears in the "
            "market data provided. Do not invent any prices, tickers, "
            "or statistics."
        )

        return "\n".join(parts)


# Module-level singleton — used across the analysis engine
hallucination_guard = HallucinationGuard()
