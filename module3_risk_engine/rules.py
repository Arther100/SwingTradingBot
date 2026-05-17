"""
SwingAdvisorBot — Module 3: Risk Management Engine
rules.py — Immutable risk rules

These rules are the LAW.
Never modified at runtime.
No user input overrides these.
No Claude API output overrides these.

A trader who loses 50% needs 100% gain to recover.
These rules exist to prevent that scenario.

Rules:
  VIX_GATES          → Max VIX for each risk tolerance
  RISK_PCT_LIMITS    → Max % of capital to risk per trade
  MAX_POSITION_PCT   → Max position size as % of capital
  MAX_SECTOR_PCT     → Max sector exposure as % of capital
  MIN_RISK_REWARD    → Minimum acceptable risk/reward ratio
  MAX_OPEN_TRADES    → Maximum concurrent positions
  MIN_CAPITAL        → Minimum capital to trade

Sector map:
  20 Nifty 50 stocks mapped to their sectors.
  Unknown tickers default to "Other".
"""

from __future__ import annotations

from decimal import Decimal


class RiskRules:
    """Hardcoded risk rules — immutable at runtime.

    No user input overrides these.
    No Claude API output overrides these.
    Violated rule = REJECTED. No exceptions.

    Usage:
        vix_limit = RiskRules.get_vix_gate("moderate")  # Decimal("20.0")
        risk_pct = RiskRules.get_risk_pct("moderate")   # Decimal("0.02")
        sector = RiskRules.get_sector("HDFCBANK")        # "Banking"
    """

    # ── VIX Gate Thresholds ──
    # VIX above these values → no new swing trades
    VIX_GATES: dict[str, Decimal] = {
        "conservative": Decimal("15.0"),
        "moderate": Decimal("20.0"),
        "aggressive": Decimal("25.0"),
    }

    # ── Max Risk Per Trade (as fraction of capital) ──
    # Conservative: 1% → lose 10 in a row = 10% drawdown
    # Moderate:     2% → lose 10 in a row = 20% drawdown
    # Aggressive:   3% → lose 10 in a row = 30% drawdown
    RISK_PCT_LIMITS: dict[str, Decimal] = {
        "conservative": Decimal("0.01"),
        "moderate": Decimal("0.02"),
        "aggressive": Decimal("0.03"),
    }

    # ── Position & Portfolio Limits ──
    MAX_POSITION_PCT: Decimal = Decimal("0.50")    # 50% max position size
    MAX_SECTOR_PCT: Decimal = Decimal("0.50")      # 50% max sector exposure
    MIN_RISK_REWARD: Decimal = Decimal("2.0")      # Minimum 1:2 R/R ratio
    IDEAL_RISK_REWARD: Decimal = Decimal("3.0")    # Ideal 1:3 R/R ratio
    MAX_OPEN_TRADES: int = 5                       # Max concurrent trades
    MIN_CAPITAL: Decimal = Decimal("5000.00")      # Minimum capital to trade

    # ── Sector Classification ──
    # 20 Nifty 50 stocks mapped to sectors.
    # Unknown tickers default to "Other" via get_sector().
    SECTOR_MAP: dict[str, str] = {
        # Banking
        "HDFCBANK": "Banking",
        "ICICIBANK": "Banking",
        "KOTAKBANK": "Banking",
        "SBIN": "Banking",
        "AXISBANK": "Banking",
        # IT
        "TCS": "IT",
        "INFY": "IT",
        "WIPRO": "IT",
        "TECHM": "IT",
        "HCLTECH": "IT",
        # Energy
        "RELIANCE": "Energy",
        "ONGC": "Energy",
        "NTPC": "Energy",
        # Telecom
        "BHARTIARTL": "Telecom",
        # FMCG
        "HINDUNILVR": "FMCG",
        "ITC": "FMCG",
        # Auto
        "MARUTI": "Auto",
        "TATAMOTORS": "Auto",
        # Infrastructure
        "LT": "Infrastructure",
        # Consumer
        "ASIANPAINT": "Consumer",
    }

    # ── VIX Signal Classification ──
    # Matches M1 VIXSignal enum thresholds
    VIX_THRESHOLDS: dict[str, tuple[Decimal, Decimal]] = {
        "low_fear": (Decimal("0"), Decimal("14")),
        "moderate_fear": (Decimal("14"), Decimal("20")),
        "high_fear": (Decimal("20"), Decimal("30")),
        "extreme_fear": (Decimal("30"), Decimal("999")),
    }

    @classmethod
    def get_sector(cls, ticker: str) -> str:
        """Get sector for a ticker. Defaults to 'Other' if unknown.

        Args:
            ticker: NSE ticker symbol, e.g. 'HDFCBANK'.

        Returns:
            Sector name string, e.g. 'Banking'.
        """
        return cls.SECTOR_MAP.get(ticker.upper(), "Other")

    @classmethod
    def get_vix_gate(cls, tolerance: str) -> Decimal:
        """Get VIX gate threshold for a risk tolerance level.

        Args:
            tolerance: 'conservative', 'moderate', or 'aggressive'.

        Returns:
            Decimal VIX limit. Defaults to moderate (20.0) if unknown.
        """
        return cls.VIX_GATES.get(tolerance, cls.VIX_GATES["moderate"])

    @classmethod
    def get_risk_pct(cls, tolerance: str) -> Decimal:
        """Get max risk percentage for a risk tolerance level.

        Args:
            tolerance: 'conservative', 'moderate', or 'aggressive'.

        Returns:
            Decimal fraction (e.g. 0.02 for 2%). Defaults to moderate.
        """
        return cls.RISK_PCT_LIMITS.get(
            tolerance, cls.RISK_PCT_LIMITS["moderate"]
        )

    @classmethod
    def classify_vix(cls, vix_value: Decimal) -> str:
        """Classify VIX value into a signal category.

        Args:
            vix_value: Current India VIX as Decimal.

        Returns:
            Signal string: 'low_fear', 'moderate_fear',
            'high_fear', or 'extreme_fear'.
        """
        for signal, (low, high) in cls.VIX_THRESHOLDS.items():
            if low <= vix_value < high:
                return signal
        return "extreme_fear"

    @classmethod
    def get_suggested_target(
        cls,
        entry_price: Decimal,
        stop_loss: Decimal,
        min_rr: Decimal | None = None,
    ) -> Decimal:
        """Calculate minimum target price for acceptable R/R.

        Args:
            entry_price: Proposed entry price.
            stop_loss: Proposed stop loss price.
            min_rr: Minimum R/R ratio. Defaults to MIN_RISK_REWARD (2.0).

        Returns:
            Minimum target price as Decimal.
        """
        if min_rr is None:
            min_rr = cls.MIN_RISK_REWARD
        risk = entry_price - stop_loss
        return entry_price + (risk * min_rr)
