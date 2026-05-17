"""
SwingAdvisorBot — Module 3: Risk Management Engine
calculators/portfolio_calculator.py — Portfolio-level risk math

Calculates portfolio-wide metrics from open positions:
  → Total invested capital
  → Available capital for new trades
  → Sector exposure percentages
  → Total capital at risk
  → Open trade count

Used by:
  - portfolio_validator.py (sector + trade count checks)
  - check_portfolio_risk MCP tool
  - RiskAssessmentAgent (Steps 8-9 of CoT)

All math uses Decimal. Never float for money.
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP

from module3_risk_engine.models import OpenPosition, PortfolioRiskReport
from module3_risk_engine.rules import RiskRules

logger = logging.getLogger("swing_advisor.portfolio_calculator")


class PortfolioCalculator:
    """Calculate portfolio-level risk metrics.

    Takes a list of OpenPosition objects and capital,
    returns aggregated risk metrics.

    Usage:
        calc = PortfolioCalculator()
        report = calc.calculate(
            capital=Decimal("50000.00"),
            positions=[position1, position2],
        )
    """

    def calculate(
        self,
        capital: Decimal,
        positions: list[OpenPosition],
    ) -> PortfolioRiskReport:
        """Calculate complete portfolio risk snapshot.

        Args:
            capital: Total trading capital in INR.
            positions: List of open positions.

        Returns:
            PortfolioRiskReport with all metrics.
        """
        total_invested = Decimal("0.00")
        total_risk = Decimal("0.00")
        sector_values: dict[str, Decimal] = {}

        for pos in positions:
            pos_value = pos.position_value
            pos_risk = pos.risk_amount
            total_invested += pos_value
            total_risk += pos_risk

            sector = pos.sector or RiskRules.get_sector(pos.ticker)
            sector_values[sector] = sector_values.get(
                sector, Decimal("0.00")
            ) + pos_value

        # Calculate percentages
        available_capital = (capital - total_invested).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        total_invested = total_invested.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        total_risk = total_risk.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        total_risk_pct = Decimal("0.00")
        if capital > Decimal("0"):
            total_risk_pct = (
                (total_risk / capital) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Sector exposure as % of capital
        sector_exposures: dict[str, Decimal] = {}
        if capital > Decimal("0"):
            for sector, value in sector_values.items():
                pct = (
                    (value / capital) * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                sector_exposures[sector] = pct

        logger.info(
            f"[PortfolioCalc] Capital={capital}, "
            f"Invested={total_invested}, "
            f"Available={available_capital}, "
            f"Risk={total_risk} ({total_risk_pct}%), "
            f"Positions={len(positions)}, "
            f"Sectors={sector_exposures}"
        )

        return PortfolioRiskReport(
            total_capital=capital,
            total_invested=total_invested,
            available_capital=available_capital,
            total_risk_rupees=total_risk,
            total_risk_pct=total_risk_pct,
            sector_exposures=sector_exposures,
            open_trade_count=len(positions),
            max_trades=RiskRules.MAX_OPEN_TRADES,
            positions=positions,
        )

    def get_sector_exposure(
        self,
        capital: Decimal,
        positions: list[OpenPosition],
        new_ticker: str,
        new_position_value: Decimal,
    ) -> dict:
        """Calculate sector exposure INCLUDING a proposed new trade.

        Used by Step 8 of the CoT to check if adding a new
        position would breach the 25% sector limit.

        Args:
            capital: Total trading capital.
            positions: Current open positions.
            new_ticker: Ticker of proposed new trade.
            new_position_value: Value of proposed new position.

        Returns:
            Dict with sector, current_pct, after_pct,
            limit_pct, within_limit.
        """
        new_sector = RiskRules.get_sector(new_ticker)

        # Current sector value
        current_sector_value = Decimal("0.00")
        for pos in positions:
            pos_sector = pos.sector or RiskRules.get_sector(pos.ticker)
            if pos_sector == new_sector:
                current_sector_value += pos.position_value

        # After adding new trade
        after_sector_value = current_sector_value + new_position_value

        # Percentages
        current_pct = Decimal("0.00")
        after_pct = Decimal("0.00")
        limit_pct = (RiskRules.MAX_SECTOR_PCT * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if capital > Decimal("0"):
            current_pct = (
                (current_sector_value / capital) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            after_pct = (
                (after_sector_value / capital) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        within_limit = after_pct <= limit_pct

        logger.info(
            f"[PortfolioCalc] Sector check: {new_ticker} ({new_sector}). "
            f"Current={current_pct}%, After={after_pct}%, "
            f"Limit={limit_pct}%, Within={within_limit}"
        )

        return {
            "sector": new_sector,
            "current_pct": current_pct,
            "after_pct": after_pct,
            "limit_pct": limit_pct,
            "within_limit": within_limit,
            "current_value": current_sector_value.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            "after_value": after_sector_value.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
        }

    def get_suggested_alternatives(
        self,
        positions: list[OpenPosition],
        exclude_sector: str,
    ) -> list[str]:
        """Suggest alternative tickers from different sectors.

        When a trade is rejected for sector overexposure,
        suggest stocks from underrepresented sectors.

        Args:
            positions: Current open positions.
            exclude_sector: Sector to avoid.

        Returns:
            List of up to 3 ticker suggestions.
        """
        # Find sectors already held
        held_sectors = set()
        for pos in positions:
            sector = pos.sector or RiskRules.get_sector(pos.ticker)
            held_sectors.add(sector)

        # Find tickers from other sectors
        alternatives = []
        for ticker, sector in RiskRules.SECTOR_MAP.items():
            if sector != exclude_sector and sector not in held_sectors:
                alternatives.append(ticker)
                if len(alternatives) >= 3:
                    break

        return alternatives


# Module-level singleton
portfolio_calculator = PortfolioCalculator()
