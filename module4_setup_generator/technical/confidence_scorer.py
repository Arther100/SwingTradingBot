"""
SwingAdvisorBot — Module 4: Trade Setup Generator
technical/confidence_scorer.py — Score setup quality 4.0 to 9.5

Pure math scoring — no AI, no API calls.
Takes stock data + market context → confidence score.

Scoring components (total range 4.0 – 9.5):
  Base score:       5.0
  VIX bonus:        +0.0 to +1.0 (lower VIX = higher confidence)
  Sector alignment: +0.0 to +1.0 (sector mood matches trade direction)
  Volume signal:    +0.0 to +1.0 (stronger volume = higher confidence)
  Flag quality:     +0.0 to +1.0 (breakout > accumulation > momentum)
  R/R quality:      +0.0 to +0.5 (risk/reward ratio bonus)

Penalties:
  VIX > 20:           -0.5
  Sector bearish:     -0.5
  Below avg volume:   -0.5
  Weak flag:          -0.5

Design decisions:
  - Average setup should score 6.5–7.5
  - Excellent setups (all factors aligned): 8.0–9.5
  - Min score clamped to 4.0 (never show garbage scores)
  - Max score clamped to 9.5 (nothing is perfect)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from module1_data_layer.models import StockData, VolumeSignal

logger = logging.getLogger("swing_advisor.confidence_scorer")


class ConfidenceScorer:
    """Score a trade setup's quality based on multiple factors.

    Usage:
        scorer = ConfidenceScorer()
        score = scorer.score(
            stock=stock_data,
            india_vix=14.2,
            sector_mood="bullish",
            risk_reward_ratio=Decimal("3.0"),
        )
        # Returns float in range 4.0 – 9.5
    """

    BASE_SCORE = 5.0
    MIN_SCORE = 4.0
    MAX_SCORE = 9.5

    def score(
        self,
        stock: StockData,
        india_vix: float = 15.0,
        sector_mood: Optional[str] = None,
        risk_reward_ratio: Optional[Decimal] = None,
    ) -> float:
        """Calculate confidence score for a setup.

        Args:
            stock: M1 StockData with advisor_flag and volume_signal.
            india_vix: Current India VIX value.
            sector_mood: M2 sector mood string (bullish/bearish/neutral).
            risk_reward_ratio: Calculated R/R ratio as Decimal.

        Returns:
            Confidence score clamped to 4.0 – 9.5.
        """
        score = self.BASE_SCORE
        breakdown: list[str] = []

        # ── 1. VIX Component: +1.0 to -0.5 ──
        vix_adj = self._score_vix(india_vix)
        score += vix_adj
        breakdown.append(f"vix={vix_adj:+.1f}")

        # ── 2. Sector Alignment: +1.0 to -0.5 ──
        sector_adj = self._score_sector(sector_mood)
        score += sector_adj
        breakdown.append(f"sector={sector_adj:+.1f}")

        # ── 3. Volume Signal: +1.0 to -0.5 ──
        volume_adj = self._score_volume(stock.volume_signal)
        score += volume_adj
        breakdown.append(f"volume={volume_adj:+.1f}")

        # ── 4. Flag Quality: +1.0 to -0.5 ──
        flag_adj = self._score_flag(stock)
        score += flag_adj
        breakdown.append(f"flag={flag_adj:+.1f}")

        # ── 5. R/R Quality: +0.0 to +0.5 ──
        rr_adj = self._score_risk_reward(risk_reward_ratio)
        score += rr_adj
        breakdown.append(f"rr={rr_adj:+.1f}")

        # ── Clamp to valid range ──
        final = round(max(self.MIN_SCORE, min(self.MAX_SCORE, score)), 1)

        logger.info(
            f"[ConfidenceScorer] {stock.ticker}: "
            f"base={self.BASE_SCORE} {' '.join(breakdown)} → {final}"
        )

        return final

    def _score_vix(self, india_vix: float) -> float:
        """VIX scoring — low VIX rewards, high VIX penalises."""
        if india_vix <= 12.0:
            return 1.0   # Very low fear — ideal
        elif india_vix <= 15.0:
            return 0.7   # Low fear — good
        elif india_vix <= 18.0:
            return 0.3   # Moderate — acceptable
        elif india_vix <= 20.0:
            return 0.0   # Elevated — no bonus
        elif india_vix <= 25.0:
            return -0.5  # High fear — penalty
        else:
            return -0.5  # Extreme — max penalty (capped)

    def _score_sector(self, sector_mood: Optional[str]) -> float:
        """Sector mood alignment — bullish rewards, bearish penalises."""
        if sector_mood is None:
            return 0.0

        mood = sector_mood.upper()
        if mood in ("BULLISH", "CAUTIOUS_BULLISH"):
            return 1.0
        elif mood == "NEUTRAL":
            return 0.0
        elif mood in ("CAUTIOUS_BEARISH", "BEARISH"):
            return -0.5
        elif mood == "EXTREME_FEAR":
            return -0.5
        return 0.0

    def _score_volume(self, volume_signal: VolumeSignal) -> float:
        """Volume signal scoring — higher volume confirms the setup."""
        scores = {
            VolumeSignal.UNUSUAL_SPIKE: 1.0,
            VolumeSignal.ABOVE_AVERAGE: 0.5,
            VolumeSignal.NORMAL: 0.0,
            VolumeSignal.BELOW_AVERAGE: -0.5,
        }
        return scores.get(volume_signal, 0.0)

    def _score_flag(self, stock: StockData) -> float:
        """Advisor flag quality — stronger signals score higher."""
        if stock.advisor_flag is None:
            return -0.5

        flag = stock.advisor_flag.value.upper() if hasattr(stock.advisor_flag, "value") else str(stock.advisor_flag).upper()

        flag_scores: dict[str, float] = {
            "BREAKOUT_WATCH": 1.0,
            "UNUSUAL_ACTIVITY": 0.7,
            "MOMENTUM_BUILDING": 0.5,
            "ACCUMULATION_ZONE": 0.3,
            "CONSOLIDATION": 0.0,
        }
        return flag_scores.get(flag, -0.5)

    def _score_risk_reward(self, risk_reward_ratio: Optional[Decimal]) -> float:
        """R/R bonus — only positive, never penalises."""
        if risk_reward_ratio is None:
            return 0.0

        rr = float(risk_reward_ratio)
        if rr >= 3.5:
            return 0.5
        elif rr >= 3.0:
            return 0.3
        elif rr >= 2.5:
            return 0.1
        return 0.0


# Module-level singleton
confidence_scorer = ConfidenceScorer()
