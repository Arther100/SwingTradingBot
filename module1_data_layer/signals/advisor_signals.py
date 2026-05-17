"""
SwingAdvisorBot — Module 1: Data Layer
signals/advisor_signals.py — Chain of Thought signal calculator

This is where raw stock data transforms into advisor intelligence.
The fetchers give us numbers — this module gives us meaning.

A stock at ₹1623 with 8.5M volume is data.
A stock at ₹1623, 9.5% below 52w high, volume 37% above average,
flagged as "accumulation_zone" with CoT reasoning — that is a signal
a senior finance advisor can act on.

Every signal follows the explicit 5-step Chain of Thought pattern:
  Step 1: Analyze 52-week range position (where is the stock?)
  Step 2: Analyze volume anomaly (is someone accumulating/distributing?)
  Step 3: Analyze daily price movement (momentum direction + magnitude)
  Step 4: Combine signals → decide advisor_flag
  Step 5: Generate cot_reasoning string explaining the decision

Signal decision matrix (Step 4 logic):
  ┌─────────────────┬──────────────────┬──────────────────┬────────────────────┐
  │ 52w Position     │ Volume Signal    │ Price Change     │ Advisor Flag       │
  ├─────────────────┼──────────────────┼──────────────────┼────────────────────┤
  │ near_high        │ above_average    │ > +1%            │ breakout_watch     │
  │ near_high        │ unusual_spike    │ any              │ breakout_watch     │
  │ near_high        │ above_average    │ < -1%            │ distribution_zone  │
  │ upper/middle     │ above_average    │ > 0%             │ momentum_building  │
  │ lower            │ above_average    │ > 0%             │ accumulation_zone  │
  │ near_low         │ above_average    │ > 0%             │ accumulation_zone  │
  │ lower/near_low   │ above_average    │ < -1%            │ selling_pressure   │
  │ any              │ unusual_spike    │ < -2%            │ selling_pressure   │
  │ any              │ unusual_spike    │ > +2%            │ breakout_watch     │
  │ any              │ normal           │ -0.5% to +0.5%  │ consolidation      │
  │ any              │ below_average    │ any              │ consolidation      │
  │ (default)        │ any              │ any              │ neutral            │
  └─────────────────┴──────────────────┴──────────────────┴────────────────────┘

Data flow:
  StockData (from stock_fetcher) → calculate_advisor_flag() → StockData with signals
  List[StockData] → calculate_all_signals() → List[StockData] sorted by signal strength
"""

from __future__ import annotations

import logging

from module1_data_layer.models import (
    AdvisorFlag,
    RangePosition,
    StockData,
    VolumeSignal,
)

logger = logging.getLogger("swing_advisor.signals.advisor")

# Signal strength ranking — used for sorting stocks by signal importance.
# Higher number = more interesting to the advisor.
# Stocks are sorted by this ranking so the most actionable signals
# appear first in the MarketData.stocks list (and survive token trimming).
SIGNAL_STRENGTH: dict[AdvisorFlag, int] = {
    AdvisorFlag.BREAKOUT_WATCH: 90,
    AdvisorFlag.UNUSUAL_ACTIVITY: 85,
    AdvisorFlag.SELLING_PRESSURE: 80,
    AdvisorFlag.DISTRIBUTION_ZONE: 75,
    AdvisorFlag.ACCUMULATION_ZONE: 70,
    AdvisorFlag.MOMENTUM_BUILDING: 60,
    AdvisorFlag.CONSOLIDATION: 30,
    AdvisorFlag.NEUTRAL: 10,
}


def calculate_advisor_flag(stock: StockData) -> tuple[AdvisorFlag, str]:
    """Calculate the advisor flag and CoT reasoning for a single stock.

    Follows the explicit 5-step Chain of Thought pattern.
    Every decision is documented in the reasoning string so
    the advisor (Module 2) can explain WHY a stock was flagged.

    Args:
        stock: StockData with price, volume, and 52w range data populated.
               Must have volume_signal, position_in_52w_range, and change_pct
               set (computed by Pydantic model_validator in stock_fetcher).

    Returns:
        Tuple of (AdvisorFlag, cot_reasoning_string).
        The flag is the primary signal label.
        The reasoning is a multi-step explanation of how the flag was derived.
    """
    reasoning_steps: list[str] = []

    # ── Step 1: 52-week range analysis ──
    range_position = stock.position_in_52w_range
    if stock.high_52w > 0 and stock.low_52w > 0 and stock.high_52w > stock.low_52w:
        range_total = stock.high_52w - stock.low_52w
        position_pct = (stock.price - stock.low_52w) / range_total
        distance_from_high = ((stock.high_52w - stock.price) / stock.high_52w) * 100
        reasoning_steps.append(
            f"Step 1: Price at {position_pct:.0%} of 52w range "
            f"(₹{stock.low_52w:,.2f} to ₹{stock.high_52w:,.2f}). "
            f"{distance_from_high:.1f}% below 52w high. "
            f"Position: {range_position.value}."
        )
    else:
        reasoning_steps.append(
            f"Step 1: 52-week range data unavailable for {stock.ticker}. "
            f"Cannot assess range position — defaulting to middle."
        )

    # ── Step 2: Volume analysis ──
    volume_signal = stock.volume_signal
    volume_ratio = stock.volume_ratio
    if volume_ratio > 0:
        reasoning_steps.append(
            f"Step 2: Volume ratio {volume_ratio:.2f}x "
            f"(today: {stock.volume:,} vs 30d avg: {stock.avg_volume_30d:,}). "
            f"Signal: {volume_signal.value}."
        )
    else:
        reasoning_steps.append(
            f"Step 2: Volume analysis unavailable — "
            f"30d average volume data missing. Signal: {volume_signal.value}."
        )

    # ── Step 3: Daily price movement ──
    change_pct = stock.change_pct
    if change_pct > 2:
        movement_desc = "strong upward momentum"
    elif change_pct > 0.5:
        movement_desc = "positive movement"
    elif change_pct > -0.5:
        movement_desc = "flat/sideways"
    elif change_pct > -2:
        movement_desc = "negative movement"
    else:
        movement_desc = "strong downward pressure"

    reasoning_steps.append(
        f"Step 3: Daily change {change_pct:+.2f}% — {movement_desc}."
    )

    # ── Step 4: Signal decision ──
    advisor_flag = _decide_flag(range_position, volume_signal, change_pct)

    flag_explanation = _explain_flag_decision(
        advisor_flag, range_position, volume_signal, change_pct
    )
    reasoning_steps.append(f"Step 4: {flag_explanation}")

    # ── Step 5: Combine reasoning ──
    cot_reasoning = " | ".join(reasoning_steps)
    reasoning_steps.append(
        f"Step 5: Final signal → {advisor_flag.value} for {stock.ticker}."
    )

    full_reasoning = " | ".join(reasoning_steps)

    logger.info(
        f"[{stock.ticker}] Advisor flag: {advisor_flag.value} — "
        f"52w: {range_position.value}, vol: {volume_signal.value}, "
        f"change: {change_pct:+.2f}%."
    )

    return advisor_flag, full_reasoning


def _decide_flag(
    range_pos: RangePosition,
    vol_signal: VolumeSignal,
    change_pct: float,
) -> AdvisorFlag:
    """Core decision logic — maps signal combination to AdvisorFlag.

    This implements the decision matrix documented in the module docstring.
    Rules are evaluated in priority order — first match wins.

    Priority order ensures that the most actionable signals (breakout,
    unusual activity) are detected before less urgent ones (consolidation).

    Args:
        range_pos: Stock's position in 52-week range.
        vol_signal: Volume classification relative to 30d average.
        change_pct: Today's percentage change.

    Returns:
        AdvisorFlag enum member.
    """
    # ── Rule 1: Unusual volume spike with strong price move → high priority ──
    if vol_signal == VolumeSignal.UNUSUAL_SPIKE:
        if change_pct > 2.0:
            return AdvisorFlag.BREAKOUT_WATCH
        elif change_pct < -2.0:
            return AdvisorFlag.SELLING_PRESSURE
        else:
            return AdvisorFlag.UNUSUAL_ACTIVITY

    # ── Rule 2: Near 52w high scenarios ──
    if range_pos == RangePosition.NEAR_HIGH:
        if vol_signal == VolumeSignal.ABOVE_AVERAGE and change_pct > 1.0:
            return AdvisorFlag.BREAKOUT_WATCH
        elif vol_signal == VolumeSignal.ABOVE_AVERAGE and change_pct < -1.0:
            return AdvisorFlag.DISTRIBUTION_ZONE
        elif vol_signal == VolumeSignal.ABOVE_AVERAGE:
            return AdvisorFlag.MOMENTUM_BUILDING

    # ── Rule 3: Near 52w low / lower range with above-average volume ──
    if range_pos in (RangePosition.NEAR_LOW, RangePosition.LOWER):
        if vol_signal == VolumeSignal.ABOVE_AVERAGE and change_pct > 0:
            return AdvisorFlag.ACCUMULATION_ZONE
        elif vol_signal == VolumeSignal.ABOVE_AVERAGE and change_pct < -1.0:
            return AdvisorFlag.SELLING_PRESSURE

    # ── Rule 4: Upper/middle range with above-average volume and positive move ──
    if range_pos in (RangePosition.UPPER, RangePosition.MIDDLE):
        if vol_signal == VolumeSignal.ABOVE_AVERAGE and change_pct > 0:
            return AdvisorFlag.MOMENTUM_BUILDING
        elif vol_signal == VolumeSignal.ABOVE_AVERAGE and change_pct < -1.0:
            return AdvisorFlag.SELLING_PRESSURE

    # ── Rule 5: Low volume or flat price → consolidation ──
    if vol_signal == VolumeSignal.BELOW_AVERAGE:
        return AdvisorFlag.CONSOLIDATION

    if vol_signal == VolumeSignal.NORMAL and -0.5 <= change_pct <= 0.5:
        return AdvisorFlag.CONSOLIDATION

    # ── Default: No clear signal ──
    return AdvisorFlag.NEUTRAL


def _explain_flag_decision(
    flag: AdvisorFlag,
    range_pos: RangePosition,
    vol_signal: VolumeSignal,
    change_pct: float,
) -> str:
    """Generate a plain English explanation of why this flag was chosen.

    Used as Step 4 in the CoT reasoning. Must be understandable
    by a non-technical advisor — no jargon, no abbreviations.

    Args:
        flag: The decided AdvisorFlag.
        range_pos: 52-week range position.
        vol_signal: Volume signal classification.
        change_pct: Daily percentage change.

    Returns:
        One-sentence explanation of the signal decision.
    """
    explanations: dict[AdvisorFlag, str] = {
        AdvisorFlag.BREAKOUT_WATCH: (
            f"Price {range_pos.value} in 52w range with {vol_signal.value} volume "
            f"and {change_pct:+.2f}% move — potential breakout setup. "
            f"Watch for confirmation above resistance."
        ),
        AdvisorFlag.ACCUMULATION_ZONE: (
            f"Price in {range_pos.value} 52w range but volume is {vol_signal.value} "
            f"with positive change ({change_pct:+.2f}%) — suggests institutional buying. "
            f"Watchlist candidate for next 3-5 sessions."
        ),
        AdvisorFlag.UNUSUAL_ACTIVITY: (
            f"Volume spike ({vol_signal.value}) without strong directional move "
            f"({change_pct:+.2f}%) — unusual activity detected. "
            f"Could be block deal, institutional repositioning, or news-driven. Investigate."
        ),
        AdvisorFlag.SELLING_PRESSURE: (
            f"Negative price action ({change_pct:+.2f}%) with {vol_signal.value} volume "
            f"at {range_pos.value} range — distribution/selling pressure detected. "
            f"Caution advised. Review stop-loss levels."
        ),
        AdvisorFlag.DISTRIBUTION_ZONE: (
            f"Price near 52w high ({range_pos.value}) but falling ({change_pct:+.2f}%) "
            f"on {vol_signal.value} volume — smart money may be exiting. "
            f"Reduce exposure or tighten trailing stops."
        ),
        AdvisorFlag.MOMENTUM_BUILDING: (
            f"Positive move ({change_pct:+.2f}%) at {range_pos.value} range "
            f"with {vol_signal.value} volume — progressive momentum building. "
            f"Watch for higher highs and higher lows pattern."
        ),
        AdvisorFlag.CONSOLIDATION: (
            f"Price moving sideways ({change_pct:+.2f}%) with {vol_signal.value} volume "
            f"at {range_pos.value} range — consolidation phase. "
            f"Wait for direction before entering."
        ),
        AdvisorFlag.NEUTRAL: (
            f"No clear actionable signal. Price at {range_pos.value} range, "
            f"volume {vol_signal.value}, change {change_pct:+.2f}%. "
            f"Monitor but no immediate action warranted."
        ),
    }

    return explanations.get(flag, f"Signal: {flag.value}.")


def calculate_all_signals(
    stocks: list[StockData],
    enable_cot: bool = True,
) -> list[StockData]:
    """Calculate advisor signals for all stocks and sort by signal strength.

    This is the primary entry point called by the pipeline (Step 6).
    Processes each stock through the 5-step CoT pattern, sets the
    advisor_flag and cot_reasoning, then sorts stocks by signal
    strength (most actionable first).

    Sorting by signal strength ensures that when MarketData.trim_to_budget()
    trims to top 10 or top 5 stocks, the most interesting ones survive.

    Args:
        stocks: List of StockData from stock_fetcher, enriched with
                volume analysis and 52w range data.
        enable_cot: Whether to generate full CoT reasoning strings.
                    Disable to save tokens when CoT is not needed.

    Returns:
        List of StockData with advisor_flag and cot_reasoning set,
        sorted by signal strength descending (breakout_watch first,
        neutral last).
    """
    if not stocks:
        logger.warning(
            "No stocks provided for signal calculation. "
            "Returning empty list — advisor has no stocks to analyze."
        )
        return []

    signaled_stocks: list[StockData] = []

    for stock in stocks:
        advisor_flag, cot_reasoning = calculate_advisor_flag(stock)
        stock.advisor_flag = advisor_flag
        stock.cot_reasoning = cot_reasoning if enable_cot else None
        signaled_stocks.append(stock)

    # Sort by signal strength (highest first)
    signaled_stocks.sort(
        key=lambda s: SIGNAL_STRENGTH.get(s.advisor_flag, 0)
        if s.advisor_flag
        else 0,
        reverse=True,
    )

    # Log signal distribution
    signal_counts: dict[str, int] = {}
    for stock in signaled_stocks:
        flag_name = stock.advisor_flag.value if stock.advisor_flag else "none"
        signal_counts[flag_name] = signal_counts.get(flag_name, 0) + 1

    distribution = ", ".join(
        f"{flag}: {count}" for flag, count in sorted(signal_counts.items())
    )
    logger.info(
        f"Advisor signals calculated for {len(signaled_stocks)} stocks. "
        f"Distribution: {distribution}. "
        f"Top signal: {signaled_stocks[0].ticker} → "
        f"{signaled_stocks[0].advisor_flag.value if signaled_stocks[0].advisor_flag else 'none'}."
    )

    return signaled_stocks
