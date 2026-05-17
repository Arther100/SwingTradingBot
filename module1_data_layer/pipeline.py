"""
SwingAdvisorBot — Module 1: Data Layer
pipeline.py — Main orchestrator for the data collection pipeline

This is the brain of Module 1. Every market data request flows through
run_data_pipeline(), which executes a 9-step Chain of Thought process
to assemble a complete, validated, signal-rich MarketData object.

The DataCollectorAgent (File 07) calls run_data_pipeline() from its
execute() method. The pipeline coordinates all fetchers, applies signals,
runs health checks, and trims to token budget — producing the single
MarketData object that Module 2 (Claude AI) reasons about.

9-Step CoT Pipeline:
  Step 1: Determine market status from current IST time.
  Step 2: Check cache for fresh MarketData (return early if valid).
  Step 3: Fetch stock data via Kite Connect with rate limiting.
  Step 4: Fetch and score news via NewsAPI + news_scorer.
  Step 5: Fetch VIX, sectors, and economic data (parallel where possible).
  Step 6: Calculate advisor signals for all stocks.
  Step 7: Run 7-step pipeline health check (self-reflection).
  Step 8: Trim MarketData to 2500 token budget.
  Step 9: Cache and return validated MarketData.

7-Step Health Check (Self-Reflection):
  Step 1: Count stocks fetched (need >= 10 of 15 requested).
  Step 2: Verify all advisor_flags are set (not None).
  Step 3: Verify all timestamps are IST.
  Step 4: Verify is_real_data is True.
  Step 5: Estimate token count (must be <= 2500).
  Step 6: Verify market_status matches current IST time.
  Step 7: Generate PipelineHealthReport.

Resilience model:
  - VIX is mandatory. If VIX fails → DataFetchError (abort).
  - Stocks are mandatory. If < 10 stocks fetched → PipelineStatus.DEGRADED.
  - News, sectors, economic data are optional. Failures → logged, empty list.
  - Token budget is mandatory. If over-budget after 5 trim steps → TokenBudgetError.
  - Health check determines final pipeline_status (healthy / degraded / failed).

Data flow:
  DataCollectorAgent.execute()
    → run_data_pipeline(tickers, config, agent)
      → fetchers (parallel where possible)
      → signals/advisor_signals.py
      → signals/news_scorer.py
      → health check (self-reflection)
      → trim_to_budget()
    → MarketData (complete, validated, within budget)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time

from zoneinfo import ZoneInfo

from module1_data_layer.cache import cache
from module1_data_layer.config import (
    NSE_HOLIDAYS_2026,
    NSE_MARKET_CLOSE,
    NSE_MARKET_OPEN,
    NSE_POST_MARKET_CLOSE,
    NSE_PRE_MARKET_OPEN,
    DataFetchConfig,
)
from module1_data_layer.models import (
    DataFetchError,
    DataFreshness,
    MarketData,
    MarketStatus,
    PipelineHealthReport,
    PipelineHealthError,
    PipelineStatus,
    TokenBudgetError,
)

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.pipeline")


# ─────────────────────────────────────────────────────────────
# Step 1: Market Status Determination
# ─────────────────────────────────────────────────────────────


def determine_market_status() -> tuple[MarketStatus, str]:
    """Determine current NSE market status from IST time.

    Used by:
      - run_data_pipeline() Step 1
      - vix_fetcher.fetch_market_status_data() (MCP tool)

    NSE schedule:
      9:00–9:15 IST    → PRE_MARKET (opening auction session)
      9:15–15:30 IST   → OPEN (regular trading session)
      Otherwise         → CLOSED

    Also checks:
      - Weekends (Saturday/Sunday) → CLOSED
      - NSE holidays (NSE_HOLIDAYS_2026) → CLOSED

    Returns:
        Tuple of (MarketStatus, human_readable_reason).
        The reason string goes into MarketData.market_status_reason
        so the advisor knows WHY the market is in that state.
    """
    now = datetime.now(IST)
    current_time = now.time()
    current_date = now.date()

    # Weekend check
    weekday = current_date.weekday()
    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return (
            MarketStatus.CLOSED,
            f"NSE is closed — {day_name}. Next trading session opens Monday 9:15 IST.",
        )

    # Holiday check
    date_str = current_date.isoformat()
    if date_str in NSE_HOLIDAYS_2026:
        return (
            MarketStatus.CLOSED,
            f"NSE is closed — market holiday on {date_str}. "
            f"Check NSE website for the next trading day.",
        )

    # Pre-market session: 9:00 – 9:15 IST
    if NSE_PRE_MARKET_OPEN <= current_time < NSE_MARKET_OPEN:
        return (
            MarketStatus.PRE_MARKET,
            f"NSE pre-market session is active ({NSE_PRE_MARKET_OPEN.strftime('%H:%M')}–"
            f"{NSE_MARKET_OPEN.strftime('%H:%M')} IST). "
            f"Opening auction in progress. Regular trading starts at 9:15 IST.",
        )

    # Regular trading session: 9:15 – 15:30 IST
    if NSE_MARKET_OPEN <= current_time < NSE_MARKET_CLOSE:
        return (
            MarketStatus.OPEN,
            f"NSE is open for regular trading ({NSE_MARKET_OPEN.strftime('%H:%M')}–"
            f"{NSE_MARKET_CLOSE.strftime('%H:%M')} IST). Live prices available.",
        )

    # Before pre-market
    if current_time < NSE_PRE_MARKET_OPEN:
        return (
            MarketStatus.CLOSED,
            f"NSE has not opened yet today. "
            f"Pre-market starts at {NSE_PRE_MARKET_OPEN.strftime('%H:%M')} IST, "
            f"regular trading at {NSE_MARKET_OPEN.strftime('%H:%M')} IST.",
        )

    # After market close
    return (
        MarketStatus.CLOSED,
        f"NSE regular trading ended at {NSE_MARKET_CLOSE.strftime('%H:%M')} IST. "
        f"Showing end-of-day prices.",
    )


def _determine_data_freshness(
    market_status: MarketStatus,
    fetch_started_at: datetime,
) -> DataFreshness:
    """Determine data freshness classification.

    Freshness depends on both market state and how old the data is:
      Market open + fetched < 3 min ago  → REAL_TIME
      Market open + fetched < 15 min ago → DELAYED
      Market open + fetched ≥ 15 min ago → STALE
      Market closed or pre-market        → END_OF_DAY

    Args:
        market_status: Current market state.
        fetch_started_at: When this pipeline run started fetching.

    Returns:
        DataFreshness classification.
    """
    if market_status in (MarketStatus.CLOSED, MarketStatus.PRE_MARKET):
        return DataFreshness.END_OF_DAY

    now = datetime.now(IST)
    age_seconds = (now - fetch_started_at).total_seconds()

    if age_seconds < 180:
        return DataFreshness.REAL_TIME
    elif age_seconds < 900:
        return DataFreshness.DELAYED
    else:
        return DataFreshness.STALE


# ─────────────────────────────────────────────────────────────
# Step 7: Pipeline Health Check (7-Step Self-Reflection)
# ─────────────────────────────────────────────────────────────


def _run_health_check(
    market_data: MarketData,
    market_status: MarketStatus,
    config: DataFetchConfig,
) -> PipelineHealthReport:
    """Run the 7-step self-reflection health check on assembled MarketData.

    The pipeline doesn't blindly hand data to the advisor. It first
    checks its own work. Like a senior analyst reviewing their report
    before sending it to the portfolio manager.

    Steps:
      1. Count stocks (>= 10 expected)
      2. Verify all advisor_flags set
      3. Verify all timestamps IST
      4. Verify is_real_data is True
      5. Estimate token count (<= budget)
      6. Verify market_status matches IST time
      7. Generate report

    If any critical check fails, pipeline_status → DEGRADED or FAILED.
    Non-critical issues are logged but don't block the pipeline.

    Args:
        market_data: The assembled MarketData to check.
        market_status: Expected market status from Step 1.
        config: DataFetchConfig with token_budget.

    Returns:
        PipelineHealthReport with detailed check results.
    """
    issues: list[str] = []

    # ── Check 1: Stock count ──
    stocks_fetched = len(market_data.stocks)
    stocks_ok = stocks_fetched >= 10
    if not stocks_ok:
        issues.append(
            f"Only {stocks_fetched} stocks fetched (expected >= 10). "
            f"Some tickers may have failed to fetch from Kite Connect."
        )

    # ── Check 2: Advisor flags ──
    stocks_without_flags = [
        s.ticker for s in market_data.stocks if s.advisor_flag is None
    ]
    all_signals_set = len(stocks_without_flags) == 0
    if not all_signals_set:
        issues.append(
            f"Missing advisor_flag on {len(stocks_without_flags)} stocks: "
            f"{', '.join(stocks_without_flags[:5])}."
        )

    # ── Check 3: Timestamps IST ──
    all_timestamps_ist = True
    for stock in market_data.stocks:
        if stock.last_updated.tzinfo is None:
            all_timestamps_ist = False
            issues.append(
                f"Stock {stock.ticker} has timezone-naive last_updated timestamp."
            )
            break
        tz_name = str(stock.last_updated.tzinfo)
        if "Asia/Kolkata" not in tz_name and "IST" not in tz_name:
            all_timestamps_ist = False
            issues.append(
                f"Stock {stock.ticker} has non-IST timezone: {tz_name}."
            )
            break

    for news in market_data.news:
        if news.published_at.tzinfo is None:
            all_timestamps_ist = False
            issues.append("News item has timezone-naive published_at timestamp.")
            break

    if market_data.timestamp.tzinfo is None:
        all_timestamps_ist = False
        issues.append("MarketData.timestamp is timezone-naive.")

    # ── Check 4: Real data ──
    all_real_data = market_data.is_real_data
    if not all_real_data:
        issues.append(
            "is_real_data is False — SwingAdvisorBot never operates on mock data."
        )

    # ── Check 5: Token budget ──
    token_estimate = market_data.estimate_tokens()
    token_within_budget = token_estimate <= config.token_budget
    if not token_within_budget:
        issues.append(
            f"Token estimate {token_estimate} exceeds budget {config.token_budget}. "
            f"Trimming required."
        )

    # ── Check 6: Market status consistency ──
    # Re-check market status to ensure it hasn't changed between Step 1 and Step 7
    current_status, _ = determine_market_status()
    market_status_correct = market_data.market_status == current_status
    if not market_status_correct:
        # Market status may have changed during pipeline execution
        # (e.g., pipeline started at 15:29 and health check runs at 15:31).
        # This is informational, not a failure.
        issues.append(
            f"Market status changed during pipeline: "
            f"started as '{market_status.value}', "
            f"now '{current_status.value}'. Timestamp boundary crossed."
        )

    # ── Check 7: Generate report ──
    # Determine overall pipeline status
    critical_issues = (
        not all_real_data  # Non-negotiable
        or not market_data.india_vix > 0  # VIX is mandatory
    )
    degraded_issues = (
        not stocks_ok
        or not all_signals_set
        or not token_within_budget
    )

    if critical_issues:
        status = PipelineStatus.FAILED
    elif degraded_issues:
        status = PipelineStatus.DEGRADED
    else:
        status = PipelineStatus.HEALTHY

    report = PipelineHealthReport(
        status=status,
        stocks_fetched=stocks_fetched,
        news_fetched=len(market_data.news),
        vix_available=market_data.india_vix > 0,
        all_signals_set=all_signals_set,
        all_timestamps_ist=all_timestamps_ist,
        all_real_data=all_real_data,
        token_estimate=token_estimate,
        token_within_budget=token_within_budget,
        market_status_correct=market_status_correct,
        issues=issues,
    )

    logger.info(
        f"Health check complete: status={status.value}, "
        f"stocks={stocks_fetched}, news={len(market_data.news)}, "
        f"VIX={'available' if report.vix_available else 'MISSING'}, "
        f"tokens={token_estimate}/{config.token_budget}, "
        f"issues={len(issues)}."
    )

    return report


def _generate_morning_signal(market_data: MarketData) -> str:
    """Generate the advisor_morning_signal for MarketData.

    This is the 2-3 sentence market summary that Module 2's AI advisor
    uses as the opening context when speaking to the user. It must be
    informative enough to start a conversation about today's market
    without sounding generic or hollow.

    Constructs the signal from:
      - Market status (open/closed/pre-market)
      - VIX level and fear signal
      - Nifty/Sensex direction
      - Top stock signal (if breakout_watch or selling_pressure)
      - Top news headline (if high relevance)

    Args:
        market_data: Assembled MarketData with all fields populated.

    Returns:
        2-3 sentence plain English market summary string.
    """
    parts: list[str] = []

    # Market state + VIX context
    if market_data.market_status == MarketStatus.OPEN:
        vix_desc = {
            "low_fear": "calm",
            "moderate_fear": "moderate",
            "high_fear": "elevated",
            "extreme_fear": "under extreme stress",
        }.get(market_data.vix_signal.value, "unknown")

        nifty_direction = "up" if market_data.nifty50_change_pct >= 0 else "down"
        parts.append(
            f"Markets are open with Nifty {nifty_direction} "
            f"{abs(market_data.nifty50_change_pct):.1f}% and "
            f"VIX at {market_data.india_vix:.1f} ({vix_desc})."
        )
    elif market_data.market_status == MarketStatus.PRE_MARKET:
        parts.append(
            f"Pre-market session is active. "
            f"VIX at {market_data.india_vix:.1f} ({market_data.vix_signal.value.replace('_', ' ')})."
        )
    else:
        parts.append(
            f"Markets are closed. "
            f"Nifty ended at {market_data.nifty50_value:,.0f} "
            f"({'+' if market_data.nifty50_change_pct >= 0 else ''}"
            f"{market_data.nifty50_change_pct:.1f}%), "
            f"VIX at {market_data.india_vix:.1f}."
        )

    # Top stock signal
    if market_data.stocks:
        top_stock = market_data.stocks[0]
        if top_stock.advisor_flag and top_stock.advisor_flag.value != "neutral":
            flag_readable = top_stock.advisor_flag.value.replace("_", " ")
            parts.append(
                f"{top_stock.ticker} showing {flag_readable} "
                f"({'+' if top_stock.change_pct >= 0 else ''}"
                f"{top_stock.change_pct:.1f}%, "
                f"vol {top_stock.volume_ratio:.1f}x avg)."
            )

    # Top news
    if market_data.news:
        top_news = market_data.news[0]
        if top_news.relevance_score >= 0.80:
            parts.append(
                f"Key news: \"{top_news.headline[:80]}\" "
                f"({top_news.sentiment.value}, "
                f"impact: {top_news.market_impact.value})."
            )

    return " ".join(parts) if parts else "Market data assembled. Review stocks and news for details."


# ─────────────────────────────────────────────────────────────
# Main Pipeline Orchestrator (9-Step CoT)
# ─────────────────────────────────────────────────────────────


async def run_data_pipeline(
    tickers: list[str],
    config: DataFetchConfig,
    agent: object | None = None,
) -> MarketData:
    """Execute the complete 9-step data collection pipeline.

    This is the main entry point called by DataCollectorAgent.execute().
    Orchestrates all fetchers, applies signals, runs health checks,
    and trims to token budget.

    The pipeline is designed for resilience:
      - VIX is mandatory → failure aborts the pipeline.
      - Stocks are mandatory → < 10 stocks → PipelineStatus.DEGRADED.
      - News is optional → failure → empty list, logged.
      - Sectors are optional → failure → empty list, logged.
      - Economic events are optional → failure → empty list, logged.

    Args:
        tickers: NSE ticker symbols to fetch.
        config: DataFetchConfig controlling fetch limits and budget.
        agent: Optional SwingAdvisorBaseAgent for CoT logging.
               Passed from DataCollectorAgent.execute().

    Returns:
        MarketData: Complete, validated, within token budget.

    Raises:
        DataFetchError: If a mandatory data source fails (VIX).
        PipelineHealthError: If health check finds critical failures.
        TokenBudgetError: If data exceeds budget after all trim steps.
    """
    fetch_started_at = datetime.now(IST)
    pipeline_cache_key = f"pipeline:full:{'|'.join(sorted(tickers[:15]))}"

    def _log(step: int, thought: str) -> None:
        """Log CoT step to both logger and agent (if available)."""
        logger.info(f"Pipeline Step {step}: {thought}")
        if agent and hasattr(agent, "log_reasoning"):
            agent.log_reasoning(step=step, thought=thought)

    # ── Step 1: Determine market status ──
    market_status, status_reason = determine_market_status()
    _log(
        step=1,
        thought=(
            f"Market status: {market_status.value}. {status_reason} "
            f"Time: {fetch_started_at.strftime('%H:%M:%S IST')}."
        ),
    )

    # ── Step 2: Check cache for fresh MarketData ──
    cached_data = cache.get(pipeline_cache_key)
    if cached_data is not None:
        cache_age = cache.get_age(pipeline_cache_key)
        _log(
            step=2,
            thought=(
                f"Fresh MarketData found in cache (age: {cache_age:.0f}s). "
                f"Returning cached data — no API calls needed."
            ),
        )
        return cached_data

    _log(
        step=2,
        thought="No fresh cache available. Proceeding with full data fetch.",
    )

    # ── Step 3: Fetch stock data ──
    from module1_data_layer.fetchers.stock_fetcher import fetch_stocks

    stocks = []
    try:
        stocks = await fetch_stocks(tickers, config)
        _log(
            step=3,
            thought=(
                f"Fetched {len(stocks)} stocks successfully. "
                f"Tickers: {', '.join(s.ticker for s in stocks[:5])}"
                f"{'...' if len(stocks) > 5 else ''}."
            ),
        )
    except DataFetchError as e:
        _log(
            step=3,
            thought=(
                f"Stock fetch failed: {e}. "
                f"Pipeline will continue but will be degraded."
            ),
        )

    # ── Step 4: Fetch and score news ──
    from module1_data_layer.fetchers.news_fetcher import fetch_news
    from module1_data_layer.signals.news_scorer import score_news_items

    scored_news = []
    try:
        raw_news = await fetch_news(config)
        if raw_news:
            scored_news = score_news_items(raw_news)
            # Filter by minimum relevance
            scored_news = [
                item
                for item in scored_news
                if item.relevance_score >= config.min_news_relevance
            ]
            # Cap at max_news
            scored_news = scored_news[: config.max_news]

        _log(
            step=4,
            thought=(
                f"Fetched {len(raw_news)} raw news items, "
                f"scored and filtered to {len(scored_news)} relevant items "
                f"(threshold: {config.min_news_relevance})."
            ),
        )
    except DataFetchError as e:
        _log(
            step=4,
            thought=(
                f"News fetch failed: {e}. "
                f"Continuing without news — Priority 4, non-critical."
            ),
        )

    # ── Step 5: Fetch VIX, sectors, and economic data ──
    # VIX is mandatory. Sectors and economic data are optional.
    # Fetch VIX first (mandatory), then sectors + economic in parallel.
    from module1_data_layer.fetchers.vix_fetcher import fetch_vix_and_indices
    from module1_data_layer.fetchers.sector_fetcher import fetch_sectors
    from module1_data_layer.fetchers.economic_fetcher import fetch_economic_events

    # VIX — mandatory
    vix_data = await fetch_vix_and_indices(config)
    _log(
        step=5,
        thought=(
            f"VIX fetched: {vix_data.india_vix:.2f} ({vix_data.vix_signal.value}). "
            f"Nifty: {vix_data.nifty50_value:,.0f} ({vix_data.nifty50_change_pct:+.1f}%). "
            f"Sensex: {vix_data.sensex_value:,.0f} ({vix_data.sensex_change_pct:+.1f}%)."
        ),
    )

    # Sectors + Economic — optional, fetch in parallel
    sectors = []
    economic_events = []

    async def _fetch_sectors_safe() -> list:
        try:
            return await fetch_sectors(config, stocks)
        except Exception as e:
            logger.warning(f"Sector fetch failed (non-critical): {e}")
            return []

    async def _fetch_economic_safe() -> list:
        try:
            return await fetch_economic_events(config)
        except DataFetchError as e:
            logger.warning(f"Economic fetch failed (non-critical): {e}")
            return []

    sector_result, economic_result = await asyncio.gather(
        _fetch_sectors_safe(),
        _fetch_economic_safe(),
    )
    sectors = sector_result
    economic_events = economic_result

    _log(
        step=5,
        thought=(
            f"Parallel fetch complete. "
            f"Sectors: {len(sectors)} fetched. "
            f"Economic events: {len(economic_events)} fetched."
        ),
    )

    # ── Step 6: Calculate advisor signals ──
    from module1_data_layer.signals.advisor_signals import calculate_all_signals

    if stocks:
        stocks = calculate_all_signals(stocks, enable_cot=config.enable_cot_reasoning)
        _log(
            step=6,
            thought=(
                f"Advisor signals calculated for {len(stocks)} stocks. "
                f"Top signal: {stocks[0].ticker} → "
                f"{stocks[0].advisor_flag.value if stocks[0].advisor_flag else 'none'}."
            ),
        )
    else:
        _log(
            step=6,
            thought="No stocks available for signal calculation. Skipping Step 6.",
        )

    # ── Assemble MarketData ──
    data_freshness = _determine_data_freshness(market_status, fetch_started_at)

    market_data = MarketData(
        market_status=market_status,
        market_status_reason=status_reason,
        data_freshness=data_freshness,
        nifty50_value=vix_data.nifty50_value,
        nifty50_change_pct=vix_data.nifty50_change_pct,
        sensex_value=vix_data.sensex_value,
        sensex_change_pct=vix_data.sensex_change_pct,
        india_vix=vix_data.india_vix,
        vix_signal=vix_data.vix_signal,
        stocks=stocks,
        news=scored_news,
        sectors=sectors,
        economic_events=economic_events,
        is_real_data=True,
        timestamp=datetime.now(IST),
    )

    # Generate morning signal from assembled data
    market_data.advisor_morning_signal = _generate_morning_signal(market_data)

    # ── Step 7: Health check (self-reflection) ──
    health_report = _run_health_check(market_data, market_status, config)
    market_data.pipeline_status = health_report.status
    market_data.pipeline_health_report = health_report

    _log(
        step=7,
        thought=(
            f"Health check: {health_report.status.value}. "
            f"Stocks: {health_report.stocks_fetched}, "
            f"VIX: {'OK' if health_report.vix_available else 'MISSING'}, "
            f"Signals: {'all set' if health_report.all_signals_set else 'INCOMPLETE'}, "
            f"Tokens: {health_report.token_estimate}/{config.token_budget}, "
            f"Issues: {len(health_report.issues)}."
        ),
    )

    if health_report.status == PipelineStatus.FAILED:
        raise PipelineHealthError(
            step=7,
            reason=(
                f"Pipeline health check failed with {len(health_report.issues)} issue(s): "
                f"{'; '.join(health_report.issues)}"
            ),
        )

    # ── Step 8: Trim to token budget ──
    pre_trim_tokens = market_data.estimate_tokens()
    market_data.trim_to_budget(config.token_budget)
    post_trim_tokens = market_data.estimate_tokens()

    if pre_trim_tokens != post_trim_tokens:
        _log(
            step=8,
            thought=(
                f"Token budget trimming applied. "
                f"Before: {pre_trim_tokens} tokens → After: {post_trim_tokens} tokens. "
                f"Budget: {config.token_budget} tokens. "
                f"Stocks after trim: {len(market_data.stocks)}, "
                f"News after trim: {len(market_data.news)}."
            ),
        )
    else:
        _log(
            step=8,
            thought=(
                f"Data within budget ({post_trim_tokens}/{config.token_budget} tokens). "
                f"No trimming needed."
            ),
        )

    # ── Step 9: Cache and return ──
    # Cache the full MarketData for the shorter of stock TTL or VIX TTL
    cache_ttl = min(config.cache_ttl_stocks, config.cache_ttl_vix)
    cache.set(pipeline_cache_key, market_data, ttl=cache_ttl)

    _log(
        step=9,
        thought=(
            f"Pipeline complete. MarketData assembled with "
            f"{len(market_data.stocks)} stocks, {len(market_data.news)} news, "
            f"{len(market_data.sectors)} sectors, {len(market_data.economic_events)} economic events. "
            f"VIX={market_data.india_vix:.1f}, "
            f"status={market_data.market_status.value}, "
            f"freshness={market_data.data_freshness.value}, "
            f"pipeline={market_data.pipeline_status.value}. "
            f"Cached for {cache_ttl}s."
        ),
    )

    return market_data
