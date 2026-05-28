"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
backtest_engine.py — Main orchestrator for the full backtest pipeline

This is the single entry point for running the backtesting engine.
All other modules are orchestrated from here.

Full pipeline (run_full_backtest):
  1. Resolve tickers and date range
  2. Fetch OHLCV data for all tickers concurrently (Kite → SQLite cache)
  3. Fetch Nifty 50 benchmark data
  4. For each ticker × signal_type pair:
       a. Replay signals (SignalReplayer — no look-ahead)
       b. Simulate trades (TradeSimulator — entry, target, stop, costs)
       c. Compute metrics (MetricsCalculator — win_rate, PF, Sharpe, etc.)
       d. Run walk-forward (WalkForwardEngine — in/out-of-sample split)
       e. Build BacktestResult
  5. Aggregate all trade simulations → portfolio PerformanceMetrics
  6. Determine best/worst ticker by return
  7. Build PortfolioBacktestResult
  8. Generate Claude advisor report (ReportGenerator)
  9. Update M4 signal weights (WeightUpdater → SQLite)
  10. Cache portfolio result (BacktestResultCache → SQLite)
  11. Return PortfolioBacktestResult

Key design decisions:
  - Tickers fetched concurrently (asyncio.gather) with rate-limit semaphore
  - Each ticker's data is processed independently — failure on one ticker
    logs a warning but does not abort the full run
  - signal_results list contains one BacktestResult per (ticker, signal_type)
  - Portfolio metrics are computed from ALL trade simulations aggregated
  - ending_capital = starting_capital + sum(all net_pnl)

Usage:
    engine = BacktestEngine()
    result = await engine.run_full_backtest(
        tickers=["HDFCBANK", "RELIANCE", "TCS"],
        period_months=12,
    )
    print(f"Win rate: {result.metrics.win_rate:.1f}%")
    print(f"Alpha: {result.metrics.alpha:+.1f}%")
    print(f"Verdict: {result.advisor_verdict}")
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

from zoneinfo import ZoneInfo

from module_backtest.config import (
    BACKTEST_DEFAULT_TICKERS,
    BACKTEST_FROM_DATE,
    BACKTEST_TO_DATE,
    BACKTEST_UNIVERSE,
    IN_SAMPLE_END_DATE,
    NSE_STOCK_UNIVERSE,
    STARTING_CAPITAL,
)
from module_backtest.data.historical_fetcher import historical_fetcher
from module_backtest.engine.metrics_calculator import MetricsCalculator
from module_backtest.engine.signal_replayer import SignalReplayer
from module_backtest.engine.trade_simulator import TradeSimulator
from module_backtest.engine.walk_forward import WalkForwardEngine
from module_backtest.models import (
    AdvisorVerdict,
    BacktestConfig,
    BacktestResult,
    ExitReason,
    OHLCVBar,
    PerformanceMetrics,
    PortfolioBacktestResult,
    SignalType,
    TradeSimulation,
)
from module_backtest.reporting.report_generator import ReportGenerator
from module_backtest.reporting.weight_updater import WeightUpdater

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("swing_advisor.backtest.engine")

_ALL_SIGNAL_TYPES = [
    SignalType.BREAKOUT_WATCH,
    SignalType.ACCUMULATION_ZONE,
    SignalType.UNUSUAL_ACTIVITY,
    SignalType.FII_BUYING,
]


class BacktestEngine:
    """Full backtesting pipeline orchestrator.

    Usage:
        engine = BacktestEngine()
        result = await engine.run_full_backtest(
            tickers=["HDFCBANK", "RELIANCE", "TCS"],
            period_months=12,
        )
    """

    def __init__(
        self,
        starting_capital: Decimal = STARTING_CAPITAL,
        generate_report: bool = True,
        update_weights: bool = True,
    ) -> None:
        self._starting_capital = starting_capital
        self._generate_report = generate_report
        self._update_weights = update_weights
        self._calculator = MetricsCalculator(starting_capital=starting_capital)
        self._simulator = TradeSimulator()
        self._weight_updater = WeightUpdater()
        self._report_generator = ReportGenerator()

    # ── Public API ──────────────────────────────────────────

    async def run_full_backtest(
        self,
        tickers: Optional[list[str]] = None,
        period_months: int = 12,
        signal_types: Optional[list[SignalType]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> PortfolioBacktestResult:
        """Run the full backtesting pipeline.

        Args:
            tickers:       NSE tickers to backtest. Defaults to BACKTEST_DEFAULT_TICKERS.
            period_months: Backtest period in months from today (used if from_date/to_date
                           not supplied). Default 12.
            signal_types:  Signal types to evaluate. Defaults to all 4.
            from_date:     Override start date.
            to_date:       Override end date (defaults to today).

        Returns:
            PortfolioBacktestResult with all metrics, advisor narrative, and weights.
        """
        # If no tickers supplied, run on the validated 52-stock BACKTEST_UNIVERSE
        if tickers is None:
            tickers = BACKTEST_UNIVERSE
            logger.info(
                f"[BacktestEngine] No tickers specified — using "
                f"BACKTEST_UNIVERSE ({len(tickers)} stocks, 3 excluded)"
            )
        signal_types = signal_types or _ALL_SIGNAL_TYPES

        # Resolve date range
        actual_to = to_date or BACKTEST_TO_DATE
        if from_date:
            actual_from = from_date
        else:
            actual_from = date(actual_to.year, actual_to.month, actual_to.day)
            # Subtract period_months
            month = actual_from.month - period_months
            year = actual_from.year
            while month <= 0:
                month += 12
                year -= 1
            actual_from = actual_from.replace(year=year, month=month)

        period_str = f"{actual_from.strftime('%b %Y')} – {actual_to.strftime('%b %Y')}"

        logger.info(
            f"BacktestEngine starting — {len(tickers)} tickers, "
            f"{len(signal_types)} signal types, {period_str}"
        )

        # Build BacktestConfig
        config = BacktestConfig(
            tickers=tickers,
            signal_types=signal_types,
            from_date=actual_from,
            to_date=actual_to,
            in_sample_end_date=IN_SAMPLE_END_DATE if actual_from <= IN_SAMPLE_END_DATE <= actual_to else None,
            starting_capital=self._starting_capital,
        )

        # ── Step 1: Fetch all OHLCV data concurrently ────────
        logger.info(f"Fetching OHLCV data for {len(tickers)} tickers + Nifty...")
        bar_map, nifty_bars = await self._fetch_all_data(tickers, actual_from, actual_to)

        if not bar_map:
            logger.error("No OHLCV data fetched — aborting backtest")
            return self._empty_result(tickers, actual_from, actual_to)

        logger.info(f"Data fetched: {len(bar_map)} tickers with data")

        # ── Step 2: Per-ticker, per-signal simulation ─────────
        all_trades: list[TradeSimulation] = []
        signal_results: list[BacktestResult] = []
        ticker_pnl: dict[str, Decimal] = {}

        for ticker in tickers:
            bars = bar_map.get(ticker)
            if not bars:
                logger.warning(f"[{ticker}] No bars available — skipping")
                continue

            try:
                ticker_trades, ticker_results = self._run_ticker(
                    ticker, bars, signal_types, config, nifty_bars
                )
                all_trades.extend(ticker_trades)
                signal_results.extend(ticker_results)
                ticker_pnl[ticker] = sum(
                    t.net_pnl for t in ticker_trades
                    if t.exit_reason != ExitReason.NEVER_ENTERED
                )
            except Exception as exc:
                logger.error(f"[{ticker}] Simulation error — skipping: {exc}", exc_info=True)

        if not all_trades:
            logger.warning("No trades generated across all tickers")
            return self._empty_result(tickers, actual_from, actual_to)

        # ── Step 3: Portfolio-level metrics ───────────────────
        portfolio_metrics = self._calculator.compute(all_trades)

        # ── Step 4: Best/worst ticker by cumulative P&L ───────
        best_ticker: Optional[str] = None
        worst_ticker: Optional[str] = None
        if ticker_pnl:
            best_ticker = max(ticker_pnl, key=lambda k: ticker_pnl[k])
            worst_ticker = min(ticker_pnl, key=lambda k: ticker_pnl[k])

        # ── Step 5: Advisor verdict ───────────────────────────
        verdict, verdict_note = self._calculator.portfolio_verdict(portfolio_metrics)

        # ── Step 6: Ending capital ────────────────────────────
        total_net_pnl = sum(
            t.net_pnl for t in all_trades
            if t.exit_reason != ExitReason.NEVER_ENTERED
        )
        ending_capital = (self._starting_capital + total_net_pnl).quantize(
            Decimal("0.01")
        )

        # ── Step 7: Build portfolio result ────────────────────
        portfolio_result = PortfolioBacktestResult(
            period_start=actual_from,
            period_end=actual_to,
            starting_capital=self._starting_capital,
            ending_capital=ending_capital,
            metrics=portfolio_metrics,
            signal_results=signal_results,
            tickers_tested=list(bar_map.keys()),
            best_ticker=best_ticker,
            worst_ticker=worst_ticker,
            advisor_verdict=verdict,
            advisor_note=verdict_note,
        )

        logger.info(
            f"Portfolio backtest complete — "
            f"{portfolio_metrics.total_trades} trades, "
            f"win_rate={portfolio_metrics.win_rate:.1f}%, "
            f"PF={portfolio_metrics.profit_factor:.2f}, "
            f"verdict={verdict.value}"
        )

        # ── Step 8: Claude advisor report ─────────────────────
        if self._generate_report:
            try:
                portfolio_result = await self._report_generator.generate(portfolio_result)
                logger.info("Advisor report generated")
            except Exception as exc:
                logger.warning(f"Report generation failed — continuing: {exc}")

        # ── Step 9: Update M4 signal weights ──────────────────
        if self._update_weights and signal_results:
            try:
                await self._weight_updater.update_and_store(
                    signal_results=signal_results,
                    backtest_period=period_str,
                )
                logger.info("Signal weights updated in SQLite")
            except Exception as exc:
                logger.warning(f"Weight update failed — continuing: {exc}")

        # ── Step 10: Cache portfolio result ───────────────────
        try:
            from module_backtest.data.data_cache import backtest_result_cache
            backtest_result_cache.store_portfolio_result(portfolio_result)
        except Exception as exc:
            logger.warning(f"Could not cache portfolio result: {exc}")

        return portfolio_result

    async def run_ticker_backtest(
        self,
        ticker: str,
        bars: Optional[list[OHLCVBar]] = None,
        signal_types: Optional[list[SignalType]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        nifty_bars: Optional[list[OHLCVBar]] = None,
    ) -> list[BacktestResult]:
        """Run backtest for a single ticker across all signal types.

        Useful for targeted analysis or debugging a specific ticker.
        Returns list[BacktestResult] — one per signal_type.

        Args:
            ticker:       NSE ticker symbol.
            bars:         Pre-fetched OHLCVBar list. Fetched from Kite if None.
            signal_types: Signal types to evaluate. Defaults to all 4.
            from_date:    Start date. Defaults to BACKTEST_FROM_DATE.
            to_date:      End date. Defaults to BACKTEST_TO_DATE.
            nifty_bars:   Optional benchmark bars.

        Returns:
            list[BacktestResult], one per signal_type.
        """
        actual_from = from_date or BACKTEST_FROM_DATE
        actual_to = to_date or BACKTEST_TO_DATE
        signal_types = signal_types or _ALL_SIGNAL_TYPES

        if bars is None:
            bars = await historical_fetcher.fetch(ticker, actual_from, actual_to)

        if not bars:
            logger.warning(f"[{ticker}] No bars available")
            return []

        config = BacktestConfig(
            tickers=[ticker],
            signal_types=signal_types,
            from_date=actual_from,
            to_date=actual_to,
            in_sample_end_date=IN_SAMPLE_END_DATE if actual_from <= IN_SAMPLE_END_DATE <= actual_to else None,
            starting_capital=self._starting_capital,
        )

        _, results = self._run_ticker(ticker, bars, signal_types, config, nifty_bars)
        return results

    # ── Internal pipeline ───────────────────────────────────

    def _run_ticker(
        self,
        ticker: str,
        bars: list[OHLCVBar],
        signal_types: list[SignalType],
        config: BacktestConfig,
        nifty_bars: Optional[list[OHLCVBar]],
    ) -> tuple[list[TradeSimulation], list[BacktestResult]]:
        """Run full simulation pipeline for one ticker.

        Returns: (all_trades_flat, per_signal_backtest_results)
        """
        simulator = TradeSimulator(config)
        all_trades: list[TradeSimulation] = []
        results: list[BacktestResult] = []

        for signal_type in signal_types:
            # ── Signal replay ──────────────────────────────────
            signals = SignalReplayer.replay(ticker, bars, signal_types=[signal_type], nifty_bars=nifty_bars)
            if not signals:
                continue

            # ── Trade simulation ───────────────────────────────
            trades = simulator.simulate_signal_type(
                signals, bars, signal_type, nifty_bars=nifty_bars
            )
            all_trades.extend(trades)

            # ── Metrics ────────────────────────────────────────
            metrics = self._calculator.compute(trades)

            # ── Walk-forward ───────────────────────────────────
            wf = None
            if config.in_sample_end_date:
                try:
                    wf = WalkForwardEngine.run(
                        ticker, bars, signal_type,
                        config=config, nifty_bars=nifty_bars
                    )
                except Exception as exc:
                    logger.debug(f"[{ticker}:{signal_type.value}] Walk-forward skipped: {exc}")

            # ── Verdict ────────────────────────────────────────
            verdict, verdict_note = self._calculator.verdict(metrics)

            # ── Best / worst trade ─────────────────────────────
            entered = [t for t in trades if t.exit_reason != ExitReason.NEVER_ENTERED]
            best_trade = (
                max(entered, key=lambda t: t.return_pct) if entered else None
            )
            worst_trade = (
                min(entered, key=lambda t: t.return_pct) if entered else None
            )

            # ── Gross profit / loss ────────────────────────────
            gross_profit = sum(
                (t.net_pnl for t in entered if t.net_pnl > 0), Decimal("0")
            )
            gross_loss = sum(
                (t.net_pnl for t in entered if t.net_pnl <= 0), Decimal("0")
            )
            total_net = gross_profit + gross_loss
            ending_cap = (self._starting_capital + total_net).quantize(Decimal("0.01"))

            result = BacktestResult(
                signal_type=signal_type,
                ticker=ticker,
                period_start=config.from_date,
                period_end=config.to_date,
                total_signals=len(signals),
                trades_taken=len(entered),
                trades_skipped=len(trades) - len(entered),
                metrics=metrics,
                best_trade=best_trade,
                worst_trade=worst_trade,
                in_sample_metrics=wf.in_sample if wf else None,
                out_of_sample_metrics=wf.out_of_sample if wf else None,
                is_overfit=wf.is_overfit if wf else None,
                advisor_verdict=verdict,
                advisor_note=verdict_note,
                total_gross_profit=gross_profit.quantize(Decimal("0.01")),
                total_gross_loss=gross_loss.quantize(Decimal("0.01")),
                ending_capital=ending_cap,
            )
            results.append(result)

            logger.debug(
                f"[{ticker}:{signal_type.value}] "
                f"{len(signals)} signals → {len(entered)} trades → "
                f"WR={metrics.win_rate:.1f}%, PF={metrics.profit_factor:.2f} "
                f"[{verdict.value}]"
            )

        return all_trades, results

    # ── Data fetching ───────────────────────────────────────

    async def _fetch_all_data(
        self,
        tickers: list[str],
        from_date: date,
        to_date: date,
    ) -> tuple[dict[str, list[OHLCVBar]], Optional[list[OHLCVBar]]]:
        """Fetch OHLCV data for all tickers and Nifty concurrently.

        Returns: (bar_map, nifty_bars)
        """
        # Fetch all tickers + Nifty in one concurrent batch
        all_tickers = list(tickers)
        nifty_key = "__NIFTY__"

        async def _fetch_one(ticker: str) -> tuple[str, list[OHLCVBar]]:
            try:
                if ticker == nifty_key:
                    bars = await historical_fetcher.fetch_nifty(from_date, to_date)
                else:
                    bars = await historical_fetcher.fetch(ticker, from_date, to_date)
                return ticker, bars
            except Exception as exc:
                logger.warning(f"[{ticker}] Fetch failed: {exc}")
                return ticker, []

        tasks = [_fetch_one(t) for t in all_tickers] + [_fetch_one(nifty_key)]
        fetched = await asyncio.gather(*tasks)

        bar_map: dict[str, list[OHLCVBar]] = {}
        nifty_bars: Optional[list[OHLCVBar]] = None

        for ticker, bars in fetched:
            if ticker == nifty_key:
                nifty_bars = bars if bars else None
            elif bars:
                bar_map[ticker] = bars

        return bar_map, nifty_bars

    # ── Empty result ────────────────────────────────────────

    def _empty_result(
        self,
        tickers: list[str],
        from_date: date,
        to_date: date,
    ) -> PortfolioBacktestResult:
        """Return an empty result when no data is available."""
        return PortfolioBacktestResult(
            period_start=from_date,
            period_end=to_date,
            starting_capital=self._starting_capital,
            ending_capital=self._starting_capital,
            metrics=PerformanceMetrics(),
            tickers_tested=tickers,
            advisor_verdict=AdvisorVerdict.INSUFFICIENT_DATA,
            advisor_note="No data available to run backtest.",
        )
