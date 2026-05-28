"""
SwingAdvisorBot — run_full_backtest.py
Full universe backtest runner — 55 NSE stocks, 12 months.

Usage:
    python run_full_backtest.py

Takes ~60-90 seconds for 55 stocks (Kite rate-limited to 3 req/s).
Results are cached in SQLite so re-runs are near-instant.
"""

import asyncio
import logging

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.WARNING,   # suppress INFO noise during run
    format="%(levelname)s %(name)s: %(message)s",
)
# Keep backtest engine INFO visible
logging.getLogger("swing_advisor.backtest.engine").setLevel(logging.INFO)

from module_backtest.backtest_engine import BacktestEngine
from module_backtest.config import NSE_STOCK_UNIVERSE


async def main() -> None:
    print("=" * 56)
    print("  SWINGADVISORBOT — FULL UNIVERSE BACKTEST")
    print(f"  Universe : {len(NSE_STOCK_UNIVERSE)} NSE stocks")
    print(f"  Period   : 12 months (May 2025 – May 2026)")
    print("=" * 56)
    print()
    print("Step 1: Resolving instrument tokens...")
    print("Step 2: Fetching historical data (rate-limited 3 req/s)")
    print("        (~60-90 seconds for 55 stocks on first run)")
    print("        (Near-instant on subsequent runs — SQLite cache)")
    print()

    engine = BacktestEngine(generate_report=True, update_weights=True)

    results = await engine.run_full_backtest(
        tickers=None,   # uses all 55 from NSE_STOCK_UNIVERSE
        period_months=12,
    )

    m = results.metrics

    print()
    print("=" * 56)
    print("  BACKTEST RESULTS — FULL UNIVERSE")
    print("=" * 56)
    print(f"  Stocks tested  : {len(results.tickers_tested)}")
    skipped = len(NSE_STOCK_UNIVERSE) - len(results.tickers_tested)
    print(f"  Stocks skipped : {skipped}  (no data / token not found)")
    print(f"  Total trades   : {m.total_trades}")
    print(f"  Win rate       : {m.win_rate:.1f}%")
    print(f"  Total return   : {m.total_return_pct:+.1f}%")
    print(f"  Profit factor  : {m.profit_factor:.2f}")
    print(f"  Sharpe ratio   : {m.sharpe_ratio:.2f}")
    alpha_str = f"{m.alpha:+.1f}%" if m.alpha is not None else "n/a"
    print(f"  Alpha vs Nifty : {alpha_str}")
    print(f"  Avg hold days  : {m.avg_hold_days:.1f}")
    print(f"  Max drawdown   : {m.max_drawdown_pct:.1f}%")
    print(f"  Capital        : ₹{results.starting_capital:,} → ₹{results.ending_capital:,}")
    print()
    print(f"  VERDICT: {results.advisor_verdict.value}")
    print()

    # Per-signal breakdown from signal_results
    if results.signal_results:
        from collections import defaultdict
        from module_backtest.engine.metrics_calculator import MetricsCalculator
        from decimal import Decimal

        # Aggregate per signal_type across all tickers
        by_signal: dict[str, list] = defaultdict(list)
        for r in results.signal_results:
            by_signal[r.signal_type.value].append(r.metrics)

        print("  SIGNAL PERFORMANCE:")
        calc = MetricsCalculator(starting_capital=Decimal("50000"))
        for sig, metrics_list in sorted(by_signal.items()):
            total_trades = sum(mm.total_trades for mm in metrics_list)
            total_wins   = sum(mm.wins for mm in metrics_list)
            wr = (total_wins / total_trades * 100) if total_trades else 0.0
            gross_p = sum(mm.profit_factor * mm.losses for mm in metrics_list if mm.losses > 0)
            gross_l = sum(mm.losses for mm in metrics_list if mm.losses > 0)
            pf = gross_p / gross_l if gross_l > 0 else 0.0
            print(f"    {sig:<22}: WR={wr:.0f}%  PF={pf:.2f}  ({total_trades} trades)")
        print()

    # Top / worst tickers by win rate (min 3 trades)
    ticker_map: dict[str, dict] = {}
    for r in results.signal_results:
        key = r.ticker
        if key not in ticker_map:
            ticker_map[key] = {"wins": 0, "trades": 0, "pct": 0.0}
        ticker_map[key]["wins"]   += r.metrics.wins
        ticker_map[key]["trades"] += r.metrics.total_trades
    for key, d in ticker_map.items():
        d["wr"] = d["wins"] / d["trades"] * 100 if d["trades"] else 0.0

    qualified = [(k, d) for k, d in ticker_map.items() if d["trades"] >= 3]
    qualified.sort(key=lambda x: x[1]["wr"], reverse=True)

    if qualified:
        print("  TOP 5 PERFORMING STOCKS  (≥3 trades):")
        for ticker, d in qualified[:5]:
            print(f"    {ticker:<12}: WR={d['wr']:.0f}%  ({d['trades']} trades)")
        print()
        print("  BOTTOM 5 PERFORMING STOCKS  (≥3 trades):")
        for ticker, d in qualified[-5:]:
            print(f"    {ticker:<12}: WR={d['wr']:.0f}%  ({d['trades']} trades)")
        print()

    if results.best_ticker:
        print(f"  Best ticker  : {results.best_ticker}")
    if results.worst_ticker:
        print(f"  Worst ticker : {results.worst_ticker}")

    # Advisor narrative
    if results.advisor_note:
        print()
        print("  ADVISOR NARRATIVE:")
        for line in results.advisor_note.strip().split("\n")[:8]:
            print(f"    {line}")

    print()
    print("=" * 56)


if __name__ == "__main__":
    asyncio.run(main())
