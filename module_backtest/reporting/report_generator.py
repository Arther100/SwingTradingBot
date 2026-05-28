"""
SwingAdvisorBot — Module Backtest: Backtesting Engine
reporting/report_generator.py — Claude-powered backtest advisor report

This is the ONLY file in the backtesting engine that calls Claude.
All signal replay, trade simulation, and metric computation is pure Python.
Claude is invoked once at the very end to write a plain-English interpretation.

Report generation flow:
  PortfolioBacktestResult
    → _build_summary_text()   ← compact ~300-token text representation
    → Claude (call_claude_raw)  ← system prompt from config.py
    → HTML response text
    → populate result.advisor_note + result.telegram_text

Token budget: ~500 tokens in + ~400 tokens out = ~900 total
(deliberately small — this is just an interpretation, not the main analysis)

Claude is NOT used for:
  - Signal replay (deterministic Python)
  - Trade simulation (deterministic Python)
  - Metric calculation (deterministic Python)
  - Verdict determination (rule-based from thresholds)

Fallback behaviour:
  If Claude is unavailable (no API key, network error, rate limit), the
  report_generator returns the pre-built plain-text summary instead.
  The backtest result is still complete — only the narrative is missing.
  The advisor_note will contain a fallback message noting this.

Usage:
    generator = ReportGenerator()
    result = await generator.generate(portfolio_result)
    # result.advisor_note  → multi-paragraph HTML for Telegram
    # result.telegram_text → formatted Telegram message
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from module_backtest.config import (
    ANTHROPIC_API_KEY,
    BACKTEST_ADVISOR_SYSTEM_PROMPT,
    BACKTEST_REPORT_TOKEN_BUDGET,
    CLAUDE_MODEL,
)
from module_backtest.models import (
    AdvisorVerdict,
    BacktestResult,
    PerformanceMetrics,
    PortfolioBacktestResult,
    SignalType,
)

logger = logging.getLogger("swing_advisor.backtest.report_generator")

# Max characters for the plain-text summary sent to Claude
_MAX_SUMMARY_CHARS = 1800


class ReportGenerator:
    """Generates the Claude-powered advisor narrative for a backtest run.

    Usage:
        generator = ReportGenerator()
        updated_result = await generator.generate(portfolio_result)

    The returned PortfolioBacktestResult has advisor_note and telegram_text
    populated. The original result object is not mutated — a new instance
    is returned with the added fields.
    """

    async def generate(
        self,
        result: PortfolioBacktestResult,
    ) -> PortfolioBacktestResult:
        """Generate Claude advisor report and attach to portfolio result.

        Args:
            result: Completed PortfolioBacktestResult (from BacktestEngine).

        Returns:
            New PortfolioBacktestResult with advisor_note and telegram_text set.
            If Claude is unavailable, returns result with fallback advisor_note.
        """
        summary_text = _build_summary_text(result)

        advisor_note = await self._call_claude(summary_text)
        if advisor_note is None:
            advisor_note = _fallback_note(result)

        telegram_text = _build_telegram_message(result, advisor_note)

        # Return a new model instance with the narrative fields populated
        return result.model_copy(
            update={
                "advisor_note": advisor_note,
                "telegram_text": telegram_text,
            }
        )

    async def _call_claude(self, summary_text: str) -> Optional[str]:
        """Call Claude API with the backtest summary.

        Uses raw HTTP via httpx (same pattern as ClaudeClient) to avoid
        importing the full M2 stack (which has its own heavy config chain).

        Returns the response text on success, None on any failure.
        """
        api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — skipping Claude report")
            return None

        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed — skipping Claude report")
            return None

        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": BACKTEST_REPORT_TOKEN_BUDGET["output_budget"],
            "temperature": 0.3,
            "system": BACKTEST_ADVISOR_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": summary_text,
                }
            ],
        }

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["content"][0]["text"].strip()
                logger.info(
                    f"Claude report generated — "
                    f"{data.get('usage', {}).get('output_tokens', '?')} output tokens"
                )
                return text

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                logger.error("Claude API: invalid API key (401)")
            elif status == 429:
                logger.warning("Claude API: rate limited (429) — using fallback")
            elif status == 529:
                logger.warning("Claude API: overloaded (529) — using fallback")
            else:
                logger.error(f"Claude API HTTP error: {status}")
            return None

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.warning(f"Claude API connection error — using fallback: {exc}")
            return None

        except (KeyError, IndexError, ValueError) as exc:
            logger.error(f"Claude API response parse error: {exc}")
            return None


# ═══════════════════════════════════════════════════════════
# SUMMARY TEXT BUILDER (Claude input)
# ═══════════════════════════════════════════════════════════


def _build_summary_text(result: PortfolioBacktestResult) -> str:
    """Build a compact plain-text summary of the portfolio backtest result.

    Target: < 500 tokens (≈ 1800 chars) to stay within token budget.
    Claude will use this to write a plain-English interpretation.
    """
    m = result.metrics
    period = f"{result.period_start} to {result.period_end}"
    tickers_str = ", ".join(result.tickers_tested[:10])
    if len(result.tickers_tested) > 10:
        tickers_str += f" (+{len(result.tickers_tested) - 10} more)"

    lines = [
        f"BACKTEST SUMMARY — {period}",
        f"Capital: ₹{result.starting_capital:,} → ₹{result.ending_capital:,}",
        f"Tickers tested: {tickers_str}",
        "",
        "PORTFOLIO METRICS:",
        f"  Win Rate:       {m.win_rate:.1f}%  (target: ≥52%)",
        f"  Profit Factor:  {m.profit_factor:.2f}  (target: ≥1.3)",
        f"  Total Trades:   {m.total_trades}  (wins: {m.wins}, losses: {m.losses}, timeouts: {m.timeouts})",
        f"  Total Return:   {m.total_return_pct:+.1f}%",
    ]

    if m.nifty_return_pct is not None:
        lines.append(f"  Nifty Return:   {m.nifty_return_pct:+.1f}%")
    if m.alpha is not None:
        lines.append(f"  Alpha:          {m.alpha:+.1f}%  (target: positive)")

    lines += [
        f"  Max Drawdown:   {m.max_drawdown_pct:.1f}%  (safe: <15%)",
        f"  Sharpe Ratio:   {m.sharpe_ratio:.2f}  (target: ≥1.0)",
        f"  Avg Hold Days:  {m.avg_hold_days:.1f}",
    ]

    if m.best_month:
        lines.append(f"  Best Month:     {m.best_month}")
    if m.worst_month:
        lines.append(f"  Worst Month:    {m.worst_month}")

    lines.append("")
    lines.append("VERDICT: " + result.advisor_verdict.value)

    # Per-signal breakdown (top signals only)
    if result.signal_results:
        lines.append("")
        lines.append("SIGNAL BREAKDOWN:")
        for sr in result.signal_results[:8]:  # cap at 8 to stay within budget
            wf = ""
            if sr.in_sample_metrics and sr.out_of_sample_metrics:
                delta = sr.out_of_sample_metrics.win_rate - sr.in_sample_metrics.win_rate
                wf = f" | walk-fwd delta: {delta:+.1f}pp"
                if sr.is_overfit:
                    wf += " [OVERFIT]"
            wr = sr.metrics.win_rate
            pf = sr.metrics.profit_factor
            n = sr.metrics.total_trades
            lines.append(
                f"  {sr.signal_type.value:<20} {wr:.0f}% WR  PF {pf:.2f}  "
                f"n={n}  [{sr.advisor_verdict.value}]{wf}"
            )

    if result.best_ticker:
        lines.append(f"\nBest ticker:  {result.best_ticker}")
    if result.worst_ticker:
        lines.append(f"Worst ticker: {result.worst_ticker}")

    summary = "\n".join(lines)

    # Hard truncation to stay within budget
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + "\n[truncated]"

    return summary


# ═══════════════════════════════════════════════════════════
# TELEGRAM MESSAGE BUILDER
# ═══════════════════════════════════════════════════════════


def _build_telegram_message(
    result: PortfolioBacktestResult,
    advisor_note: str,
) -> str:
    """Build the Telegram-ready HTML message for Vijay.

    Combines the pre-formatted Claude advisor note with a compact
    metrics header so Vijay can see the numbers at a glance.
    """
    m = result.metrics
    period = f"{result.period_start.strftime('%b %Y')} – {result.period_end.strftime('%b %Y')}"

    verdict_emoji = {
        AdvisorVerdict.STRATEGY_VALIDATED: "✅",
        AdvisorVerdict.VALID_SIGNAL: "✅",
        AdvisorVerdict.WEAK_SIGNAL: "⚠️",
        AdvisorVerdict.INVALID_SIGNAL: "❌",
        AdvisorVerdict.INSUFFICIENT_DATA: "❓",
    }.get(result.advisor_verdict, "📊")

    alpha_str = ""
    if m.alpha is not None:
        sign = "+" if m.alpha >= 0 else ""
        alpha_str = f" | Alpha {sign}{m.alpha:.1f}%"

    nifty_str = ""
    if m.nifty_return_pct is not None:
        sign = "+" if m.nifty_return_pct >= 0 else ""
        nifty_str = f" | Nifty {sign}{m.nifty_return_pct:.1f}%"

    capital_return = float(result.ending_capital - result.starting_capital)
    capital_sign = "+" if capital_return >= 0 else ""

    lines = [
        f"<b>📊 Backtest Results ({period})</b>",
        "",
        f"{verdict_emoji} <b>{result.advisor_verdict.value}</b>",
        "",
        f"Win Rate: <b>{m.win_rate:.1f}%</b>  |  Profit Factor: <b>{m.profit_factor:.2f}</b>",
        f"Return: <b>{m.total_return_pct:+.1f}%</b>{nifty_str}{alpha_str}",
        f"Capital: ₹{result.starting_capital:,.0f} → <b>₹{result.ending_capital:,.0f}</b> ({capital_sign}₹{abs(capital_return):,.0f})",
        f"Trades: {m.total_trades}  |  Max Drawdown: {m.max_drawdown_pct:.1f}%  |  Sharpe: {m.sharpe_ratio:.2f}",
        "",
        advisor_note,
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# FALLBACK (when Claude unavailable)
# ═══════════════════════════════════════════════════════════


def _fallback_note(result: PortfolioBacktestResult) -> str:
    """Plain-text fallback when Claude API is unavailable.

    Returns a rule-based interpretation using the same thresholds
    as the verdict logic. No Claude tokens used.
    """
    m = result.metrics
    v = result.advisor_verdict

    if v == AdvisorVerdict.STRATEGY_VALIDATED:
        quality = "strong"
        action = "Results support live trading with Vijay's 2% risk rules."
    elif v == AdvisorVerdict.VALID_SIGNAL:
        quality = "valid"
        action = "Consider live trading on strongest signals with reduced size."
    elif v == AdvisorVerdict.WEAK_SIGNAL:
        quality = "weak"
        action = "Paper trade for 1 more month before committing capital."
    elif v == AdvisorVerdict.INVALID_SIGNAL:
        quality = "poor"
        action = "Do not trade live. Review signal logic before re-testing."
    else:
        quality = "inconclusive"
        action = "Collect more data — run backtest on wider date range."

    alpha_str = ""
    if m.alpha is not None:
        direction = "beats" if m.alpha > 0 else "lags"
        alpha_str = f" Strategy {direction} Nifty by {abs(m.alpha):.1f}%."

    return (
        f"<b>Backtest Results (1 Year)</b>\n"
        f"Win rate {m.win_rate:.1f}%, profit factor {m.profit_factor:.2f} "
        f"across {m.total_trades} trades — <i>{quality}</i> signal quality.{alpha_str} "
        f"Max drawdown {m.max_drawdown_pct:.1f}%, Sharpe {m.sharpe_ratio:.2f}.\n\n"
        f"<b>Your Action:</b> {action}"
    )
