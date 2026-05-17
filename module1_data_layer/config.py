"""
SwingAdvisorBot — Module 1: Data Layer
config.py — Central configuration and environment management

This file is the single source of truth for all configuration values
across Module 1. Every API key, rate limit, cache TTL, and fetch
parameter lives here — never scattered across fetcher files.

Design decisions:
  - All secrets loaded from .env via python-dotenv (never hardcoded)
  - DataFetchConfig controls what and how much data we fetch
  - Settings is a singleton-like frozen config loaded once at startup
  - NSE market hours are codified here — no magic numbers elsewhere

Token budget awareness:
  DataFetchConfig.token_budget = 2500 tokens max for MarketData payload.
  Every fetcher respects max_stocks, max_news, max_economic_events
  to prevent over-fetching that wastes API calls and Claude tokens.
"""

from __future__ import annotations

import os
from datetime import time
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")


# ─────────────────────────────────────────────────────────────
# NSE Market Schedule — Codified once, referenced everywhere
# ─────────────────────────────────────────────────────────────

NSE_PRE_MARKET_OPEN = time(9, 0)     # 9:00 AM IST — pre-market session starts
NSE_MARKET_OPEN = time(9, 15)        # 9:15 AM IST — regular trading begins
NSE_MARKET_CLOSE = time(15, 30)      # 3:30 PM IST — regular trading ends
NSE_POST_MARKET_CLOSE = time(16, 0)  # 4:00 PM IST — post-market session ends

# NSE holidays are date objects — add manually or fetch from NSE calendar.
# For production, this should be updated annually.
NSE_HOLIDAYS_2026: list[str] = [
    "2026-01-26",  # Republic Day
    "2026-03-10",  # Maha Shivaratri
    "2026-03-17",  # Holi
    "2026-03-31",  # Id-ul-Fitr
    "2026-04-02",  # Ram Navami
    "2026-04-03",  # Mahavir Jayanti
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-04-18",  # Good Friday
    "2026-05-01",  # Maharashtra Day
    "2026-06-07",  # Id-ul-Adha (Bakri Id)
    "2026-07-07",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-08-16",  # Janmashtami (tentative)
    "2026-09-05",  # Milad-un-Nabi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-09",  # Diwali (Laxmi Puja)
    "2026-11-10",  # Diwali (Balipratipada)
    "2026-11-30",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
]


# ─────────────────────────────────────────────────────────────
# Default Watchlist — Core NSE stocks for swing trading
# These are the most liquid, widely tracked large-caps.
# The advisor always monitors these unless overridden.
# ─────────────────────────────────────────────────────────────

DEFAULT_WATCHLIST: list[str] = [
    "HDFCBANK",
    "RELIANCE",
    "TCS",
    "INFY",
    "ICICIBANK",
    "BHARTIARTL",
    "SBIN",
    "ITC",
    "KOTAKBANK",
    "LT",
    "HINDUNILVR",
    "AXISBANK",
    "BAJFINANCE",
    "MARUTI",
    "TATAMOTORS",
]


# ─────────────────────────────────────────────────────────────
# Sector-to-Index Mapping — NSE sector indices for sector_fetcher
# Maps sector names to their Kite instrument tokens or NSE symbols.
# ─────────────────────────────────────────────────────────────

NSE_SECTOR_INDICES: dict[str, str] = {
    "Banking": "NIFTY BANK",
    "IT": "NIFTY IT",
    "Pharma": "NIFTY PHARMA",
    "Auto": "NIFTY AUTO",
    "Finance": "NIFTY FIN SERVICE",
    "FMCG": "NIFTY FMCG",
    "Metal": "NIFTY METAL",
    "Realty": "NIFTY REALTY",
    "Energy": "NIFTY ENERGY",
    "Media": "NIFTY MEDIA",
}

# ─────────────────────────────────────────────────────────────
# Stock Metadata — Sector and company name lookup
# Used by stock_fetcher to enrich raw Kite data with context
# that the advisor needs for sector-aware analysis.
# ─────────────────────────────────────────────────────────────

STOCK_METADATA: dict[str, dict[str, str]] = {
    "HDFCBANK": {"company_name": "HDFC Bank Limited", "sector": "Banking"},
    "RELIANCE": {"company_name": "Reliance Industries Limited", "sector": "Energy"},
    "TCS": {"company_name": "Tata Consultancy Services Limited", "sector": "IT"},
    "INFY": {"company_name": "Infosys Limited", "sector": "IT"},
    "ICICIBANK": {"company_name": "ICICI Bank Limited", "sector": "Banking"},
    "BHARTIARTL": {"company_name": "Bharti Airtel Limited", "sector": "Telecom"},
    "SBIN": {"company_name": "State Bank of India", "sector": "Banking"},
    "ITC": {"company_name": "ITC Limited", "sector": "FMCG"},
    "KOTAKBANK": {"company_name": "Kotak Mahindra Bank Limited", "sector": "Banking"},
    "LT": {"company_name": "Larsen & Toubro Limited", "sector": "Infrastructure"},
    "HINDUNILVR": {"company_name": "Hindustan Unilever Limited", "sector": "FMCG"},
    "AXISBANK": {"company_name": "Axis Bank Limited", "sector": "Banking"},
    "BAJFINANCE": {"company_name": "Bajaj Finance Limited", "sector": "Finance"},
    "MARUTI": {"company_name": "Maruti Suzuki India Limited", "sector": "Auto"},
    "TATAMOTORS": {"company_name": "Tata Motors Limited", "sector": "Auto"},
}


# ─────────────────────────────────────────────────────────────
# FRED Series — Macro indicators tracked for Indian market context
# ─────────────────────────────────────────────────────────────

FRED_SERIES: dict[str, dict[str, str]] = {
    "FEDFUNDS": {
        "event_name": "Federal Funds Rate",
        "impact_note": "US rate changes affect FII flows into Indian equities. Higher rates pull money to US.",
    },
    "CPIAUCSL": {
        "event_name": "US Consumer Price Index",
        "impact_note": "US inflation data drives Fed policy expectations, affecting global risk sentiment.",
    },
    "DCOILWTICO": {
        "event_name": "Crude Oil WTI",
        "impact_note": "India imports 85% of crude oil. Rising crude weakens INR and pressures fiscal deficit.",
    },
    "DGS10": {
        "event_name": "US 10-Year Treasury Yield",
        "impact_note": "Rising US yields compete with Indian equities for FII capital. Inverse correlation.",
    },
    "DEXINUS": {
        "event_name": "USD/INR Exchange Rate",
        "impact_note": "Rupee depreciation hurts IT importers but benefits IT exporters like TCS, Infosys.",
    },
}


# ─────────────────────────────────────────────────────────────
# DataFetchConfig — Controls what and how much data we fetch
# Every fetcher reads this config to respect limits.
# Token budget is enforced via MarketData.trim_to_budget().
# ─────────────────────────────────────────────────────────────


class DataFetchConfig(BaseModel):
    """Controls the scope and limits of every data fetch operation.

    This configuration is passed to DataCollectorAgent.execute()
    and propagated to all fetcher functions. It ensures we never
    over-fetch (wasting API calls and money) or under-fetch
    (leaving the advisor blind to market conditions).

    Rate limit awareness:
      Kite Connect: 3 requests/second → Semaphore(3) in rate_limiter.py
      NewsAPI: 100 requests/day → cache 15 minutes
      FRED API: 120 requests/minute → cache 60 minutes

    Token budget awareness:
      max_stocks * ~100 tokens/stock = ~600 tokens for stocks
      max_news * ~80 tokens/item = ~240 tokens for news
      VIX + indices + metadata ≈ ~200 tokens
      Buffer for JSON structure ≈ ~300 tokens
      Total budget: 1340 tokens → enforced by TokenController.trim_to_budget()
    """

    max_stocks: int = Field(
        default=6,
        description="Maximum number of stocks to fetch. 6 keeps MarketData within 1340 token budget.",
    )
    max_news: int = Field(
        default=3,
        description="Maximum news items to return after relevance filtering. 3 keeps token budget tight.",
    )
    max_economic_events: int = Field(
        default=3,
        description="Maximum FRED economic events to include. 3 is sufficient for macro context.",
    )
    cache_ttl_stocks: int = Field(
        default=180,
        description="Cache TTL for stock data in seconds (3 minutes). Balances freshness vs rate limits.",
    )
    cache_ttl_news: int = Field(
        default=900,
        description="Cache TTL for news in seconds (15 minutes). NewsAPI has 100/day limit — cache hard.",
    )
    cache_ttl_events: int = Field(
        default=3600,
        description="Cache TTL for FRED economic data in seconds (60 minutes). Macro data changes slowly.",
    )
    cache_ttl_vix: int = Field(
        default=300,
        description="Cache TTL for India VIX in seconds (5 minutes). VIX updates every few seconds during market hours.",
    )
    min_news_relevance: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum relevance_score for a news item to reach the advisor. Below this = noise.",
    )
    enable_cot_reasoning: bool = Field(
        default=True,
        description="Whether to generate Chain of Thought reasoning strings. Disable to save tokens.",
    )
    token_budget: int = Field(
        default=2500,
        description="Hard token limit for serialized MarketData. Claude API costs per token — every token earns its place.",
    )


# ─────────────────────────────────────────────────────────────
# Settings — Environment variables loaded from .env
# Loaded once via get_settings(), cached with lru_cache.
# ─────────────────────────────────────────────────────────────


class Settings(BaseModel):
    """Application settings loaded from environment variables.

    All API keys and secrets come from .env — never hardcoded.
    Missing keys will default to empty string; each fetcher
    validates its own keys before making API calls and raises
    KiteAuthError or DataFetchError with a clear message.
    """

    model_config = {"frozen": True}

    # Kite Connect (Zerodha)
    kite_api_key: str = Field(
        default="",
        description="Kite Connect API key from Zerodha developer console",
    )
    kite_api_secret: str = Field(
        default="",
        description="Kite Connect API secret from Zerodha developer console",
    )
    kite_access_token: str = Field(
        default="",
        description="Kite Connect access token — refreshed daily via kite_auth.py",
    )
    kite_client_id: str = Field(
        default="",
        description="Zerodha client ID (e.g. XCU700) for login flow",
    )

    # NewsAPI
    news_api_key: str = Field(
        default="",
        description="NewsAPI.org API key for market news headlines",
    )

    # FRED
    fred_api_key: str = Field(
        default="",
        description="Federal Reserve Economic Data API key for macro indicators",
    )

    # MCP Server
    mcp_host: str = Field(
        default="0.0.0.0",
        description="MCP server bind host. 0.0.0.0 for container, 127.0.0.1 for local.",
    )
    mcp_port: int = Field(
        default=8001,
        description="MCP server port. Module 2 connects to http://localhost:8001",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings from environment variables.

    Called once at startup. Uses lru_cache to avoid re-reading .env
    on every access. All fetchers call get_settings() to get their keys.

    If a key is missing from .env, it defaults to empty string.
    Each fetcher validates its own keys and raises descriptive errors.
    """
    return Settings(
        kite_api_key=os.getenv("KITE_API_KEY", ""),
        kite_api_secret=os.getenv("KITE_API_SECRET", ""),
        kite_access_token=os.getenv("KITE_ACCESS_TOKEN", ""),
        kite_client_id=os.getenv("ZERODHA_CLIENT_ID", ""),
        news_api_key=os.getenv("NEWS_API_KEY", ""),
        fred_api_key=os.getenv("FRED_API_KEY", ""),
        mcp_host=os.getenv("MCP_HOST", "0.0.0.0"),
        mcp_port=int(os.getenv("MCP_PORT", "8001")),
    )
