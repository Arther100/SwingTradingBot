"""
SwingAdvisorBot — Module 1: Data Layer
fetchers/news_fetcher.py — NewsAPI headline fetcher for Indian markets

This fetcher pulls market-relevant news headlines from NewsAPI and
transforms them into raw NewsItem objects ready for scoring by
signals/news_scorer.py (File 14).

The advisor needs news context to explain price movements:
  "HDFCBANK up 2% — RBI rate hold announced this morning,
   positive for banking sector NIMs."

Without news, the advisor can only describe WHAT happened,
not WHY it happened. News gives the advisor its reasoning voice.

Data flow:
  NewsAPI → raw articles → NewsItem (basic) → news_scorer.py → NewsItem (scored)

API specifics:
  - NewsAPI free tier: 100 requests/day
  - Endpoint: /v2/everything (for keyword search)
  - Language: English (en)
  - Sort: publishedAt (most recent first)
  - Search queries tuned for Indian market relevance

Rate limiting:
  - DailyQuotaLimiter: 95 effective requests/day (5 safety margin)
  - Cache TTL: 15 minutes (cache_ttl_news = 900s)
  - With 15-minute caching, even running every 3 minutes uses only
    ~96 requests/day (well within quota)

Caching strategy:
  - Headlines cached for 15 minutes — news doesn't go stale that fast.
  - If daily quota is exhausted, cache serves the rest of the day.
  - Cache key: "news:headlines" — single key for all market news.

Edge cases:
  - Daily quota exhausted → return cached data or empty list (logged clearly)
  - API key missing → DataFetchError with setup instructions
  - No results for query → return empty list (market may be quiet)
  - Non-English/irrelevant results → filtered by news_scorer.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx
from zoneinfo import ZoneInfo

from module1_data_layer.cache import cache
from module1_data_layer.config import DataFetchConfig, get_settings
from module1_data_layer.models import DataFetchError, NewsItem, NewsSentiment
from module1_data_layer.rate_limiter import news_limiter

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.fetchers.news")

# NewsAPI base URL
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

# Search queries tuned for Indian market relevance.
# Multiple queries are ORed to capture broad market news.
# These keywords are designed to catch:
#   - RBI policy decisions (rates, inflation targets)
#   - NSE/BSE market movements (Nifty, Sensex)
#   - Major company news (Reliance, TCS, HDFC, etc.)
#   - Sector-specific events (banking, IT, pharma)
#   - Global events affecting India (Fed, crude oil, FII flows)
INDIA_MARKET_QUERY = (
    "(NSE OR BSE OR Nifty OR Sensex OR RBI) "
    "AND (market OR stock OR trading OR economy OR rate)"
)

# Domains we trust for Indian market news — filters out noise
TRUSTED_DOMAINS = (
    "economictimes.indiatimes.com,"
    "moneycontrol.com,"
    "livemint.com,"
    "ndtv.com,"
    "reuters.com,"
    "bloomberg.com,"
    "business-standard.com,"
    "financialexpress.com,"
    "zeebiz.com,"
    "cnbctv18.com"
)


async def fetch_news(
    config: DataFetchConfig,
) -> list[NewsItem]:
    """Fetch Indian market news headlines from NewsAPI.

    This is the primary entry point called by the pipeline (Step 4).
    Returns raw NewsItem objects with basic fields populated:
      - headline, source, url, published_at
      - sentiment defaults to NEUTRAL (news_scorer.py will refine)
      - relevance_score defaults to 0.0 (news_scorer.py will score)

    The scoring and enrichment (sentiment, relevance, affected_sectors,
    advisor_note, cot_reasoning) is done by news_scorer.py AFTER this
    function returns. This separation keeps fetching fast and scoring
    independent.

    Args:
        config: DataFetchConfig controlling max_news, cache TTL,
                and min_news_relevance.

    Returns:
        List of NewsItem objects with basic fields. May be empty if:
          - No news found for the search query
          - NewsAPI daily quota exhausted (serving from cache)
          - API key is missing (DataFetchError raised)

    Raises:
        DataFetchError: If NEWS_API_KEY is missing from .env.
            Individual API errors are handled gracefully (return cached/empty).
    """
    settings = get_settings()

    if not settings.news_api_key:
        raise DataFetchError(
            source="NewsAPI",
            reason="NEWS_API_KEY not found in .env file",
            suggestion=(
                "Add NEWS_API_KEY=<your_key> to .env. "
                "Get a free API key from https://newsapi.org/register"
            ),
        )

    # Check cache first — 15-minute TTL means we rarely hit the API
    cache_key = "news:headlines"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(
            f"Returning {len(cached)} news items from cache "
            f"(age: {cache.get_age(cache_key):.0f}s, TTL: {config.cache_ttl_news}s)."
        )
        return cached

    # Check daily quota before making the API call
    if not news_limiter.try_acquire():
        quota = news_limiter.usage_report
        logger.warning(
            f"NewsAPI daily quota exhausted — {quota['used']}/{quota['hard_limit']} "
            f"requests used today. Returning empty news list. "
            f"Cache must serve remaining requests until midnight reset."
        )
        return []

    try:
        # Build request parameters
        # Look back 24 hours for fresh news
        from_date = (datetime.now(IST) - timedelta(hours=24)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        params = {
            "q": INDIA_MARKET_QUERY,
            "domains": TRUSTED_DOMAINS,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(config.max_news * 3, 30),  # Fetch 3x to allow for filtering
            "apiKey": settings.news_api_key,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(NEWSAPI_BASE_URL, params=params)

        if response.status_code == 401:
            raise DataFetchError(
                source="NewsAPI",
                reason="API key is invalid or expired (HTTP 401)",
                suggestion=(
                    "Verify NEWS_API_KEY in .env. "
                    "Get a new key from https://newsapi.org/account"
                ),
            )

        if response.status_code == 429:
            logger.warning(
                "NewsAPI rate limit hit (HTTP 429). "
                "This shouldn't happen with our DailyQuotaLimiter — "
                "possible concurrent bot instance consuming quota. "
                "Returning empty news list."
            )
            return []

        if response.status_code != 200:
            logger.warning(
                f"NewsAPI returned HTTP {response.status_code}: "
                f"{response.text[:200]}. Returning empty news list."
            )
            return []

        data = response.json()

        if data.get("status") != "ok":
            logger.warning(
                f"NewsAPI returned non-ok status: {data.get('status')}. "
                f"Message: {data.get('message', 'none')}. "
                f"Returning empty news list."
            )
            return []

        articles = data.get("articles", [])

        if not articles:
            logger.info(
                "NewsAPI returned no articles for the market query. "
                "Market may be quiet or query too narrow. "
                "This is normal outside Indian business hours."
            )
            return []

        # Transform API articles into NewsItem objects
        news_items: list[NewsItem] = []
        for article in articles:
            # Skip articles with [Removed] content (NewsAPI placeholder)
            title = article.get("title", "")
            if not title or title == "[Removed]":
                continue

            source_name = article.get("source", {}).get("name", "Unknown")

            # Parse published date → IST
            published_str = article.get("publishedAt", "")
            published_at = _parse_published_date(published_str)

            news_item = NewsItem(
                headline=title.strip(),
                source=source_name,
                url=article.get("url", ""),
                sentiment=NewsSentiment.NEUTRAL,  # Scored by news_scorer.py
                relevance_score=0.0,  # Scored by news_scorer.py
                published_at=published_at,
            )
            news_items.append(news_item)

        # Cache the raw news items
        cache.set(cache_key, news_items, ttl=config.cache_ttl_news)

        quota = news_limiter.usage_report
        logger.info(
            f"Fetched {len(news_items)} news items from NewsAPI "
            f"(filtered from {len(articles)} raw articles). "
            f"Daily quota: {quota['remaining']} requests remaining. "
            f"Cached for {config.cache_ttl_news}s."
        )

        return news_items

    except DataFetchError:
        # Re-raise our typed errors
        raise

    except httpx.TimeoutException:
        logger.warning(
            "NewsAPI request timed out after 15 seconds. "
            "Possible causes: slow network, NewsAPI server issues. "
            "Returning empty news list — advisor will proceed without news context."
        )
        return []

    except httpx.HTTPError as e:
        logger.warning(
            f"NewsAPI HTTP error: {type(e).__name__}: {e}. "
            f"Returning empty news list."
        )
        return []

    except Exception as e:
        logger.warning(
            f"Unexpected error fetching news: {type(e).__name__}: {e}. "
            f"Returning empty news list — pipeline continues without news."
        )
        return []


def _parse_published_date(date_str: str) -> datetime:
    """Parse NewsAPI published date string into IST datetime.

    NewsAPI returns dates in ISO 8601 UTC format:
      "2026-05-14T08:15:00Z"

    This function parses it and converts to IST (UTC+5:30).
    Falls back to current IST time if parsing fails.

    Args:
        date_str: ISO 8601 date string from NewsAPI.

    Returns:
        datetime in IST timezone.
    """
    if not date_str:
        return datetime.now(IST)

    try:
        # Remove trailing Z and parse as UTC
        clean = date_str.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(clean)

        # Convert to IST
        return parsed.astimezone(IST)

    except (ValueError, TypeError):
        logger.debug(
            f"Could not parse news date '{date_str}', using current IST time."
        )
        return datetime.now(IST)


async def fetch_top_news(
    config: DataFetchConfig | None = None,
) -> list[NewsItem]:
    """Fetch and return only high-relevance news items.

    Convenience wrapper for the MCP tool "get_top_news".
    Fetches raw news → scores via news_scorer → filters by
    min_news_relevance → returns top N.

    This function imports and calls news_scorer to provide
    a complete scored result in one call.

    Args:
        config: Optional DataFetchConfig. Uses defaults if not provided.

    Returns:
        List of NewsItem objects scored and filtered by relevance.
        Only items with relevance_score >= min_news_relevance are included.
        Sorted by relevance_score descending. Capped at max_news.
    """
    effective_config = config or DataFetchConfig()

    # Fetch raw news
    raw_news = await fetch_news(effective_config)

    if not raw_news:
        return []

    # Score the news items
    from module1_data_layer.signals.news_scorer import score_news_items

    scored_news = score_news_items(raw_news)

    # Filter by minimum relevance and sort by score
    relevant_news = [
        item
        for item in scored_news
        if item.relevance_score >= effective_config.min_news_relevance
    ]
    relevant_news.sort(key=lambda n: n.relevance_score, reverse=True)

    # Cap at max_news
    return relevant_news[: effective_config.max_news]
