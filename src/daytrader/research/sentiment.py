"""News sentiment scoring (the soft, context-only signal).

Important framing: sentiment here is a *context filter / tie-breaker*, never a
standalone trade trigger. We fetch recent headlines from Finnhub or NewsAPI and
score them with a lightweight lexicon. It is intentionally simple and
transparent rather than a black-box NLP model.

All providers degrade gracefully: missing API key or a network error yields a
neutral result (score ``None``) instead of raising, so research never blocks on
sentiment being unavailable.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from daytrader.config.settings import Settings
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Tiny finance-flavored sentiment lexicon. Transparent and tunable.
_POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgraded", "bullish", "record", "growth", "gains", "gain",
    "outperform", "strong", "jump", "jumps", "rises", "rise", "boost", "wins",
    "profit", "profits", "raise", "raised", "buy", "breakout", "positive",
}
_NEGATIVE = {
    "miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops",
    "downgrade", "downgraded", "bearish", "loss", "losses", "weak", "slump",
    "cut", "cuts", "lawsuit", "probe", "recall", "warns", "warning", "decline",
    "sell", "selloff", "negative", "fraud", "bankruptcy", "slumps", "tumble",
}


@dataclass
class SentimentResult:
    """Outcome of scoring a symbol's recent headlines."""

    symbol: str
    score: float | None  # normalized -1..+1, or None if insufficient data
    article_count: int

    @property
    def label(self) -> str:
        if self.score is None:
            return "n/a"
        if self.score > 0.15:
            return "positive"
        if self.score < -0.15:
            return "negative"
        return "neutral"


def score_headlines(headlines: list[str], min_articles: int) -> tuple[float | None, int]:
    """Score a list of headlines with the lexicon.

    Returns ``(score, count)``; score is ``None`` when fewer than
    ``min_articles`` headlines are available.
    """
    count = len(headlines)
    if count < min_articles:
        return None, count
    pos = neg = 0
    for line in headlines:
        tokens = {t.strip(".,!?:;\"'()").lower() for t in line.split()}
        pos += len(tokens & _POSITIVE)
        neg += len(tokens & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0, count
    return (pos - neg) / total, count


class SentimentProvider(ABC):
    """Returns a sentiment score for a symbol over a lookback window."""

    @abstractmethod
    def score_symbol(self, symbol: str, lookback_hours: int, min_articles: int) -> SentimentResult:
        ...


class NullSentiment(SentimentProvider):
    """No-op provider used when sentiment is disabled or unconfigured."""

    def score_symbol(self, symbol: str, lookback_hours: int, min_articles: int) -> SentimentResult:
        return SentimentResult(symbol=symbol, score=None, article_count=0)


class FinnhubSentiment(SentimentProvider):
    """Scores Finnhub company-news headlines."""

    BASE = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def score_symbol(self, symbol: str, lookback_hours: int, min_articles: int) -> SentimentResult:
        now = dt.datetime.now(dt.timezone.utc)
        frm = (now - dt.timedelta(hours=lookback_hours)).date().isoformat()
        to = now.date().isoformat()
        try:
            resp = requests.get(
                self.BASE,
                params={"symbol": symbol, "from": frm, "to": to, "token": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            articles = resp.json() or []
            headlines = [a.get("headline", "") for a in articles if a.get("headline")]
        except Exception:  # noqa: BLE001
            logger.warning("Finnhub sentiment fetch failed for %s", symbol, exc_info=True)
            return SentimentResult(symbol=symbol, score=None, article_count=0)
        score, count = score_headlines(headlines, min_articles)
        return SentimentResult(symbol=symbol, score=score, article_count=count)


class NewsApiSentiment(SentimentProvider):
    """Scores NewsAPI 'everything' headlines for a symbol."""

    BASE = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def score_symbol(self, symbol: str, lookback_hours: int, min_articles: int) -> SentimentResult:
        now = dt.datetime.now(dt.timezone.utc)
        frm = (now - dt.timedelta(hours=lookback_hours)).isoformat()
        try:
            resp = requests.get(
                self.BASE,
                params={
                    "q": symbol,
                    "from": frm,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 50,
                    "apiKey": self.api_key,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", []) or []
            headlines = [a.get("title", "") for a in articles if a.get("title")]
        except Exception:  # noqa: BLE001
            logger.warning("NewsAPI sentiment fetch failed for %s", symbol, exc_info=True)
            return SentimentResult(symbol=symbol, score=None, article_count=0)
        score, count = score_headlines(headlines, min_articles)
        return SentimentResult(symbol=symbol, score=score, article_count=count)


def get_sentiment_provider(settings: Settings) -> SentimentProvider:
    """Build the configured sentiment provider, falling back to Null."""
    cfg = settings.research.sentiment
    if not cfg.enabled or cfg.provider == "none":
        return NullSentiment()
    if cfg.provider == "finnhub":
        key = settings.secrets.finnhub_api_key
        if key:
            return FinnhubSentiment(key)
        logger.warning("Sentiment provider 'finnhub' selected but FINNHUB_API_KEY is missing.")
    elif cfg.provider == "newsapi":
        key = settings.secrets.newsapi_api_key
        if key:
            return NewsApiSentiment(key)
        logger.warning("Sentiment provider 'newsapi' selected but NEWSAPI_API_KEY is missing.")
    return NullSentiment()
