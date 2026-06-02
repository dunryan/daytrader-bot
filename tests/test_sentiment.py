"""Tests for lexicon sentiment scoring and provider selection."""

from __future__ import annotations

from daytrader.config.settings import Secrets, Settings
from daytrader.research.sentiment import (
    FinnhubSentiment,
    NullSentiment,
    get_sentiment_provider,
    score_headlines,
)


def test_score_headlines_positive():
    headlines = ["Stock surges as company beats earnings", "Analyst upgrade boosts shares"]
    score, count = score_headlines(headlines, min_articles=1)
    assert count == 2
    assert score is not None and score > 0


def test_score_headlines_negative():
    headlines = ["Shares plunge after earnings miss", "Downgrade and lawsuit weigh on stock"]
    score, count = score_headlines(headlines, min_articles=1)
    assert score is not None and score < 0


def test_score_headlines_insufficient_articles():
    score, count = score_headlines(["one positive beat"], min_articles=3)
    assert score is None
    assert count == 1


def test_get_provider_defaults_to_null_without_keys():
    s = Settings()  # sentiment enabled, provider finnhub, but no key
    s.secrets = Secrets(_env_file=None)  # ignore any developer .env
    assert isinstance(get_sentiment_provider(s), NullSentiment)


def test_get_provider_finnhub_with_key():
    s = Settings()
    s.secrets.finnhub_api_key = "fake-key"
    assert isinstance(get_sentiment_provider(s), FinnhubSentiment)


def test_disabled_sentiment_is_null():
    s = Settings()
    s.research.sentiment.enabled = False
    assert isinstance(get_sentiment_provider(s), NullSentiment)
