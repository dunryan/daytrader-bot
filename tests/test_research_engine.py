"""Integration test for the research engine wiring (no network)."""

from __future__ import annotations

import datetime as dt

from daytrader.config.settings import Settings
from daytrader.research.research_engine import ResearchEngine
from daytrader.research.sentiment import NullSentiment
from daytrader.research.universe import resolve_universe
from tests.conftest import FakeProvider, make_daily_frame


def test_resolve_universe_named_and_literal():
    syms = resolve_universe(["sp500", "FAKE"])
    assert "AAPL" in syms
    assert "FAKE" in syms
    # Dedup: AAPL appears once even if also in nasdaq snapshot.
    assert syms.count("AAPL") == 1


def test_engine_run_persists_ranked_watchlist(db):
    settings = Settings()
    settings.research.universe = ["GOOD", "BETTER", "FLAT"]
    settings.research.max_watchlist_size = 5

    frames = {
        "GOOD": make_daily_frame(20, 2_000_000, 100.0, 103.0, 5_000_000),
        "BETTER": make_daily_frame(20, 2_000_000, 100.0, 106.0, 9_000_000),  # bigger gap + RVOL
        "FLAT": make_daily_frame(20, 2_000_000, 100.0, 100.1, 2_050_000),  # rejected
    }
    engine = ResearchEngine(settings, FakeProvider(frames), NullSentiment(), db)

    result = engine.run(trade_date=dt.date(2026, 6, 1))
    symbols = [c.symbol for c in result]

    assert symbols == ["BETTER", "GOOD"]  # ranked by composite score
    assert result[0].rank == 1
    assert "gap-up" in result[0].reason_text

    # Persisted and retrievable.
    from daytrader.persistence.repositories import WatchlistRepository

    assert WatchlistRepository(db).symbols_for_day("2026-06-01") == ["BETTER", "GOOD"]
