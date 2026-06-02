"""Tests for the SQLAlchemy repositories (watchlist + scans)."""

from __future__ import annotations

from daytrader.persistence.repositories import ScanRepository, WatchlistRepository


def test_watchlist_replace_is_idempotent(db):
    repo = WatchlistRepository(db)
    day = "2026-06-01"

    items_v1 = [
        {"symbol": "AAPL", "reason": "gap-up", "rank": 1, "gap_percent": 3.0,
         "relative_volume": 2.0, "avg_daily_volume": 5_000_000, "sentiment_score": 0.2},
        {"symbol": "MSFT", "reason": "gap-up", "rank": 2, "gap_percent": 2.5,
         "relative_volume": 1.8, "avg_daily_volume": 4_000_000, "sentiment_score": None},
    ]
    assert repo.replace_for_day(day, items_v1) == 2
    assert repo.symbols_for_day(day) == ["AAPL", "MSFT"]

    # Re-running the same day replaces rather than duplicates.
    items_v2 = [{"symbol": "NVDA", "reason": "gap-up", "rank": 1, "gap_percent": 4.0,
                 "relative_volume": 3.0, "avg_daily_volume": 9_000_000, "sentiment_score": 0.5}]
    assert repo.replace_for_day(day, items_v2) == 1
    assert repo.symbols_for_day(day) == ["NVDA"]


def test_watchlist_ordering_by_rank(db):
    repo = WatchlistRepository(db)
    day = "2026-06-02"
    repo.replace_for_day(
        day,
        [
            {"symbol": "B", "rank": 2, "avg_daily_volume": 1},
            {"symbol": "A", "rank": 1, "avg_daily_volume": 1},
            {"symbol": "C", "rank": 3, "avg_daily_volume": 1},
        ],
    )
    assert repo.symbols_for_day(day) == ["A", "B", "C"]


def test_scan_repository_records(db):
    repo = ScanRepository(db)
    day = "2026-06-01"
    n = repo.record_many(day, [("AAPL", {"gap_percent": 3.0}), ("MSFT", {"gap_percent": 2.0})])
    assert n == 2
    rows = repo.get_for_day(day)
    assert {r.symbol for r in rows} == {"AAPL", "MSFT"}
