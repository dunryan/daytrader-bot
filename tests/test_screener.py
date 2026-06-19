"""Tests for the pure screener metric math and filter predicate."""

from __future__ import annotations

from daytrader.config.settings import ResearchFilters
from daytrader.research.screener import (
    Screener,
    apply_tod_premarket_rvol,
    compute_metrics,
    passes_filters,
)
from tests.conftest import FakeProvider, make_daily_frame


def test_compute_metrics_gap_and_rvol():
    # 20 prior days @ vol 1,000,000, prev close 100; today opens at 105 on 3M vol.
    df = make_daily_frame(
        days=20, base_volume=1_000_000, prev_close=100.0, day_open=105.0, today_volume=3_000_000
    )
    m = compute_metrics("TEST", df)
    assert m is not None
    assert m.prev_close == 100.0
    assert m.day_open == 105.0
    assert round(m.gap_percent, 2) == 5.0
    assert m.avg_daily_volume == 1_000_000
    assert round(m.relative_volume, 2) == 3.0


def test_compute_metrics_insufficient_history():
    df = make_daily_frame(days=0, base_volume=1, prev_close=10, day_open=10, today_volume=1)
    # Only one usable row -> cannot compute a gap.
    assert compute_metrics("TEST", df.iloc[:1]) is None


def test_passes_filters_accepts_strong_gapper():
    df = make_daily_frame(
        days=20, base_volume=2_000_000, prev_close=50.0, day_open=53.0, today_volume=5_000_000
    )
    m = compute_metrics("TEST", df)
    ok, reasons = passes_filters(m, ResearchFilters())
    assert ok is True
    assert any("gap-up" in r for r in reasons)
    assert any("RVOL" in r for r in reasons)


def test_passes_filters_rejects_low_volume():
    df = make_daily_frame(
        days=20, base_volume=100_000, prev_close=50.0, day_open=53.0, today_volume=120_000
    )
    m = compute_metrics("TEST", df)
    ok, reasons = passes_filters(m, ResearchFilters())  # min_avg_daily_volume=1M
    assert ok is False
    assert "avg vol" in reasons[0]


def test_passes_filters_rejects_small_gap():
    df = make_daily_frame(
        days=20, base_volume=3_000_000, prev_close=100.0, day_open=100.5, today_volume=6_000_000
    )
    m = compute_metrics("TEST", df)
    ok, reasons = passes_filters(m, ResearchFilters())  # min_gap_percent=2.0
    assert ok is False
    assert "gap" in reasons[0]


def test_screener_end_to_end_with_fake_provider():
    frames = {
        "GOOD": make_daily_frame(20, 2_000_000, 100.0, 104.0, 6_000_000),  # passes
        "FLAT": make_daily_frame(20, 2_000_000, 100.0, 100.2, 2_100_000),  # gap too small
        "THIN": make_daily_frame(20, 50_000, 100.0, 110.0, 200_000),  # illiquid
    }
    screener = Screener(FakeProvider(frames), ResearchFilters())
    passed, all_metrics = screener.screen(list(frames.keys()))
    assert {m.symbol for m in all_metrics} == {"GOOD", "FLAT", "THIN"}
    assert [m.symbol for m in passed] == ["GOOD"]
