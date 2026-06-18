"""Tests for premarket RVOL screener parity in backtests."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.backtest.screener_parity import (
    intersect_eligible_days,
    parse_cutoff_time,
    premarket_rvol_eligible_days,
    premarket_volume_by_day,
)


def _pm_bars(day: str, volumes: list[tuple[str, float]]) -> pd.DataFrame:
    """Build extended-hours bars: list of (HH:MM ET as UTC label, volume)."""
    rows = []
    for hhmm, vol in volumes:
        h, m = map(int, hhmm.split(":"))
        ts = pd.Timestamp(f"{day} {h:02d}:{m:02d}", tz="America/New_York").tz_convert("UTC")
        rows.append((ts, vol))
    idx = pd.DatetimeIndex([r[0] for r in rows])
    vol = [r[1] for r in rows]
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": vol},
        index=idx,
    )


def _daily(days: list[str], volumes: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime(days, utc=True)
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": volumes,
        },
        index=idx,
    )


def test_premarket_volume_sums_extended_hours_before_cutoff():
    df = _pm_bars("2026-06-02", [("06:30", 500_000), ("06:55", 500_000), ("09:30", 999_999)])
    vols = premarket_volume_by_day(df, cutoff=dt.time(7, 0))
    day = pd.Timestamp("2026-06-02", tz="UTC")
    assert vols[day] == 1_000_000
    assert vols[day] != 1_999_999  # RTH bar excluded


def test_premarket_rvol_tod_mode():
    daily = _daily(
        ["2026-05-30", "2026-06-02", "2026-06-03", "2026-06-04"],
        [1_000_000, 1_000_000, 1_000_000, 1_000_000],
    )
    pm = pd.concat([
        _pm_bars("2026-06-02", [("06:00", 500_000), ("06:45", 500_000)]),
        _pm_bars("2026-06-03", [("06:00", 500_000), ("06:45", 500_000)]),
        _pm_bars("2026-06-04", [("06:00", 900_000), ("06:45", 900_000)]),
    ]).sort_index()
    eligible = premarket_rvol_eligible_days(daily, pm, min_rvol=1.5, cutoff=dt.time(7, 0), mode="tod")
    assert pd.Timestamp("2026-06-04", tz="UTC") in eligible
    assert pd.Timestamp("2026-06-03", tz="UTC") not in eligible


def test_premarket_rvol_screener_mode_uses_full_day_avg():
    daily = _daily(
        ["2026-05-30", "2026-06-02", "2026-06-03"],
        [1_000_000, 1_000_000, 2_000_000],
    )
    pm = pd.concat([
        _pm_bars("2026-06-02", [("06:00", 750_000), ("06:45", 750_000)]),
        _pm_bars("2026-06-03", [("06:00", 1_500_000), ("06:45", 1_500_000)]),
    ]).sort_index()
    eligible = premarket_rvol_eligible_days(
        daily, pm, min_rvol=1.5, cutoff=dt.time(7, 0), mode="screener"
    )
    assert pd.Timestamp("2026-06-03", tz="UTC") in eligible  # 3M / 1M avg daily


def test_intersect_eligible_days():
    day = pd.Timestamp("2026-06-02", tz="UTC")
    base = {"AAPL": {day}, "TSLA": {day}}
    extra = {"AAPL": set(), "TSLA": {day}}
    out = intersect_eligible_days(base, extra, ["AAPL", "TSLA"])
    assert out["AAPL"] == set()
    assert out["TSLA"] == {day}


def test_parse_cutoff_time():
    assert parse_cutoff_time("07:00") == dt.time(7, 0)
