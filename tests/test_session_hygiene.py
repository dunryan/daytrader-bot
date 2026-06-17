"""Tests for RTH filtering and time-of-day relative volume."""

from __future__ import annotations

import pandas as pd

from daytrader.config.settings import IndicatorsConfig
from daytrader.data.data_engine import DataEngine
from daytrader.data.providers.base import Timeframe
from daytrader.data.session import filter_rth
from daytrader.strategy.util import relative_volume_tod
from tests.conftest import FakeProvider


def _bars(timestamps: list[str], volume: float = 1000.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t in timestamps])
    n = len(idx)
    return pd.DataFrame(
        {"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
         "close": [100.0] * n, "volume": [volume] * n},
        index=idx,
    )


def test_filter_rth_drops_extended_hours():
    df = _bars([
        "2026-06-01 04:00",   # pre-market
        "2026-06-01 09:25",   # pre-market
        "2026-06-01 09:30",   # first RTH bar
        "2026-06-01 12:00",
        "2026-06-01 15:55",   # last RTH bar
        "2026-06-01 16:00",   # after-hours
        "2026-06-01 19:30",   # after-hours
    ])
    out = filter_rth(df)
    assert len(out) == 3
    assert out.index[0].time().isoformat() == "09:30:00"
    assert out.index[-1].time().isoformat() == "15:55:00"


def test_filter_rth_handles_utc_index():
    # 13:30 UTC == 09:30 ET (June, EDT); 08:00 UTC == 04:00 ET pre-market.
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-06-01 08:00", tz="UTC"),
        pd.Timestamp("2026-06-01 13:30", tz="UTC"),
        pd.Timestamp("2026-06-01 19:55", tz="UTC"),
        pd.Timestamp("2026-06-01 20:00", tz="UTC"),
    ])
    df = pd.DataFrame({"open": [1.0] * 4, "high": [1.0] * 4, "low": [1.0] * 4,
                       "close": [1.0] * 4, "volume": [1.0] * 4}, index=idx)
    out = filter_rth(df)
    assert len(out) == 2  # 13:30 and 19:55 UTC are inside RTH


def test_data_engine_rth_only_fixes_opening_range_inputs():
    df = _bars([
        "2026-06-01 04:00", "2026-06-01 04:05",          # pre-market noise
        "2026-06-01 09:30", "2026-06-01 09:35", "2026-06-01 09:40",
    ])
    engine = DataEngine(FakeProvider({}), IndicatorsConfig(), rth_only=True)
    out = engine.enrich(df, Timeframe.MIN_5)
    # Pre-market rows removed: the first bar of the frame IS the 09:30 bar.
    assert out.index[0].time().isoformat() == "09:30:00"

    # Daily frames are never RTH-filtered.
    out_daily = engine.enrich(df, Timeframe.DAY)
    assert len(out_daily) == len(df)


def test_relative_volume_tod_matches_same_bar_index():
    # Two prior sessions + today; today's first bar volume is 3x the prior
    # sessions' first-bar volume, while later bars are huge (which would
    # distort a naive rolling baseline).
    rows = []
    for day in ("2026-06-01", "2026-06-02"):
        rows.append((f"{day} 09:30", 1000.0))
        rows.append((f"{day} 09:35", 50_000.0))
        rows.append((f"{day} 09:40", 50_000.0))
    rows.append(("2026-06-03 09:30", 3000.0))

    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t, _ in rows])
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
         "volume": [v for _, v in rows]},
        index=idx,
    )
    rvol = relative_volume_tod(df, sessions=10)
    # Benchmarked against prior 09:30 bars only: 3000 / mean(1000, 1000) = 3.
    assert round(rvol, 3) == 3.0


def test_relative_volume_tod_falls_back_without_history():
    df = _bars(["2026-06-01 09:30", "2026-06-01 09:35"], volume=1000.0)
    # Single session: falls back to the naive rolling measure (= 1.0 here).
    assert relative_volume_tod(df) == 1.0
