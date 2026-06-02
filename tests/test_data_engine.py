"""Tests for the DataEngine enrichment and snapshot assembly (no network)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from daytrader.config.settings import IndicatorsConfig
from daytrader.data.data_engine import DataEngine, MarketSnapshot
from daytrader.data.providers.base import Timeframe
from tests.conftest import FakeProvider


def _frame(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range(end=dt.datetime.now(dt.timezone.utc), periods=n, freq="5min")
    prices = pd.Series(np.linspace(50, 60, n))
    return pd.DataFrame(
        {
            "open": prices.values,
            "high": (prices + 0.3).values,
            "low": (prices - 0.3).values,
            "close": prices.values,
            "volume": np.random.randint(5000, 20000, n).astype(float),
        },
        index=idx,
    )


def test_enrich_adds_indicator_columns():
    engine = DataEngine(FakeProvider({}), IndicatorsConfig())
    out = engine.enrich(_frame(), Timeframe.MIN_5)
    assert "vwap" in out.columns
    assert "ema_21" in out.columns
    assert "macd_hist" in out.columns


def test_daily_frame_has_no_vwap():
    engine = DataEngine(FakeProvider({}), IndicatorsConfig())
    out = engine.enrich(_frame(), Timeframe.DAY)
    assert "vwap" not in out.columns


def test_build_snapshot_assembles_frames_and_levels():
    frame = _frame(120)
    provider = FakeProvider({"AAPL": frame})
    engine = DataEngine(provider, IndicatorsConfig())

    snap = engine.build_snapshot("AAPL", [Timeframe.DAY, Timeframe.MIN_5])
    assert isinstance(snap, MarketSnapshot)
    assert Timeframe.MIN_5 in snap.frames
    # Pivots derived from the (fake) daily frame.
    assert "pivot" in snap.pivots
    # Swing levels detected on intraday frame.
    assert isinstance(snap.support, list)
    assert isinstance(snap.resistance, list)

    price = snap.latest_price()
    assert price is not None and price > 0
    assert snap.indicator(Timeframe.MIN_5, "ema_9") is not None
