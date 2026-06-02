"""Deterministic unit tests for each strategy's decision logic."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.strategy.base import Direction
from daytrader.strategy.momentum_scalp import MomentumScalpStrategy
from daytrader.strategy.orb import OpeningRangeBreakoutStrategy
from daytrader.strategy.vwap_pullback import VwapPullbackStrategy


def _snapshot(df: pd.DataFrame, symbol: str = "TEST") -> MarketSnapshot:
    snap = MarketSnapshot(symbol=symbol, as_of=dt.datetime.now(dt.timezone.utc))
    snap.frames[Timeframe.MIN_5] = df
    return snap


def _intraday(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01 09:30", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx)


# ── VWAP pullback ────────────────────────────────────────────
def test_vwap_pullback_long():
    df = _intraday(
        [
            {"close": 100.0, "vwap": 100.0, "ema_21": 99.0, "atr": 1.0, "high": 100, "low": 99, "open": 99, "volume": 1000},
            {"close": 101.0, "vwap": 100.8, "ema_21": 100.0, "atr": 1.0, "high": 101, "low": 100, "open": 100, "volume": 1000},
        ]
    )
    sig = VwapPullbackStrategy(trend_ema=21, max_distance_from_vwap_atr=0.5).evaluate(_snapshot(df))
    assert sig.direction is Direction.BUY
    assert sig.stop_hint is not None and sig.stop_hint < sig.price


def test_vwap_pullback_holds_when_far_from_vwap():
    df = _intraday(
        [
            {"close": 100.0, "vwap": 100.0, "ema_21": 99.0, "atr": 1.0, "high": 100, "low": 99, "open": 99, "volume": 1000},
            {"close": 105.0, "vwap": 100.0, "ema_21": 100.0, "atr": 1.0, "high": 105, "low": 100, "open": 100, "volume": 1000},
        ]
    )
    sig = VwapPullbackStrategy().evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD


def test_vwap_pullback_short_in_downtrend():
    df = _intraday(
        [
            {"close": 100.0, "vwap": 100.0, "ema_21": 101.0, "atr": 1.0, "high": 100, "low": 99, "open": 100, "volume": 1000},
            {"close": 99.0, "vwap": 99.2, "ema_21": 100.0, "atr": 1.0, "high": 100, "low": 99, "open": 100, "volume": 1000},
        ]
    )
    sig = VwapPullbackStrategy().evaluate(_snapshot(df))
    assert sig.direction is Direction.SELL


# ── Opening Range Breakout ───────────────────────────────────
def test_orb_long_breakout():
    # First 3 bars (15m / 5m) define the range; later bar breaks above.
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "atr": 1.0},
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 102, "low": 100, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 103, "low": 101, "close": 103, "volume": 5000, "atr": 1.0},  # breakout
    ]
    df = _intraday(rows)
    sig = OpeningRangeBreakoutStrategy(opening_range_minutes=15).evaluate(_snapshot(df))
    assert sig.direction is Direction.BUY
    assert sig.indicators["or_high"] == 102.0
    assert sig.stop_hint == sig.indicators["or_low"]


def test_orb_holds_inside_range_window():
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "atr": 1.0},
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "atr": 1.0},
    ]
    df = _intraday(rows)  # only 2 bars, still inside the 3-bar OR window
    sig = OpeningRangeBreakoutStrategy(opening_range_minutes=15).evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD


# ── Momentum scalp ───────────────────────────────────────────
def test_momentum_long_on_fresh_cross_with_volume():
    rows = [
        {"close": 100, "ema_9": 99.5, "ema_21": 100.0, "atr": 1.0, "high": 100, "low": 99, "open": 99, "volume": 1000},
        {"close": 100, "ema_9": 99.8, "ema_21": 100.0, "atr": 1.0, "high": 100, "low": 99, "open": 99, "volume": 1000},  # fast below
        {"close": 102, "ema_9": 100.5, "ema_21": 100.0, "atr": 1.0, "high": 102, "low": 100, "open": 100, "volume": 5000},  # cross up + vol
    ]
    df = _intraday(rows)
    sig = MomentumScalpStrategy(min_relative_volume=2.0).evaluate(_snapshot(df))
    assert sig.direction is Direction.BUY


def test_momentum_holds_without_volume_spike():
    rows = [
        {"close": 100, "ema_9": 99.5, "ema_21": 100.0, "atr": 1.0, "high": 100, "low": 99, "open": 99, "volume": 1000},
        {"close": 100, "ema_9": 99.8, "ema_21": 100.0, "atr": 1.0, "high": 100, "low": 99, "open": 99, "volume": 1000},
        {"close": 102, "ema_9": 100.5, "ema_21": 100.0, "atr": 1.0, "high": 102, "low": 100, "open": 100, "volume": 1000},  # no spike
    ]
    df = _intraday(rows)
    sig = MomentumScalpStrategy(min_relative_volume=2.0).evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD
