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


# ── Opening Range Breakout (legacy breakout mode) ────────────
def test_orb_long_breakout():
    # First 3 bars (15m / 5m) define the range; later bar breaks above.
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "atr": 1.0},
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 102, "low": 100, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 103, "low": 101, "close": 103, "volume": 5000, "atr": 1.0},  # breakout
    ]
    df = _intraday(rows)
    sig = OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, entry_mode="breakout"
    ).evaluate(_snapshot(df))
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


# ── Opening Range Breakout (retest mode) ─────────────────────
# Range bars below: or_high=102, or_low=99 -> mid=100.5, size=3. ATR=1,
# touch tolerance 0.25 -> long touch zone is low <= 102.25, close >= 101.75.
_OR_BARS = [
    {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "atr": 1.0},
    {"open": 100, "high": 102, "low": 99.5, "close": 101, "volume": 1000, "atr": 1.0},
    {"open": 101, "high": 102, "low": 100, "close": 101, "volume": 1000, "atr": 1.0},
]
_BREAK_UP = {"open": 101, "high": 103, "low": 101, "close": 102.6, "volume": 1000, "atr": 1.0}
_RETEST_UP = {"open": 102.5, "high": 102.6, "low": 102.1, "close": 102.3, "volume": 1000, "atr": 1.0}


def _orb_retest(**kwargs) -> OpeningRangeBreakoutStrategy:
    return OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, volume_confirmation=False, entry_mode="retest", **kwargs
    )


def test_orb_retest_does_not_chase_break_bar():
    df = _intraday(_OR_BARS + [_BREAK_UP])
    sig = _orb_retest().evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD  # break alone is not an entry


def test_orb_retest_long_fires_at_level_with_structural_stop():
    df = _intraday(_OR_BARS + [_BREAK_UP, _RETEST_UP])
    sig = _orb_retest().evaluate(_snapshot(df))
    assert sig.direction is Direction.BUY
    assert sig.stop_hint == 100.5          # range midpoint, not or_low
    assert sig.target_hint == 105.0        # measured move from the level
    # Entry near the level: better R:R than the chased breakout close.
    assert sig.price - 102.0 < 1.0


def test_orb_retest_fires_only_once_per_session():
    later_touch = {"open": 102.3, "high": 102.4, "low": 102.2, "close": 102.3,
                   "volume": 1000, "atr": 1.0}
    df = _intraday(_OR_BARS + [_BREAK_UP, _RETEST_UP, later_touch])
    sig = _orb_retest().evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD  # first retest already happened


def test_orb_retest_invalidated_by_close_below_midpoint():
    failed = {"open": 102.5, "high": 102.5, "low": 100.0, "close": 100.2,
              "volume": 1000, "atr": 1.0}  # closes below mid 100.5
    touch = {"open": 101.9, "high": 102.2, "low": 102.0, "close": 102.1,
             "volume": 1000, "atr": 1.0}
    df = _intraday(_OR_BARS + [_BREAK_UP, failed, touch])
    sig = _orb_retest().evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD


def test_orb_retest_goes_stale_after_max_bars():
    drift = {"open": 102.6, "high": 103.0, "low": 102.4, "close": 102.8,
             "volume": 1000, "atr": 1.0}  # stays above level, never touches
    df = _intraday(_OR_BARS + [_BREAK_UP, drift, drift, _RETEST_UP])
    sig = _orb_retest(retest_max_bars=2).evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD


def test_orb_retest_short_mirror():
    break_dn = {"open": 100, "high": 100, "low": 98, "close": 98.4, "volume": 1000, "atr": 1.0}
    retest_dn = {"open": 98.5, "high": 98.9, "low": 98.3, "close": 98.8, "volume": 1000, "atr": 1.0}
    df = _intraday(_OR_BARS + [break_dn, retest_dn])
    sig = _orb_retest().evaluate(_snapshot(df))
    assert sig.direction is Direction.SELL
    assert sig.stop_hint == 100.5
    assert sig.target_hint == 96.0


def test_orb_gap_direction_blocks_counter_gap_short():
    rows = [
        {"open": 103, "high": 104, "low": 102, "close": 103, "volume": 1000, "atr": 1.0},
        {"open": 103, "high": 104, "low": 102.5, "close": 103.5, "volume": 1000, "atr": 1.0},
        {"open": 103.5, "high": 104, "low": 103, "close": 103.5, "volume": 1000, "atr": 1.0},
        {"open": 103, "high": 103.2, "low": 101, "close": 101.5, "volume": 5000, "atr": 1.0},
    ]
    df = _intraday(rows)
    daily = pd.DataFrame(
        {"open": [100], "high": [101], "low": [99], "close": [100], "atr": [1.0], "volume": 1e6},
        index=pd.date_range("2026-05-30", periods=1, freq="B", tz="UTC"),
    )
    snap = _snapshot(df)
    snap.frames[Timeframe.DAY] = daily
    orb = OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, volume_confirmation=False, require_gap_direction_match=True
    )
    sig = orb.evaluate(snap)
    assert sig.direction is Direction.HOLD


def test_orb_gap_direction_allows_aligned_long():
    rows = [
        {"open": 103, "high": 104, "low": 102, "close": 103, "volume": 1000, "atr": 1.0},
        {"open": 103, "high": 104, "low": 102.5, "close": 103.5, "volume": 1000, "atr": 1.0},
        {"open": 103.5, "high": 104, "low": 103, "close": 103.5, "volume": 1000, "atr": 1.0},
        {"open": 103.5, "high": 105, "low": 103.5, "close": 104.5, "volume": 5000, "atr": 1.0},
    ]
    df = _intraday(rows)
    daily = pd.DataFrame(
        {"open": [100], "high": [101], "low": [99], "close": [100], "atr": [1.0], "volume": 1e6},
        index=pd.date_range("2026-05-30", periods=1, freq="B", tz="UTC"),
    )
    snap = _snapshot(df)
    snap.frames[Timeframe.DAY] = daily
    orb = OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, volume_confirmation=False, require_gap_direction_match=True
    )
    sig = orb.evaluate(snap)
    assert sig.direction is Direction.BUY


def test_orb_blocks_entry_after_morning_window():
    # 09:30 + 19 bars @ 5m = 10:05 (35 min) — inside 30m window after OR forms.
    rows_early = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "atr": 1.0},
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 102, "low": 100, "close": 101, "volume": 1000, "atr": 1.0},
    ]
    breakout = {"open": 101, "high": 103, "low": 101, "close": 103, "volume": 5000, "atr": 1.0}
    df = _intraday(rows_early + [breakout] * 16)  # bar 19 at ~10:05
    sig = OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, volume_confirmation=False, max_entry_minutes_after_open=30
    ).evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD

    df_ok = _intraday(rows_early + [breakout] * 2)  # bar 5 at ~09:50
    sig_ok = OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, volume_confirmation=False, max_entry_minutes_after_open=30
    ).evaluate(_snapshot(df_ok))
    assert sig_ok.direction is Direction.BUY


def test_orb_blocks_low_volume_breakout_when_hv_filter_enabled():
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "atr": 1.0},
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 102, "low": 100, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 103, "low": 101, "close": 103, "volume": 1100, "atr": 1.0},
    ]
    df = _intraday(rows)
    orb = OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, volume_confirmation=False, min_breakout_rvol=2.5
    )
    sig = orb.evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD


def test_orb_blocks_narrow_opening_range():
    rows = [
        {"open": 100, "high": 100.05, "low": 99.95, "close": 100, "volume": 1000, "atr": 1.0},
        {"open": 100, "high": 100.08, "low": 99.98, "close": 100.02, "volume": 1000, "atr": 1.0},
        {"open": 100.02, "high": 100.1, "low": 100, "close": 100.05, "volume": 1000, "atr": 1.0},
        {"open": 100.1, "high": 101, "low": 100.08, "close": 100.9, "volume": 5000, "atr": 1.0},
    ]
    df = _intraday(rows)
    orb = OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, volume_confirmation=False, min_or_width_pct=0.3
    )
    sig = orb.evaluate(_snapshot(df))
    assert sig.direction is Direction.HOLD
    assert "narrow" in sig.rationale.lower()


def test_orb_blocks_wide_opening_range():
    rows = [
        {"open": 100, "high": 104, "low": 96, "close": 100, "volume": 1000, "atr": 1.0},
        {"open": 100, "high": 104, "low": 96, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 104, "low": 96, "close": 101, "volume": 1000, "atr": 1.0},
        {"open": 101, "high": 105, "low": 100, "close": 104, "volume": 5000, "atr": 1.0},
    ]
    df = _intraday(rows)
    daily = pd.DataFrame(
        {"open": [98], "high": [102], "low": [97], "close": [99], "atr": [2.0], "volume": [1e6]},
        index=pd.date_range("2026-05-30", periods=1, freq="B", tz="UTC"),
    )
    snap = _snapshot(df)
    snap.frames[Timeframe.DAY] = daily
    orb = OpeningRangeBreakoutStrategy(
        opening_range_minutes=15, volume_confirmation=False, max_or_width_atr=1.5
    )
    sig = orb.evaluate(snap)
    assert sig.direction is Direction.HOLD
    assert "wide" in sig.rationale.lower()


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
