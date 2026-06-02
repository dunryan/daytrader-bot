"""Indicator tests against analytically known values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daytrader.data import indicators as ind


def _intraday_index(n: int, start="2026-06-01 09:30", freq="1min"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


# ── EMA ──────────────────────────────────────────────────────
def test_ema_of_constant_series_is_constant():
    s = pd.Series([10.0] * 30)
    out = ind.ema(s, 9)
    assert out.iloc[-1] == pytest.approx(10.0)


# ── RSI ──────────────────────────────────────────────────────
def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1.0, 40.0))  # strictly increasing
    out = ind.rsi(s, 14)
    assert out.iloc[-1] == pytest.approx(100.0)


def test_rsi_flat_series_is_neutral_50():
    s = pd.Series([25.0] * 30)
    out = ind.rsi(s, 14)
    assert out.iloc[-1] == pytest.approx(50.0)


# ── ATR ──────────────────────────────────────────────────────
def test_atr_constant_true_range():
    n = 50
    df = pd.DataFrame(
        {
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.0] * n,
            "volume": [1000] * n,
        }
    )
    out = ind.atr(df["high"], df["low"], df["close"], 14)
    # TR is exactly 2.0 every bar -> ATR converges to 2.0.
    assert out.iloc[-1] == pytest.approx(2.0)


# ── MACD ─────────────────────────────────────────────────────
def test_macd_constant_series_is_zero():
    s = pd.Series([100.0] * 60)
    out = ind.macd(s, 12, 26, 9)
    assert out["macd"].iloc[-1] == pytest.approx(0.0)
    assert out["macd_hist"].iloc[-1] == pytest.approx(0.0)


# ── VWAP ─────────────────────────────────────────────────────
def test_vwap_volume_weighting_and_daily_reset():
    idx = pd.to_datetime(
        [
            "2026-06-01 09:30",
            "2026-06-01 09:31",
            "2026-06-02 09:30",  # new day -> reset
        ]
    ).tz_localize("UTC")
    df = pd.DataFrame(
        {
            "high": [10.0, 20.0, 30.0],
            "low": [10.0, 20.0, 30.0],
            "close": [10.0, 20.0, 30.0],
            "volume": [100, 300, 50],
        },
        index=idx,
    )
    v = ind.vwap(df)
    assert v.iloc[0] == pytest.approx(10.0)
    # (10*100 + 20*300) / 400 = 17.5
    assert v.iloc[1] == pytest.approx(17.5)
    # New day resets: only the 30.0 bar counts.
    assert v.iloc[2] == pytest.approx(30.0)


# ── Pivot points ─────────────────────────────────────────────
def test_pivot_points_known_values():
    p = ind.pivot_points(prev_high=110.0, prev_low=90.0, prev_close=100.0)
    assert p["pivot"] == pytest.approx(100.0)
    assert p["r1"] == pytest.approx(110.0)
    assert p["s1"] == pytest.approx(90.0)
    assert p["r2"] == pytest.approx(120.0)
    assert p["s2"] == pytest.approx(80.0)


# ── Swing levels ─────────────────────────────────────────────
def test_swing_levels_detects_peak_and_trough():
    # A clear peak at index 5 and trough at index 11.
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10, 9, 10, 11, 12, 13, 14]
    lows = [h - 1 for h in highs]
    lows[11] = 1  # trough
    df = pd.DataFrame({"high": highs, "low": lows})
    levels = ind.swing_levels(df, window=3)
    assert 20.0 in levels["resistance"]
    assert 1.0 in levels["support"]


# ── add_indicators integration ───────────────────────────────
def test_add_indicators_appends_expected_columns():
    n = 260
    idx = _intraday_index(n)
    prices = pd.Series(np.linspace(100, 130, n))
    df = pd.DataFrame(
        {
            "open": prices.values,
            "high": (prices + 0.5).values,
            "low": (prices - 0.5).values,
            "close": prices.values,
            "volume": np.full(n, 10_000),
        },
        index=idx,
    )
    out = ind.add_indicators(df, ema_periods=[9, 21, 50, 200])
    for col in ["ema_9", "ema_21", "ema_50", "ema_200", "rsi", "atr", "atr_pct",
                "macd", "macd_signal", "macd_hist", "vwap"]:
        assert col in out.columns
    # With 260 rows, EMA-200 is populated at the tail.
    assert not np.isnan(out["ema_200"].iloc[-1])
    # Rising market -> RSI elevated.
    assert out["rsi"].iloc[-1] > 70
