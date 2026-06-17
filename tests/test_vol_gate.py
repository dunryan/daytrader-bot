"""Tests for the VIX / SPY volatility day gate."""

from __future__ import annotations

import pandas as pd

from daytrader.config.settings import VolGateConfig
from daytrader.strategy.vol_gate import day_blocked


def _vix(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=len(closes), freq="B", tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


def test_vol_gate_blocks_high_vix():
    idx = pd.date_range("2026-06-01", periods=30, freq="B", tz="UTC")
    vix = pd.DataFrame({"close": [20.0] * 29 + [35.0]}, index=idx)
    day = idx[-1] + pd.offsets.BDay(1)
    cfg = VolGateConfig(mode="enforce", max_vix=28.0)
    blocked, details = day_blocked(day, cfg, vix_daily=vix)
    assert blocked is True
    assert details["vix_prior"] == 35.0


def test_vol_gate_off_allows():
    vix = _vix([35.0] * 60)
    day = pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=59)
    cfg = VolGateConfig(mode="off")
    blocked, _ = day_blocked(day, cfg, vix_daily=vix)
    assert blocked is False
