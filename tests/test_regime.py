"""Tests for the market-regime classifier and router gating."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from daytrader.config.settings import RegimeFilterConfig
from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.strategy.regime import Regime, classify
from daytrader.strategy.router import StrategyRouter
from daytrader.strategy.vwap_pullback import VwapPullbackStrategy


def _daily(atr_pct_today: float, n: int = 80) -> pd.DataFrame:
    idx = pd.date_range(end="2026-06-01", periods=n, freq="B", tz="UTC")
    atr_pct = np.linspace(1.0, 2.0, n)
    atr_pct[-1] = atr_pct_today
    close = np.full(n, 100.0)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close,
         "volume": 1e6, "atr": atr_pct * close / 100.0, "atr_pct": atr_pct},
        index=idx,
    )


def _intraday(
    closes: list[float], vwap: float, day: str = "2026-06-02", spread: float = 0.2
) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", periods=len(closes), freq="5min", tz="UTC")
    closes_arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"open": closes_arr, "high": closes_arr + spread, "low": closes_arr - spread,
         "close": closes_arr, "volume": 1000.0, "vwap": vwap},
        index=idx,
    )


def _snap(intraday: pd.DataFrame, daily: pd.DataFrame) -> MarketSnapshot:
    snap = MarketSnapshot(symbol="TEST", as_of=dt.datetime.now(dt.timezone.utc))
    snap.frames[Timeframe.MIN_5] = intraday
    snap.frames[Timeframe.DAY] = daily
    return snap


def test_trend_regime_on_expanding_one_sided_session():
    # Session range ~4 vs daily ATR ~2 (extension 2.0), all closes above VWAP.
    daily = _daily(atr_pct_today=2.0)
    intraday = _intraday([100, 101, 102, 103, 104], vwap=99.0)
    regime, details = classify(_snap(intraday, daily))
    assert regime is Regime.TREND
    assert details["range_extension"] > 0.8
    assert details["vwap_one_sidedness"] >= 0.7


def test_balanced_regime_on_two_sided_rotation():
    daily = _daily(atr_pct_today=1.5)
    # Tight rotation around VWAP: half above, half below, tiny range.
    intraday = _intraday([100.1, 99.9, 100.1, 99.9, 100.0, 100.1], vwap=100.0)
    regime, _ = classify(_snap(intraday, daily))
    assert regime is Regime.BALANCED


def test_quiet_regime_on_compressed_volatility():
    # Today's ATR%% is the lowest of the lookback window and the session is dead.
    daily = _daily(atr_pct_today=0.5)
    intraday = _intraday([100.0, 100.0, 100.0, 100.0], vwap=100.0, spread=0.0)
    regime, details = classify(_snap(intraday, daily))
    assert regime is Regime.QUIET
    assert details["atr_percentile"] <= 0.2


def _vwap_long_snapshot_balanced() -> MarketSnapshot:
    """A snapshot that fires vwap_pullback and classifies as balanced."""
    daily = _daily(atr_pct_today=1.5)
    idx = pd.date_range("2026-06-02 09:30", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame(
        [
            {"open": 99.9, "high": 100.1, "low": 99.8, "close": 100.0, "volume": 1000,
             "vwap": 100.0, "ema_21": 99.0, "atr": 1.0},
            {"open": 100.0, "high": 100.2, "low": 99.8, "close": 99.9, "volume": 1000,
             "vwap": 100.0, "ema_21": 99.2, "atr": 1.0},
            {"open": 99.9, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 1000,
             "vwap": 100.0, "ema_21": 99.4, "atr": 1.0},
            {"open": 100.0, "high": 100.4, "low": 99.9, "close": 100.3, "volume": 1000,
             "vwap": 100.1, "ema_21": 99.6, "atr": 1.0},
        ],
        index=idx,
    )
    snap = MarketSnapshot(symbol="AAPL", as_of=dt.datetime.now(dt.timezone.utc))
    snap.frames[Timeframe.MIN_5] = df
    snap.frames[Timeframe.DAY] = daily
    return snap


def test_router_enforce_blocks_disallowed_regime():
    cfg = RegimeFilterConfig(mode="enforce", allowed={"vwap_pullback": ["trend"]})
    router = StrategyRouter([VwapPullbackStrategy()], regime_config=cfg)
    # Snapshot classifies balanced; vwap_pullback only allowed in trend -> blocked.
    signals = router.evaluate_snapshot(_vwap_long_snapshot_balanced())
    assert signals == []


def test_router_shadow_annotates_but_passes():
    cfg = RegimeFilterConfig(mode="shadow", allowed={"vwap_pullback": ["trend"]})
    router = StrategyRouter([VwapPullbackStrategy()], regime_config=cfg)
    signals = router.evaluate_snapshot(_vwap_long_snapshot_balanced())
    assert len(signals) == 1
    assert signals[0].indicators.get("regime_block") is True
    assert signals[0].indicators.get("regime") == "balanced"


def test_router_allowed_regime_passes_clean():
    cfg = RegimeFilterConfig(mode="enforce", allowed={"vwap_pullback": ["balanced"]})
    router = StrategyRouter([VwapPullbackStrategy()], regime_config=cfg)
    signals = router.evaluate_snapshot(_vwap_long_snapshot_balanced())
    assert len(signals) == 1
    assert "regime_block" not in signals[0].indicators
