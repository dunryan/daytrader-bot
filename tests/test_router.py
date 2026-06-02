"""Tests for strategy construction from config and the router pipeline."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.config.settings import Settings
from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.persistence.repositories import SignalRepository
from daytrader.strategy.base import Direction
from daytrader.strategy.router import StrategyRouter, build_strategies
from daytrader.strategy.vwap_pullback import VwapPullbackStrategy


def test_build_strategies_respects_toggles():
    # Settings() defaults have all strategies OFF (safe default); enable two.
    s = Settings()
    s.strategies.vwap_pullback.enabled = True
    s.strategies.opening_range_breakout.enabled = True
    names = {strat.name for strat in build_strategies(s)}
    assert names == {"vwap_pullback", "opening_range_breakout"}

    s.strategies.momentum_scalp.enabled = True
    s.strategies.vwap_pullback.enabled = False
    names = {strat.name for strat in build_strategies(s)}
    assert names == {"opening_range_breakout", "momentum_scalp"}


def _vwap_long_snapshot() -> MarketSnapshot:
    idx = pd.date_range("2026-06-01 09:30", periods=2, freq="5min", tz="UTC")
    df = pd.DataFrame(
        [
            {"close": 100.0, "vwap": 100.0, "ema_21": 99.0, "atr": 1.0},
            {"close": 101.0, "vwap": 100.8, "ema_21": 100.0, "atr": 1.0},
        ],
        index=idx,
    )
    snap = MarketSnapshot(symbol="AAPL", as_of=dt.datetime.now(dt.timezone.utc))
    snap.frames[Timeframe.MIN_5] = df
    return snap


def test_router_evaluates_and_persists(db):
    repo = SignalRepository(db)
    router = StrategyRouter([VwapPullbackStrategy()], signal_repo=repo)

    signals = router.evaluate(
        {"AAPL": _vwap_long_snapshot()}, trade_date=dt.date(2026, 6, 1)
    )
    assert len(signals) == 1
    assert signals[0].direction is Direction.BUY

    persisted = repo.get_for_day("2026-06-01")
    assert len(persisted) == 1
    assert persisted[0].symbol == "AAPL"
    assert persisted[0].strategy == "vwap_pullback"
    assert persisted[0].direction == "BUY"


def test_router_isolates_failing_strategy(db):
    class Boom(VwapPullbackStrategy):
        name = "boom"

        def evaluate(self, snapshot):  # noqa: ANN001
            raise RuntimeError("kaboom")

    router = StrategyRouter([Boom(), VwapPullbackStrategy()], signal_repo=SignalRepository(db))
    signals = router.evaluate({"AAPL": _vwap_long_snapshot()}, trade_date=dt.date(2026, 6, 1))
    # The failing strategy is swallowed; the good one still fires.
    assert len(signals) == 1
    assert signals[0].strategy == "vwap_pullback"
