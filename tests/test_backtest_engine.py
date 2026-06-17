"""Tests for the event-driven backtest engine on synthetic bars."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.backtest.engine import BacktestEngine, gap_eligible_days
from daytrader.backtest.labels import signals_to_frame
from daytrader.backtest.metrics import compute_metrics, metrics_by_strategy
from daytrader.config.settings import Settings
from daytrader.data.data_engine import MarketSnapshot
from daytrader.strategy.base import Direction, Signal, Strategy


class FireOnceStrategy(Strategy):
    """Emits exactly one BUY per symbol at a fixed bar index."""

    name = "fire_once"

    def __init__(self, fire_at_index: int = 3, stop_offset: float = 2.0) -> None:
        self.fire_at_index = fire_at_index
        self.stop_offset = stop_offset
        self.fired: set[str] = set()

    def evaluate(self, snapshot: MarketSnapshot) -> Signal:
        df = snapshot.frame(self.timeframe)
        if df is None or len(df) - 1 != self.fire_at_index or snapshot.symbol in self.fired:
            return self._hold(snapshot.symbol, "waiting")
        self.fired.add(snapshot.symbol)
        close = float(df["close"].iloc[-1])
        return Signal(
            symbol=snapshot.symbol, strategy=self.name, direction=Direction.BUY,
            price=close, confidence=0.9, rationale="synthetic", timeframe=self.timeframe.value,
            indicators={"atr": 1.0, "close": close},
            stop_hint=close - self.stop_offset, target_hint=None,
        )


def _session(day: str, closes: list[float], spread: float = 0.1) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", periods=len(closes), freq="5min", tz="UTC")
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {"open": c.values, "high": (c + spread).values, "low": (c - spread).values,
         "close": c.values, "volume": 1000.0},
        index=idx,
    )


def _settings(tp_method: str = "fixed") -> Settings:
    s = Settings()
    s.risk.take_profit.method = tp_method
    s.risk.take_profit.risk_reward_ratio = 2.0
    s.strategies.regime_filter.mode = "off"
    return s


def _engine(settings: Settings, fire_at: int = 3, stop_offset: float = 2.0) -> BacktestEngine:
    return BacktestEngine(
        settings, [FireOnceStrategy(fire_at, stop_offset)],
        spread_bps=0.0, atr_impact_coeff=0.0, warmup_bars=0,
    )


def test_entry_fills_next_bar_open_not_signal_close():
    # Signal on bar 3 (close 105); bar 4 opens at 107 — fill must be 107.
    closes = [100, 101, 102, 105, 107, 107, 107, 107]
    df = _session("2026-06-01", closes)
    df.iloc[4, df.columns.get_loc("open")] = 107.0
    engine = _engine(_settings())
    result = engine.run({"TEST": df}, daily={})
    assert len(result.trades) == 1
    assert result.trades[0].entry_price == 107.0  # no signal-close fill fantasy


def test_stop_first_when_bar_touches_both():
    # Entry at bar4 open=100 with stop 98 and TP 104 (RR 2 on $2 risk).
    # Bar 5 spans 97..105: both touched -> conservative engine takes the stop.
    closes = [100, 100, 100, 100, 100, 101, 100, 100]
    df = _session("2026-06-01", closes)
    df.iloc[5, df.columns.get_loc("high")] = 105.0
    df.iloc[5, df.columns.get_loc("low")] = 97.0
    engine = _engine(_settings())
    result = engine.run({"TEST": df}, daily={})
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "SL"
    assert trade.pnl < 0


def test_take_profit_exit_and_label():
    closes = [100, 100, 100, 100, 100, 102, 105, 100]
    df = _session("2026-06-01", closes)
    engine = _engine(_settings())
    result = engine.run({"TEST": df}, daily={})
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "TP"
    assert result.trades[0].pnl > 0
    labeled = [s for s in result.signals if s.label is not None]
    assert len(labeled) == 1 and labeled[0].label == 1


def test_eod_flatten_closes_open_positions():
    # Price never hits stop or target -> flattened on the session's last bar.
    closes = [100, 100, 100, 100, 100.5, 100.5, 100.5]
    df = _session("2026-06-01", closes)
    engine = _engine(_settings())
    result = engine.run({"TEST": df}, daily={})
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "EOD"


def test_gap_through_stop_fills_at_open():
    closes = [100, 100, 100, 100, 100, 95, 95, 95]
    df = _session("2026-06-01", closes)
    engine = _engine(_settings())
    result = engine.run({"TEST": df}, daily={})
    trade = result.trades[0]
    assert trade.exit_reason == "SL"
    assert trade.exit_price == 95.0  # gapped open, not the 98 stop price


def test_slippage_is_adverse_on_both_legs():
    closes = [100, 100, 100, 100, 100, 102, 105, 100]
    df = _session("2026-06-01", closes)
    settings = _settings()
    engine = BacktestEngine(
        settings, [FireOnceStrategy(3, 2.0)],
        spread_bps=20.0, atr_impact_coeff=0.0, warmup_bars=0,
    )
    result = engine.run({"TEST": df}, daily={})
    trade = result.trades[0]
    assert trade.entry_price > 100.0   # paid up on entry
    # Exit received less than the raw exit reference.
    raw_exit = trade.exit_price / (1 - 10.0 / 10_000.0)
    assert trade.exit_price < raw_exit + 1e-9


def test_labels_frame_has_features_and_label():
    closes = [100, 100, 100, 100, 100, 102, 105, 100]
    df = _session("2026-06-01", closes)
    engine = _engine(_settings())
    result = engine.run({"TEST": df}, daily={})
    frame = signals_to_frame(result.signals)
    assert len(frame) == 1
    assert frame["label"].iloc[0] == 1
    assert any(c.startswith("feat_") for c in frame.columns)


def test_metrics_compute_sane_values():
    closes = [100, 100, 100, 100, 100, 102, 105, 100]
    df = _session("2026-06-01", closes)
    engine = _engine(_settings())
    result = engine.run({"TEST": df}, daily={})
    overall = compute_metrics(result.trades, result.equity_curve)
    assert overall["total_trades"] == 1
    assert overall["win_rate"] == 1.0
    assert overall["expectancy"] > 0
    per_strat = metrics_by_strategy(result.trades)
    assert "fire_once" in per_strat


def test_eligible_days_blocks_signal_generation():
    closes = [100, 100, 100, 100, 100, 102, 105, 100]
    df = _session("2026-06-01", closes)
    engine = _engine(_settings())
    # Session not in the eligible set -> no entries at all.
    result = engine.run({"TEST": df}, daily={}, eligible_days={"TEST": set()})
    assert len(result.trades) == 0
    assert len(result.signals) == 0


def test_eligible_days_allows_listed_session():
    closes = [100, 100, 100, 100, 100, 102, 105, 100]
    df = _session("2026-06-01", closes)
    engine = _engine(_settings())
    day = pd.Timestamp("2026-06-01", tz="UTC")
    result = engine.run({"TEST": df}, daily={}, eligible_days={"TEST": {day}})
    assert len(result.trades) == 1


def test_gap_eligible_days_uses_open_vs_prior_close():
    idx = pd.date_range("2026-06-01", periods=4, freq="B", tz="UTC")
    daily = pd.DataFrame(
        {
            # Day 2 gaps up 3% (103 open vs 100 prior close); day 3 flat;
            # day 4 gaps down 2.9%.
            "open": [100.0, 103.0, 103.0, 100.0],
            "high": [101.0, 104.0, 104.0, 104.0],
            "low": [99.0, 102.0, 102.0, 99.0],
            "close": [100.0, 103.0, 103.0, 100.0],
            "volume": 1e6,
        },
        index=idx,
    )
    eligible = gap_eligible_days(daily, min_gap_pct=2.0)
    assert idx[1].normalize() in eligible
    assert idx[3].normalize() in eligible  # gap down counts too
    assert idx[0].normalize() not in eligible  # no prior close
    assert idx[2].normalize() not in eligible  # flat open


def test_daily_frame_is_truncated_to_prior_sessions():
    """The strategy must never see the current day's daily bar (look-ahead)."""
    seen: dict[str, dt.date] = {}

    class SpyStrategy(FireOnceStrategy):
        name = "spy"

        def evaluate(self, snapshot: MarketSnapshot) -> Signal:
            from daytrader.data.providers.base import Timeframe

            ddf = snapshot.frame(Timeframe.DAY)
            if ddf is not None and len(ddf):
                seen["max_daily"] = max(
                    seen.get("max_daily", dt.date.min), ddf.index.max().date()
                )
            return self._hold(snapshot.symbol, "spy only")

    intraday = _session("2026-06-02", [100] * 6)
    daily_idx = pd.date_range("2026-05-25", "2026-06-02", freq="B", tz="UTC")
    daily = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6},
        index=daily_idx,
    )
    engine = BacktestEngine(
        _settings(), [SpyStrategy(0)], spread_bps=0.0, atr_impact_coeff=0.0, warmup_bars=0
    )
    engine.run({"TEST": intraday}, daily={"TEST": daily})
    assert seen["max_daily"] < dt.date(2026, 6, 2)
