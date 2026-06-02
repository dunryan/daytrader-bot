"""End-to-end tests for the ExecutionEngine (sim mode, real SQLite)."""

from __future__ import annotations

import datetime as dt

from daytrader.config.settings import Settings
from daytrader.execution.execution_engine import ExecutionEngine
from daytrader.execution.risk_manager import RiskManager
from daytrader.execution.virtual_broker import VirtualBroker
from daytrader.strategy.base import Direction, Signal


def _engine(db, settings: Settings | None = None) -> ExecutionEngine:
    settings = settings or Settings()
    broker = VirtualBroker(
        starting_cash=settings.risk.starting_equity,
        slippage_pct=settings.risk.slippage_pct,
        commission_per_trade=settings.risk.commission_per_trade,
    )
    eng = ExecutionEngine(settings, broker, RiskManager(settings.risk), db)
    eng.start_day(dt.date(2026, 6, 1))
    return eng


def _buy_signal(symbol="AAPL", price=100.0, stop=98.0) -> Signal:
    return Signal(
        symbol=symbol, strategy="vwap_pullback", direction=Direction.BUY,
        price=price, confidence=0.8, rationale="test", timeframe="5m",
        indicators={"atr": 1.0}, stop_hint=stop, target_hint=None,
    )


def test_open_position_sizes_and_persists(db):
    eng = _engine(db)
    opened = eng.process_signals([_buy_signal()], prices={"AAPL": 100.0})
    assert len(opened) == 1
    pos = opened[0]
    # 1% risk = $1000, but position cap 20% of 100k / $100 = 200 shares.
    assert pos.qty == 200
    assert "AAPL" in eng.open_positions
    # Persisted as an OPEN row.
    assert len(eng.positions.get_open()) == 1


def test_take_profit_exit(db):
    eng = _engine(db)
    eng.process_signals([_buy_signal(price=100.0, stop=98.0)], prices={"AAPL": 100.0})
    # tp = entry + 2 * (entry - stop) = 100 + 4 = 104.
    closed = eng.manage_positions(prices={"AAPL": 105.0})
    assert closed == ["AAPL"]
    assert "AAPL" not in eng.open_positions
    metric = eng.metrics.get("2026-06-01")
    assert metric.total_trades == 1
    assert metric.winning_trades == 1
    assert metric.realized_pnl > 0


def test_stop_loss_exit(db):
    eng = _engine(db)
    eng.process_signals([_buy_signal(price=100.0, stop=98.0)], prices={"AAPL": 100.0})
    closed = eng.manage_positions(prices={"AAPL": 97.0})  # below stop
    assert closed == ["AAPL"]
    metric = eng.metrics.get("2026-06-01")
    assert metric.losing_trades == 1
    assert metric.realized_pnl < 0


def test_one_position_per_symbol(db):
    eng = _engine(db)
    eng.process_signals([_buy_signal()], prices={"AAPL": 100.0})
    eng.process_signals([_buy_signal()], prices={"AAPL": 100.0})  # duplicate ignored
    assert len(eng.open_positions) == 1


def test_max_open_positions_enforced(db):
    settings = Settings()
    settings.risk.max_open_positions = 1
    eng = _engine(db, settings)
    sigs = [_buy_signal("AAPL"), _buy_signal("MSFT")]
    eng.process_signals(sigs, prices={"AAPL": 100.0, "MSFT": 100.0})
    assert len(eng.open_positions) == 1


def test_kill_switch_flattens_and_halts(db):
    eng = _engine(db)
    eng.process_signals([_buy_signal(price=100.0, stop=98.0)], prices={"AAPL": 100.0})
    # Crash the price so account drawdown exceeds 3%.
    halted = eng.check_kill_switch(prices={"AAPL": 50.0})
    assert halted is True
    assert eng.halted is True
    assert len(eng.open_positions) == 0
    metric = eng.metrics.get("2026-06-01")
    assert metric.trading_halted is True
    # Further signals are ignored while halted.
    opened = eng.process_signals([_buy_signal("MSFT")], prices={"MSFT": 100.0})
    assert opened == []


def test_restart_rehydrates_open_positions(db):
    eng1 = _engine(db)
    eng1.process_signals([_buy_signal()], prices={"AAPL": 100.0})
    cash_before = eng1.broker.get_cash()

    # Simulate a process restart: brand new engine, same DB.
    eng2 = _engine(db)
    assert "AAPL" in eng2.open_positions
    assert eng2.open_positions["AAPL"].qty == 200
    # Virtual cash reconstructed to match pre-restart state.
    assert round(eng2.broker.get_cash(), 2) == round(cash_before, 2)


def test_halt_persists_across_restart(db):
    eng1 = _engine(db)
    eng1.process_signals([_buy_signal(price=100.0, stop=98.0)], prices={"AAPL": 100.0})
    eng1.check_kill_switch(prices={"AAPL": 50.0})

    eng2 = _engine(db)
    assert eng2.halted is True  # latched halt survives reboot


def test_end_of_day_flatten(db):
    eng = _engine(db)
    eng.process_signals([_buy_signal()], prices={"AAPL": 100.0})
    eng.end_of_day_flatten(prices={"AAPL": 101.0})
    assert len(eng.open_positions) == 0
    closed = eng.positions.get_closed_for_day("2026-06-01")
    assert closed[0].exit_reason == "EOD"
