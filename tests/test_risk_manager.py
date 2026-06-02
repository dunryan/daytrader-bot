"""Tests for RiskManager sizing, stops, take-profit, and guards."""

from __future__ import annotations

from daytrader.config.settings import RiskConfig
from daytrader.execution.broker_base import Side
from daytrader.execution.risk_manager import RiskManager


def _rm(**overrides) -> RiskManager:
    cfg = RiskConfig(**overrides)
    return RiskManager(cfg)


def test_position_size_risk_limited():
    rm = _rm(max_risk_per_trade_pct=1.0, max_position_size_pct=100.0)
    # 1% of 100k = $1000 risk; risk/share = $2 -> 500 shares.
    res = rm.position_size(equity=100_000, entry=100.0, stop=98.0)
    assert res.qty == 500
    assert res.capped_by == "risk"
    assert res.risk_amount == 1000.0


def test_position_size_capped_by_position_limit():
    rm = _rm(max_risk_per_trade_pct=1.0, max_position_size_pct=20.0)
    # Risk would allow 500, but 20% of 100k / $100 = 200 shares cap.
    res = rm.position_size(equity=100_000, entry=100.0, stop=98.0)
    assert res.qty == 200
    assert res.capped_by == "position_cap"


def test_position_size_capped_by_cash():
    rm = _rm(max_risk_per_trade_pct=5.0, max_position_size_pct=100.0)
    res = rm.position_size(equity=100_000, entry=100.0, stop=99.0, available_cash=5_000)
    assert res.qty == 50  # 5000 / 100
    assert res.capped_by == "cash"


def test_position_size_zero_when_no_risk_distance():
    rm = _rm()
    res = rm.position_size(equity=100_000, entry=100.0, stop=100.0)
    assert res.qty == 0


def test_resolve_stop_prefers_valid_hint():
    rm = _rm()
    assert rm.resolve_stop(Side.BUY, entry=100, atr=1.0, stop_hint=97.5) == 97.5
    # Invalid hint (above entry for a long) is ignored -> ATR fallback.
    stop = rm.resolve_stop(Side.BUY, entry=100, atr=1.0, stop_hint=101)
    assert stop == 100 - 2 * 1.0


def test_resolve_stop_atr_long_and_short():
    rm = _rm()
    rm.config.stop_loss.atr_multiplier = 2.0
    assert rm.resolve_stop(Side.BUY, 100, 1.5) == 100 - 3.0
    assert rm.resolve_stop(Side.SELL, 100, 1.5) == 100 + 3.0


def test_resolve_stop_structural():
    rm = _rm()
    rm.config.stop_loss.method = "structural"
    stop = rm.resolve_stop(Side.BUY, entry=100, atr=1.0, support_levels=[95, 97, 99, 101])
    assert stop == 99  # nearest support below entry


def test_take_profit_risk_reward():
    rm = _rm()
    rm.config.take_profit.risk_reward_ratio = 2.0
    assert rm.take_profit(Side.BUY, entry=100, stop=98) == 104  # 2 * $2 risk
    assert rm.take_profit(Side.SELL, entry=100, stop=102) == 96


def test_trailing_stop_only_ratchets_up():
    rm = _rm()
    rm.config.take_profit.trailing_atr_multiplier = 1.0
    # Long: peak 110, atr 1 -> candidate 109, above current 98 -> trails up.
    assert rm.update_trailing_stop(Side.BUY, current_stop=98, peak_price=110, atr=1.0) == 109
    # Never loosens: a lower candidate keeps the existing stop.
    assert rm.update_trailing_stop(Side.BUY, current_stop=109, peak_price=105, atr=1.0) == 109


def test_drawdown_breach():
    rm = _rm(max_daily_drawdown_pct=3.0)
    assert rm.is_drawdown_breached(100_000, 97_000) is True
    assert rm.is_drawdown_breached(100_000, 98_000) is False


def test_can_open_new_respects_max():
    rm = _rm(max_open_positions=2)
    assert rm.can_open_new(1) is True
    assert rm.can_open_new(2) is False
