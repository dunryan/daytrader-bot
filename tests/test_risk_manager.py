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


def test_trailing_inactive_before_one_r():
    rm = _rm()
    rm.config.take_profit.trailing_atr_multiplier = 1.0
    # Entry 100, initial risk $2 -> trail must NOT move until peak >= 102.
    stop = rm.update_trailing_stop(
        Side.BUY, current_stop=98, peak_price=101.5, atr=1.0, entry=100.0, initial_risk=2.0
    )
    assert stop == 98


def test_trailing_jumps_to_breakeven_then_trails():
    rm = _rm()
    rm.config.take_profit.trailing_atr_multiplier = 2.0
    # At exactly +1R with a wide trail, the stop moves to breakeven (entry).
    stop = rm.update_trailing_stop(
        Side.BUY, current_stop=98, peak_price=102.0, atr=1.5, entry=100.0, initial_risk=2.0
    )
    assert stop == 100.0  # max(98, entry 100, 102 - 3.0 = 99)
    # Further extension trails behind the peak.
    stop = rm.update_trailing_stop(
        Side.BUY, current_stop=stop, peak_price=106.0, atr=1.5, entry=100.0, initial_risk=2.0
    )
    assert stop == 103.0  # 106 - 2 * 1.5


def test_trailing_short_breakeven_and_trail():
    rm = _rm()
    rm.config.take_profit.trailing_atr_multiplier = 1.0
    stop = rm.update_trailing_stop(
        Side.SELL, current_stop=104, peak_price=96.0, atr=2.0, entry=100.0, initial_risk=4.0
    )
    assert stop == 98.0  # min(104, entry 100, 96 + 2 = 98)


def test_kelly_risk_pct_caps_and_floors():
    rm = _rm(max_risk_per_trade_pct=1.0)
    rm.config.kelly.fraction = 0.25
    # Strong edge: p=0.6, b=2 -> f* = 0.6 - 0.4/2 = 0.4; quarter-Kelly = 10%,
    # capped at the configured 1%.
    assert rm.kelly_risk_pct(win_rate=0.6, payoff_ratio=2.0) == 1.0
    # Negative edge -> 0 (trade suppressed).
    assert rm.kelly_risk_pct(win_rate=0.3, payoff_ratio=1.0) == 0.0
    # Degenerate payoff -> 0.
    assert rm.kelly_risk_pct(win_rate=0.9, payoff_ratio=0.0) == 0.0


def test_position_size_risk_pct_override_clamped():
    rm = _rm(max_risk_per_trade_pct=1.0, max_position_size_pct=100.0)
    # Override below the cap is honored: 0.5% of 100k = $500 / $2 = 250 shares.
    res = rm.position_size(equity=100_000, entry=100.0, stop=98.0, risk_pct=0.5)
    assert res.qty == 250
    # Override above the cap is clamped back to 1%.
    res = rm.position_size(equity=100_000, entry=100.0, stop=98.0, risk_pct=5.0)
    assert res.qty == 500
    # Zero override (Kelly says no edge) -> no position.
    res = rm.position_size(equity=100_000, entry=100.0, stop=98.0, risk_pct=0.0)
    assert res.qty == 0


def test_drawdown_breach():
    rm = _rm(max_daily_drawdown_pct=3.0)
    assert rm.is_drawdown_breached(100_000, 97_000) is True
    assert rm.is_drawdown_breached(100_000, 98_000) is False


def test_can_open_new_respects_max():
    rm = _rm(max_open_positions=2)
    assert rm.can_open_new(1) is True
    assert rm.can_open_new(2) is False
