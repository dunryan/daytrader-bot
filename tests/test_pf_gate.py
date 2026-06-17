"""Tests for rolling profit-factor deployment gate."""

from __future__ import annotations

from daytrader.config.settings import PfGateConfig
from daytrader.strategy.pf_gate import evaluate_pf_gate, profit_factor


def test_profit_factor_basic():
    assert profit_factor([100, -50, 80, -40]) == 180 / 90


def test_profit_factor_no_losses():
    assert profit_factor([10, 20]) == float("inf")


def test_gate_inert_when_off():
    blocked, details = evaluate_pf_gate([-100] * 20, PfGateConfig(mode="off"))
    assert blocked is False
    assert details == {}


def test_gate_inert_until_min_trades():
    cfg = PfGateConfig(mode="enforce", lookback_trades=20, min_trades=10, min_pf=0.9)
    blocked, details = evaluate_pf_gate([-100] * 5, cfg)
    assert blocked is False
    assert details["pf_gate_trades"] == 5


def test_gate_blocks_bad_streak():
    cfg = PfGateConfig(mode="enforce", lookback_trades=20, min_trades=10, min_pf=0.9)
    pnls = [-100.0] * 8 + [50.0] * 2  # PF = 100/800 = 0.125
    blocked, details = evaluate_pf_gate(pnls, cfg)
    assert blocked is True
    assert details["pf_gate_pf"] < 0.9


def test_gate_shadow_never_blocks():
    cfg = PfGateConfig(mode="shadow", lookback_trades=20, min_trades=5, min_pf=0.9)
    blocked, _ = evaluate_pf_gate([-100.0] * 10, cfg)
    assert blocked is False


def test_gate_passes_good_streak():
    cfg = PfGateConfig(mode="enforce", lookback_trades=20, min_trades=10, min_pf=0.9)
    pnls = [100.0] * 8 + [-50.0] * 2  # PF = 800/100 = 8.0
    blocked, details = evaluate_pf_gate(pnls, cfg)
    assert blocked is False
    assert details["pf_gate_pf"] >= 0.9
