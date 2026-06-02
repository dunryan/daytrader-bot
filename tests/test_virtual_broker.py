"""Tests for the VirtualBroker slippage model and cash ledger."""

from __future__ import annotations

from daytrader.execution.broker_base import Side
from daytrader.execution.virtual_broker import VirtualBroker


def test_buy_applies_positive_slippage_and_debits_cash():
    b = VirtualBroker(starting_cash=100_000, slippage_pct=0.05)
    fill = b.fill_market("AAPL", Side.BUY, qty=100, ref_price=100.0)
    # Buy fills 0.05% worse (higher).
    assert fill.fill_price == 100.0 * 1.0005
    assert b.get_cash() == 100_000 - fill.fill_price * 100


def test_sell_applies_negative_slippage_and_credits_cash():
    b = VirtualBroker(starting_cash=0.0, slippage_pct=0.05)
    fill = b.fill_market("AAPL", Side.SELL, qty=100, ref_price=100.0)
    assert fill.fill_price == 100.0 * 0.9995
    assert b.get_cash() == fill.fill_price * 100


def test_commission_charged():
    b = VirtualBroker(starting_cash=10_000, slippage_pct=0.0, commission_per_trade=1.0)
    b.fill_market("AAPL", Side.BUY, qty=10, ref_price=100.0)
    assert b.get_cash() == 10_000 - 1000 - 1.0


def test_slippage_value_reported():
    b = VirtualBroker(starting_cash=100_000, slippage_pct=0.1)
    fill = b.fill_market("X", Side.BUY, qty=200, ref_price=50.0)
    expected_slip = (50.0 * 1.001 - 50.0) * 200
    assert round(fill.slippage, 6) == round(expected_slip, 6)
