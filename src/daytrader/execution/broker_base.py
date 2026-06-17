"""Broker interface and shared execution value objects (Module 4).

The ``SIMULATION_MODE`` toggle selects which concrete broker the engine uses:
``VirtualBroker`` (no real capital) or ``LiveBroker`` (Alpaca). Both implement
the same tiny surface so the :class:`ExecutionEngine` is identical either way.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass
class FillResult:
    """Result of a (simulated or real) market fill."""

    symbol: str
    side: Side
    qty: float
    fill_price: float
    slippage: float = 0.0
    commission: float = 0.0
    broker_order_id: str | None = None
    timestamp: dt.datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = dt.datetime.now(dt.timezone.utc)

    @property
    def notional(self) -> float:
        return self.qty * self.fill_price


class Broker(ABC):
    """Broker surface: cash ledger, fills, and protective-order management.

    The bracket/stop methods have safe defaults so the :class:`VirtualBroker`
    stays minimal (software stops are authoritative in simulation); the live
    broker overrides them with real resting orders at the venue.
    """

    is_simulated: bool = True

    @abstractmethod
    def get_cash(self) -> float:
        """Available cash balance."""

    @abstractmethod
    def fill_market(self, symbol: str, side: Side, qty: float, ref_price: float) -> FillResult:
        """Execute a market order at (around) ``ref_price`` and adjust cash."""

    # ── protective-order surface (overridden by LiveBroker) ────
    def get_equity(self) -> float:
        """Total account equity. Defaults to cash for brokers that don't track positions."""
        return self.get_cash()

    def submit_bracket(
        self,
        symbol: str,
        side: Side,
        qty: float,
        ref_price: float,
        stop: float,
        target: float | None = None,
    ) -> FillResult | None:
        """Entry plus broker-side protective stop (and optional take-profit).

        Returns ``None`` when the entry could not be filled (e.g. a marketable
        limit expired unfilled). Default: plain market fill, no resting orders
        — software stop management remains authoritative.
        """
        return self.fill_market(symbol, side, qty, ref_price)

    def replace_stop(self, symbol: str, new_stop: float) -> bool:
        """Move the resting stop order for ``symbol``. Returns success."""
        return True

    def cancel_symbol_orders(self, symbol: str) -> int:
        """Cancel all open orders for ``symbol``; returns the count cancelled."""
        return 0

    def list_positions(self) -> dict[str, float] | None:
        """Open positions at the broker as ``symbol -> signed qty``.

        ``None`` means the broker cannot report positions (simulation), in
        which case reconciliation is skipped.
        """
        return None
