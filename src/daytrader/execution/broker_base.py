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
    """Minimal broker surface: a cash ledger plus market fills."""

    is_simulated: bool = True

    @abstractmethod
    def get_cash(self) -> float:
        """Available cash balance."""

    @abstractmethod
    def fill_market(self, symbol: str, side: Side, qty: float, ref_price: float) -> FillResult:
        """Execute a market order at (around) ``ref_price`` and adjust cash."""
