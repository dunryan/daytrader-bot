"""Strategy interface and the Signal value object (Module 3).

A strategy is a pure decision function: given a :class:`MarketSnapshot` it
returns a :class:`Signal`. Strategies never place orders, size positions, or
touch the database — that separation keeps Module 4 (risk/execution) the only
place capital decisions happen, and makes strategies trivially testable.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """An explicit trading decision with full metadata for the audit trail."""

    symbol: str
    strategy: str
    direction: Direction
    price: float | None
    confidence: float
    rationale: str
    timeframe: str
    indicators: dict[str, Any] = field(default_factory=dict)
    # Optional reference levels the strategy observed (Module 4 may refine).
    stop_hint: float | None = None
    target_hint: float | None = None
    timestamp: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    @property
    def is_actionable(self) -> bool:
        return self.direction in (Direction.BUY, Direction.SELL)

    @classmethod
    def hold(cls, symbol: str, strategy: str, timeframe: str, reason: str) -> "Signal":
        return cls(
            symbol=symbol,
            strategy=strategy,
            direction=Direction.HOLD,
            price=None,
            confidence=0.0,
            rationale=reason,
            timeframe=timeframe,
        )


class Strategy(ABC):
    """Base class for all strategies."""

    #: Stable identifier persisted with each signal (must match config key).
    name: str = "base"
    #: Primary timeframe this strategy reasons on.
    timeframe: Timeframe = Timeframe.MIN_5

    @abstractmethod
    def evaluate(self, snapshot: MarketSnapshot) -> Signal:
        """Return a Signal (possibly HOLD) for the given snapshot."""

    # ── shared helpers ─────────────────────────────────────────
    def _hold(self, symbol: str, reason: str) -> Signal:
        return Signal.hold(symbol, self.name, self.timeframe.value, reason)
