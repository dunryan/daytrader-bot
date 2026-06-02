"""Market-data provider interface.

Strategy and research code depend only on this ABC, never on a vendor SDK.
Bars are returned as pandas DataFrames with a canonical column set:
``open, high, low, close, volume`` indexed by timezone-aware timestamp.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume"]


class Timeframe(str, Enum):
    """Supported bar timeframes."""

    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    DAY = "1d"


class MarketDataProvider(ABC):
    """Abstract source of historical/recent OHLCV bars."""

    @abstractmethod
    def get_bars(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        start: dt.datetime,
        end: dt.datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for ``symbols``.

        Returns a mapping ``symbol -> DataFrame`` with :data:`CANONICAL_COLUMNS`.
        Symbols with no data are omitted from the mapping rather than raising.
        """

    @abstractmethod
    def get_latest_quote(self, symbols: list[str]) -> dict[str, float]:
        """Return the latest trade/quote price per symbol (best-effort)."""

    def is_available(self) -> bool:
        """Whether the provider is configured well enough to be used."""
        return True
