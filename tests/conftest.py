"""Shared test fixtures and a fake market-data provider (no network)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from daytrader.config.settings import Settings
from daytrader.data.providers.base import MarketDataProvider, Timeframe
from daytrader.persistence.database import Database


class FakeProvider(MarketDataProvider):
    """In-memory provider returning canned daily frames per symbol."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def get_bars(self, symbols, timeframe, start, end=None, limit=None):
        return {s: self.frames[s] for s in symbols if s in self.frames}

    def get_latest_quote(self, symbols):
        return {s: float(self.frames[s]["close"].iloc[-1]) for s in symbols if s in self.frames}

    def is_available(self) -> bool:
        return True


def make_daily_frame(
    days: int,
    base_volume: float,
    prev_close: float,
    day_open: float,
    today_volume: float,
    today_close: float | None = None,
) -> pd.DataFrame:
    """Build a daily OHLCV frame: ``days`` historical bars + a 'today' bar."""
    idx = pd.date_range(end=dt.date.today(), periods=days + 1, freq="D", tz="UTC")
    closes = [prev_close] * days + [today_close if today_close is not None else day_open]
    opens = [prev_close] * days + [day_open]
    volumes = [base_volume] * days + [today_volume]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) * 1.01 for o, c in zip(opens, closes)],
            "low": [min(o, c) * 0.99 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


@pytest.fixture
def settings() -> Settings:
    """Default-validated settings (no YAML/.env needed)."""
    return Settings()


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(f"sqlite:///{tmp_path / 'test.db'}")
