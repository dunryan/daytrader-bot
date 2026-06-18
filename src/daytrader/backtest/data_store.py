"""Local parquet cache for historical bars.

Backtests hammer the same date ranges repeatedly; caching per symbol/timeframe
keeps iteration fast and API usage polite. Coverage checks are fuzzy by a few
days because bars only exist on trading days.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from daytrader.data.providers.base import MarketDataProvider, Timeframe
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)

_COVERAGE_TOLERANCE = dt.timedelta(days=5)


class BarStore:
    """Per-symbol, per-timeframe parquet cache in front of a data provider."""

    def __init__(self, cache_dir: str | Path = "data/backtest_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, timeframe: Timeframe, feed: str | None = None) -> Path:
        suffix = f"_{feed}" if feed else ""
        return self.cache_dir / f"{symbol}_{timeframe.value}{suffix}.parquet"

    def get(
        self,
        provider: MarketDataProvider,
        symbols: list[str],
        timeframe: Timeframe,
        start: dt.datetime,
        end: dt.datetime,
        feed: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return bars per symbol for [start, end], fetching gaps from the provider."""
        out: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []

        for symbol in symbols:
            path = self._path(symbol, timeframe, feed)
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    idx = pd.DatetimeIndex(df.index)
                    if (
                        not df.empty
                        and idx.min() <= pd.Timestamp(start) + _COVERAGE_TOLERANCE
                        and idx.max() >= pd.Timestamp(end) - _COVERAGE_TOLERANCE
                    ):
                        out[symbol] = df[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
                        continue
                except Exception:  # noqa: BLE001
                    logger.exception("Corrupt cache file %s; refetching", path)
            to_fetch.append(symbol)

        if to_fetch:
            logger.info(
                "Fetching %d symbol(s) of %s bars %s -> %s",
                len(to_fetch), timeframe.value, start.date(), end.date(),
            )
            fetched = provider.get_bars(to_fetch, timeframe, start=start, end=end)
            for symbol, df in fetched.items():
                if df is None or df.empty:
                    continue
                try:
                    df.to_parquet(self._path(symbol, timeframe, feed))
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to cache %s %s", symbol, timeframe.value)
                out[symbol] = df

        return out
