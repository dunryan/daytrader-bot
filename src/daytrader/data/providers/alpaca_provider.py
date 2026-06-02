"""Alpaca-backed market-data provider.

Wraps ``alpaca-py``'s ``StockHistoricalDataClient``. The ``alpaca`` SDK is
imported lazily so the package still imports (and unit tests that mock data
still run) in environments without credentials or the SDK installed.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.data.providers.base import CANONICAL_COLUMNS, MarketDataProvider, Timeframe
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


class AlpacaProvider(MarketDataProvider):
    """Historical/recent bars and quotes from Alpaca Market Data."""

    def __init__(self, api_key: str | None, secret_key: str | None, feed: str = "iex") -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.feed = feed
        self._client = None
        self._timeframe_cls = None

    # ── lifecycle ──────────────────────────────────────────────
    def is_available(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def _ensure_client(self):
        """Lazily construct the Alpaca client; raise a clear error if unusable."""
        if self._client is not None:
            return self._client
        if not self.is_available():
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "in your .env to use the Alpaca data provider."
            )
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "alpaca-py is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self._timeframe_cls = (TimeFrame, TimeFrameUnit)
        return self._client

    def _to_alpaca_timeframe(self, timeframe: Timeframe):
        TimeFrame, TimeFrameUnit = self._timeframe_cls  # type: ignore[misc]
        mapping = {
            Timeframe.MIN_1: TimeFrame(1, TimeFrameUnit.Minute),
            Timeframe.MIN_5: TimeFrame(5, TimeFrameUnit.Minute),
            Timeframe.MIN_15: TimeFrame(15, TimeFrameUnit.Minute),
            Timeframe.DAY: TimeFrame(1, TimeFrameUnit.Day),
        }
        return mapping[timeframe]

    # ── data access ────────────────────────────────────────────
    def get_bars(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        start: dt.datetime,
        end: dt.datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        client = self._ensure_client()
        from alpaca.data.requests import StockBarsRequest

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self._to_alpaca_timeframe(timeframe),
            start=start,
            end=end,
            limit=limit,
            feed=self.feed,
        )
        try:
            bars = client.get_stock_bars(request)
        except Exception:  # noqa: BLE001
            logger.exception("Alpaca get_stock_bars failed for %d symbols", len(symbols))
            return {}

        df = bars.df
        if df is None or df.empty:
            return {}
        return self._split_by_symbol(df)

    @staticmethod
    def _split_by_symbol(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split Alpaca's MultiIndex (symbol, timestamp) frame per symbol."""
        out: dict[str, pd.DataFrame] = {}
        # Alpaca returns a MultiIndex of (symbol, timestamp).
        if isinstance(df.index, pd.MultiIndex):
            for symbol in df.index.get_level_values(0).unique():
                sub = df.xs(symbol, level=0).copy()
                out[str(symbol)] = AlpacaProvider._normalize(sub)
        else:
            out["__single__"] = AlpacaProvider._normalize(df)
        return out

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={c: c.lower() for c in df.columns})
        keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
        df = df[keep]
        df.index.name = "timestamp"
        return df

    def get_latest_quote(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        client = self._ensure_client()
        from alpaca.data.requests import StockLatestTradeRequest

        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=self.feed)
            trades = client.get_stock_latest_trade(req)
            return {sym: float(trade.price) for sym, trade in trades.items()}
        except Exception:  # noqa: BLE001
            logger.exception("Alpaca latest-trade fetch failed")
            return {}
