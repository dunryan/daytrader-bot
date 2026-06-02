"""Module 2 orchestrator: fetch multi-timeframe bars and enrich them.

Produces a :class:`MarketSnapshot` per symbol — the object Module 3 consumes to
evaluate strategies. The engine itself is thin; all the math lives in
``indicators.py`` so it stays testable.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from daytrader.config.settings import IndicatorsConfig
from daytrader.data import indicators as ind
from daytrader.data.providers.base import MarketDataProvider, Timeframe
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Default history to request per timeframe so EMA-200 etc. are well-formed.
# Expressed in calendar days passed to the provider's start window.
_LOOKBACK_DAYS: dict[Timeframe, int] = {
    Timeframe.MIN_1: 3,
    Timeframe.MIN_5: 10,
    Timeframe.MIN_15: 30,
    Timeframe.DAY: 400,
}


@dataclass
class MarketSnapshot:
    """Enriched, multi-timeframe view of one symbol at a point in time."""

    symbol: str
    as_of: dt.datetime
    frames: dict[Timeframe, pd.DataFrame] = field(default_factory=dict)
    pivots: dict[str, float] = field(default_factory=dict)
    support: list[float] = field(default_factory=list)
    resistance: list[float] = field(default_factory=list)

    def frame(self, timeframe: Timeframe) -> pd.DataFrame | None:
        return self.frames.get(timeframe)

    def latest(self, timeframe: Timeframe) -> pd.Series | None:
        """Most recent fully-populated row for a timeframe."""
        df = self.frames.get(timeframe)
        if df is None or df.empty:
            return None
        return df.iloc[-1]

    def latest_price(self) -> float | None:
        for tf in (Timeframe.MIN_1, Timeframe.MIN_5, Timeframe.MIN_15, Timeframe.DAY):
            row = self.latest(tf)
            if row is not None:
                return float(row["close"])
        return None

    def indicator(self, timeframe: Timeframe, name: str) -> float | None:
        row = self.latest(timeframe)
        if row is None or name not in row or pd.isna(row[name]):
            return None
        return float(row[name])


class DataEngine:
    """Fetches and enriches market data behind a provider."""

    def __init__(self, provider: MarketDataProvider, config: IndicatorsConfig) -> None:
        self.provider = provider
        self.config = config

    # ── low-level enrichment ───────────────────────────────────
    def enrich(self, df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """Append the configured indicator set; VWAP only for intraday frames."""
        return ind.add_indicators(
            df,
            ema_periods=self.config.ema_periods,
            rsi_period=self.config.rsi_period,
            atr_period=self.config.atr_period,
            macd_params=(self.config.macd.fast, self.config.macd.slow, self.config.macd.signal),
            include_vwap=(timeframe != Timeframe.DAY),
        )

    def fetch_enriched(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        as_of: dt.datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch and enrich bars for many symbols on one timeframe."""
        as_of = as_of or dt.datetime.now(dt.timezone.utc)
        start = as_of - dt.timedelta(days=_LOOKBACK_DAYS.get(timeframe, 30))
        raw = self.provider.get_bars(symbols, timeframe, start=start, end=as_of)
        return {sym: self.enrich(df, timeframe) for sym, df in raw.items()}

    # ── snapshot assembly ──────────────────────────────────────
    def build_snapshot(
        self,
        symbol: str,
        timeframes: list[Timeframe],
        as_of: dt.datetime | None = None,
        swing_window: int = 5,
    ) -> MarketSnapshot:
        """Assemble a multi-timeframe snapshot for a single symbol."""
        as_of = as_of or dt.datetime.now(dt.timezone.utc)
        snap = MarketSnapshot(symbol=symbol, as_of=as_of)

        for tf in timeframes:
            start = as_of - dt.timedelta(days=_LOOKBACK_DAYS.get(tf, 30))
            raw = self.provider.get_bars([symbol], tf, start=start, end=as_of)
            df = raw.get(symbol)
            if df is None or df.empty:
                continue
            snap.frames[tf] = self.enrich(df, tf)

        self._attach_levels(snap, swing_window)
        return snap

    def build_snapshots(
        self,
        symbols: list[str],
        timeframes: list[Timeframe],
        as_of: dt.datetime | None = None,
        swing_window: int = 5,
    ) -> dict[str, MarketSnapshot]:
        return {
            sym: self.build_snapshot(sym, timeframes, as_of, swing_window) for sym in symbols
        }

    def _attach_levels(self, snap: MarketSnapshot, swing_window: int) -> None:
        """Derive pivot points (from daily) and swing S/R (from intraday)."""
        daily = snap.frames.get(Timeframe.DAY)
        if daily is not None and len(daily) >= 2:
            prev = daily.iloc[-2]
            snap.pivots = ind.pivot_points(
                float(prev["high"]), float(prev["low"]), float(prev["close"])
            )

        # Prefer the finest intraday frame available for swing detection.
        for tf in (Timeframe.MIN_5, Timeframe.MIN_15, Timeframe.MIN_1, Timeframe.DAY):
            df = snap.frames.get(tf)
            if df is not None and len(df) > 2 * swing_window + 1:
                levels = ind.swing_levels(df, window=swing_window)
                snap.support = levels["support"]
                snap.resistance = levels["resistance"]
                break
