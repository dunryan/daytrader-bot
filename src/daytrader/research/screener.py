"""Pre-market screener: gap / relative-volume / liquidity filtering.

The metric math and filter predicate are pure functions (no I/O) so they are
trivially unit-testable. :class:`Screener` wires them to a market-data provider.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.config.settings import ResearchFilters
from daytrader.data.providers.base import MarketDataProvider, Timeframe
from daytrader.research.models import TickerMetrics
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


def compute_metrics(symbol: str, daily: pd.DataFrame, lookback: int = 20) -> TickerMetrics | None:
    """Compute pre-market metrics from a symbol's daily OHLCV frame.

    The most recent row is treated as "today" (possibly a partial pre-market
    bar); the prior row is the previous session's close. Average daily volume
    is the mean of the ``lookback`` sessions *before* today, so today's partial
    volume never contaminates the baseline.

    Returns ``None`` when there is insufficient history to compute a gap.
    """
    if daily is None or len(daily) < 2:
        return None

    daily = daily.sort_index()
    today = daily.iloc[-1]
    prev = daily.iloc[-2]

    prev_close = float(prev["close"])
    if prev_close <= 0:
        return None

    day_open = float(today["open"])
    last_close = float(today["close"])
    current_volume = float(today["volume"])

    prior = daily.iloc[:-1].tail(lookback)
    avg_daily_volume = float(prior["volume"].mean()) if not prior.empty else 0.0

    gap_percent = (day_open - prev_close) / prev_close * 100.0
    relative_volume = (current_volume / avg_daily_volume) if avg_daily_volume > 0 else 0.0

    return TickerMetrics(
        symbol=symbol,
        last_close=last_close,
        prev_close=prev_close,
        day_open=day_open,
        gap_percent=gap_percent,
        current_volume=current_volume,
        avg_daily_volume=avg_daily_volume,
        relative_volume=relative_volume,
    )


def passes_filters(metrics: TickerMetrics, filters: ResearchFilters) -> tuple[bool, list[str]]:
    """Apply liquidity / price / gap / RVOL filters.

    Returns ``(passed, reasons)`` where ``reasons`` describes the qualifying
    characteristics when passed, or the failing criterion when not.
    """
    price = metrics.last_close
    if not (filters.min_price <= price <= filters.max_price):
        return False, [f"price {price:.2f} outside [{filters.min_price}, {filters.max_price}]"]

    if metrics.avg_daily_volume < filters.min_avg_daily_volume:
        return False, [
            f"avg vol {metrics.avg_daily_volume:,.0f} < {filters.min_avg_daily_volume:,}"
        ]

    if abs(metrics.gap_percent) < filters.min_gap_percent:
        return False, [f"gap {metrics.gap_percent:+.2f}% < {filters.min_gap_percent}%"]

    if metrics.relative_volume < filters.min_relative_volume:
        return False, [f"RVOL {metrics.relative_volume:.2f} < {filters.min_relative_volume}"]

    direction = "up" if metrics.gap_percent >= 0 else "down"
    reasons = [
        f"gap-{direction} {metrics.gap_percent:+.2f}%",
        f"RVOL {metrics.relative_volume:.2f}x",
        f"avg vol {metrics.avg_daily_volume / 1e6:.1f}M",
    ]
    return True, reasons


class Screener:
    """Fetches daily bars and screens a universe for tradeable candidates."""

    def __init__(self, provider: MarketDataProvider, filters: ResearchFilters) -> None:
        self.provider = provider
        self.filters = filters

    def screen(
        self,
        symbols: list[str],
        as_of: dt.datetime | None = None,
        history_days: int = 40,
        batch_size: int = 100,
    ) -> tuple[list[TickerMetrics], list[TickerMetrics]]:
        """Screen ``symbols``.

        Returns ``(passed, all_metrics)``: candidates that cleared the filters,
        and every symbol's metrics (for the scan audit log).
        """
        as_of = as_of or dt.datetime.now(dt.timezone.utc)
        start = as_of - dt.timedelta(days=history_days)

        all_metrics: list[TickerMetrics] = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            bars = self.provider.get_bars(batch, Timeframe.DAY, start=start, end=as_of)
            for sym in batch:
                df = bars.get(sym)
                if df is None:
                    continue
                metrics = compute_metrics(sym, df)
                if metrics is not None:
                    all_metrics.append(metrics)

        passed = [m for m in all_metrics if passes_filters(m, self.filters)[0]]
        logger.info("Screened %d symbols -> %d passed filters", len(all_metrics), len(passed))
        return passed, all_metrics
