"""Pre-market screener: gap / relative-volume / liquidity filtering.

The metric math and filter predicate are pure functions (no I/O) so they are
trivially unit-testable. :class:`Screener` wires them to a market-data provider.

Premarket RVOL uses time-of-day normalization: cumulative volume through the
research cutoff (default 07:00 ET) vs the trailing average of prior sessions'
premarket volume at the same cutoff — matching backtest ``--premarket-rvol-mode tod``.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.config.settings import ResearchFilters
from daytrader.data.providers.base import MarketDataProvider, Timeframe
from daytrader.research.models import TickerMetrics
from daytrader.research.premarket_rvol import parse_cutoff_time, tod_premarket_rvol
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


def compute_metrics(symbol: str, daily: pd.DataFrame, lookback: int = 20) -> TickerMetrics | None:
    """Compute pre-market metrics from a symbol's daily OHLCV frame."""
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


def apply_tod_premarket_rvol(
    metrics: TickerMetrics,
    daily: pd.DataFrame,
    intraday_5m: pd.DataFrame | None,
    cutoff: dt.time,
    lookback: int = 20,
) -> TickerMetrics:
    """Replace ``relative_volume`` with TOD-normalized premarket RVOL when 5m data exists."""
    if intraday_5m is None or intraday_5m.empty:
        return metrics
    session_day = pd.Index(daily.sort_index().index).normalize()[-1]
    tod = tod_premarket_rvol(intraday_5m, session_day, cutoff, lookback)
    if tod is not None:
        metrics.relative_volume = tod
    return metrics


def passes_filters(metrics: TickerMetrics, filters: ResearchFilters) -> tuple[bool, list[str]]:
    """Apply liquidity / price / gap / RVOL filters."""
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
        return False, [
            f"TOD RVOL {metrics.relative_volume:.2f}x < {filters.min_relative_volume}x"
        ]

    direction = "up" if metrics.gap_percent >= 0 else "down"
    reasons = [
        f"gap-{direction} {metrics.gap_percent:+.2f}%",
        f"TOD RVOL {metrics.relative_volume:.2f}x",
        f"avg vol {metrics.avg_daily_volume / 1e6:.1f}M",
    ]
    return True, reasons


class Screener:
    """Fetches daily + 5m bars and screens a universe for tradeable candidates."""

    def __init__(
        self,
        provider: MarketDataProvider,
        filters: ResearchFilters,
        premarket_cutoff: dt.time | str = dt.time(7, 0),
        lookback: int = 20,
    ) -> None:
        self.provider = provider
        self.filters = filters
        self.premarket_cutoff = (
            parse_cutoff_time(premarket_cutoff) if isinstance(premarket_cutoff, str) else premarket_cutoff
        )
        self.lookback = lookback

    def screen(
        self,
        symbols: list[str],
        as_of: dt.datetime | None = None,
        history_days: int = 40,
        batch_size: int = 100,
    ) -> tuple[list[TickerMetrics], list[TickerMetrics]]:
        """Screen ``symbols``; returns ``(passed, all_metrics)``."""
        as_of = as_of or dt.datetime.now(dt.timezone.utc)
        start = as_of - dt.timedelta(days=history_days)

        all_metrics: list[TickerMetrics] = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            daily_bars = self.provider.get_bars(batch, Timeframe.DAY, start=start, end=as_of)
            intraday_bars = self.provider.get_bars(batch, Timeframe.MIN_5, start=start, end=as_of)
            for sym in batch:
                df = daily_bars.get(sym)
                if df is None:
                    continue
                metrics = compute_metrics(sym, df)
                if metrics is None:
                    continue
                metrics = apply_tod_premarket_rvol(
                    metrics, df, intraday_bars.get(sym), self.premarket_cutoff, self.lookback
                )
                all_metrics.append(metrics)

        passed: list[TickerMetrics] = []
        for m in all_metrics:
            ok, reasons = passes_filters(m, self.filters)
            if ok:
                passed.append(m)
                logger.info("SCREENER PASS %s: %s", m.symbol, "; ".join(reasons))
            else:
                logger.info("SCREENER REJECT %s: %s", m.symbol, reasons[0])

        logger.info(
            "Screened %d symbols -> %d passed filters (TOD RVOL by %s ET)",
            len(all_metrics), len(passed),
            self.premarket_cutoff.strftime("%H:%M"),
        )
        return passed, all_metrics
