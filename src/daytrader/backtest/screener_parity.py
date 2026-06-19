"""Backtest helpers mirroring the live pre-market screener (07:00 ET)."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.research.premarket_rvol import (
    parse_cutoff_time,
    premarket_volume_by_day,
    tod_premarket_rvol,
)

__all__ = [
    "parse_cutoff_time",
    "premarket_volume_by_day",
    "premarket_rvol_eligible_days",
    "intersect_eligible_days",
]


def premarket_rvol_eligible_days(
    daily: pd.DataFrame,
    intraday_raw: pd.DataFrame | None,
    min_rvol: float,
    cutoff: dt.time = dt.time(7, 0),
    lookback: int = 20,
    mode: str = "tod",
) -> set[pd.Timestamp]:
    """Session dates whose premarket RVOL meets the screener threshold."""
    if daily is None or len(daily) < 2 or min_rvol <= 0:
        return set()

    pm_by_day = premarket_volume_by_day(intraday_raw, cutoff) if intraday_raw is not None else {}
    d = daily.sort_index()
    days = pd.Index(d.index).normalize()
    eligible: set[pd.Timestamp] = set()

    for i in range(1, len(d)):
        day = days[i]
        if mode == "tod":
            rvol = tod_premarket_rvol(intraday_raw, day, cutoff, lookback) if intraday_raw is not None else None
        else:
            pm_vol = pm_by_day.get(day, 0.0)
            if pm_vol <= 0:
                continue
            prior = d.iloc[max(0, i - lookback) : i]
            avg = float(prior["volume"].mean()) if not prior.empty else 0.0
            rvol = pm_vol / avg if avg > 0 else None

        if rvol is not None and rvol >= min_rvol:
            eligible.add(day)

    return eligible


def intersect_eligible_days(
    base: dict[str, set[pd.Timestamp]] | None,
    extra: dict[str, set[pd.Timestamp]],
    symbols: list[str],
) -> dict[str, set[pd.Timestamp]]:
    """Intersect per-symbol eligible-day sets (gap-days ∩ premarket RVOL, etc.)."""
    if base is None:
        return {s: set(extra.get(s, set())) for s in symbols}
    return {s: set(base.get(s, set())) & set(extra.get(s, set())) for s in symbols}
