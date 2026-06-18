"""Backtest helpers mirroring the live pre-market screener (07:00 ET).

The live screener compares cumulative pre-market volume to the trailing
average daily volume (``research.filters.min_relative_volume``). This module
approximates that from extended-hours intraday bars without look-ahead.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.data.session import MARKET_TZ, RTH_OPEN

PREMARKET_OPEN = dt.time(4, 0)


def parse_cutoff_time(value: str) -> dt.time:
    """Parse ``HH:MM`` into a time (matches ``schedule.premarket_research``)."""
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    return dt.time(hour, minute)


def premarket_volume_by_day(
    intraday_raw: pd.DataFrame,
    cutoff: dt.time = dt.time(7, 0),
) -> dict[pd.Timestamp, float]:
    """Sum bar volume from extended hours through ``cutoff`` on each session day.

    Uses bar *start* timestamps in ``[04:00, cutoff)`` ET, before RTH open.
    """
    if intraday_raw is None or intraday_raw.empty:
        return {}

    idx = pd.DatetimeIndex(intraday_raw.index)
    local = idx.tz_convert(MARKET_TZ) if idx.tz is not None else idx
    out: dict[pd.Timestamp, float] = {}

    for day_norm in pd.Index(local.normalize()).unique():
        day_mask = local.normalize() == day_norm
        times = local[day_mask].time
        vols = intraday_raw.loc[day_mask, "volume"]
        mask = (times >= PREMARKET_OPEN) & (times < cutoff) & (times < RTH_OPEN)
        # Key by UTC midnight of the ET session date (matches gap_eligible_days / engine).
        et_date = local[day_mask][0].date()
        session_key = pd.Timestamp(et_date, tz="UTC")
        out[session_key] = float(vols[mask].sum())

    return out


def premarket_rvol_eligible_days(
    daily: pd.DataFrame,
    intraday_raw: pd.DataFrame | None,
    min_rvol: float,
    cutoff: dt.time = dt.time(7, 0),
    lookback: int = 20,
) -> set[pd.Timestamp]:
    """Session dates whose premarket RVOL meets the screener threshold.

    RVOL = cumulative premarket volume through ``cutoff`` / mean(prior ``lookback``
    full-session daily volumes). Same baseline as ``research.screener.compute_metrics``.
    """
    if daily is None or len(daily) < 2 or min_rvol <= 0:
        return set()

    pm_vols = premarket_volume_by_day(intraday_raw, cutoff) if intraday_raw is not None else {}
    d = daily.sort_index()
    days = pd.Index(d.index).normalize()
    eligible: set[pd.Timestamp] = set()

    for i in range(1, len(d)):
        day = days[i]
        prior = d.iloc[max(0, i - lookback) : i]
        avg_vol = float(prior["volume"].mean()) if not prior.empty else 0.0
        if avg_vol <= 0:
            continue
        pm_vol = pm_vols.get(day, 0.0)
        if pm_vol / avg_vol >= min_rvol:
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
