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
    mode: str = "tod",
) -> set[pd.Timestamp]:
    """Session dates whose premarket RVOL meets the screener threshold.

    ``mode='tod'`` (default): premarket volume through ``cutoff`` divided by the
    mean premarket volume at the same cutoff over prior ``lookback`` sessions.
    This is the scale that matches actionable gap-day screens (~1.5x).

    ``mode='screener'``: ``premarket_vol / mean(prior full daily volumes)`` —
    the literal ``research.screener.compute_metrics`` ratio at 07:00. With
    ``min_relative_volume`` 1.5 this almost never passes before the open
    (premarket is a small fraction of a full session).
    """
    if daily is None or len(daily) < 2 or min_rvol <= 0:
        return set()

    pm_by_day = premarket_volume_by_day(intraday_raw, cutoff) if intraday_raw is not None else {}
    d = daily.sort_index()
    days = pd.Index(d.index).normalize()
    eligible: set[pd.Timestamp] = set()

    for i in range(1, len(d)):
        day = days[i]
        pm_vol = pm_by_day.get(day, 0.0)
        if pm_vol <= 0:
            continue

        prior = d.iloc[max(0, i - lookback) : i]
        if mode == "screener":
            avg = float(prior["volume"].mean()) if not prior.empty else 0.0
            rvol = pm_vol / avg if avg > 0 else 0.0
        else:
            prior_days = days[max(0, i - lookback) : i]
            samples = [pm_by_day.get(pd.Timestamp(d.date(), tz="UTC"), 0.0) for d in prior_days]
            samples = [v for v in samples if v > 0]
            avg_pm = sum(samples) / len(samples) if samples else 0.0
            rvol = pm_vol / avg_pm if avg_pm > 0 else 0.0

        if rvol >= min_rvol:
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
