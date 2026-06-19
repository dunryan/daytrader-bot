"""Premarket relative-volume helpers (live screener + backtest parity)."""

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
    """Sum bar volume from extended hours through ``cutoff`` on each session day."""
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
        et_date = local[day_mask][0].date()
        session_key = pd.Timestamp(et_date, tz="UTC")
        out[session_key] = float(vols[mask].sum())

    return out


def tod_premarket_rvol(
    intraday_5m: pd.DataFrame,
    session_day: pd.Timestamp,
    cutoff: dt.time = dt.time(7, 0),
    lookback: int = 20,
) -> float | None:
    """Premarket RVOL: today's PM vol / mean prior sessions' PM vol at ``cutoff``."""
    pm_by_day = premarket_volume_by_day(intraday_5m, cutoff)
    day = pd.Timestamp(session_day.date(), tz="UTC")
    today_vol = pm_by_day.get(day, 0.0)
    if today_vol <= 0:
        return None

    prior_vols = [v for k, v in sorted(pm_by_day.items()) if k < day and v > 0][-lookback:]
    if not prior_vols:
        return None
    avg_pm = sum(prior_vols) / len(prior_vols)
    if avg_pm <= 0:
        return None
    return today_vol / avg_pm
