"""Small shared helpers for strategies (pure, testable)."""

from __future__ import annotations

import math

import pandas as pd

from daytrader.data.providers.base import Timeframe

_BAR_MINUTES: dict[Timeframe, int] = {
    Timeframe.MIN_1: 1,
    Timeframe.MIN_5: 5,
    Timeframe.MIN_15: 15,
    Timeframe.DAY: 390,  # one RTH session
}


def bar_minutes(timeframe: Timeframe) -> int:
    return _BAR_MINUTES.get(timeframe, 5)


def relative_volume(df: pd.DataFrame, window: int = 20) -> float:
    """Latest bar's volume divided by the mean of the prior ``window`` bars.

    Naive rolling baseline; prefer :func:`relative_volume_tod` for intraday
    frames since rolling windows cross session boundaries and are distorted
    by the U-shaped intraday volume profile.
    """
    if df is None or len(df) < 2:
        return 0.0
    prior = df["volume"].iloc[-(window + 1) : -1]
    avg = float(prior.mean()) if len(prior) else 0.0
    if avg <= 0:
        return 0.0
    return float(df["volume"].iloc[-1]) / avg


def relative_volume_tod(df: pd.DataFrame, sessions: int = 10) -> float:
    """Time-of-day-matched relative volume.

    Compares the latest bar's volume against the mean volume of the bar at
    the *same intra-session index* across the prior ``sessions`` sessions.
    This respects the U-shaped intraday volume profile: the 09:35 bar is
    benchmarked against prior 09:35 bars, not against overnight lulls.

    Falls back to the naive rolling measure when there is no prior-session
    history (e.g. unit tests or the first day of a fresh cache).
    """
    if df is None or len(df) < 2:
        return 0.0
    dates = pd.Index(df.index).normalize()
    last_date = dates[-1]
    bar_pos = int((dates == last_date).sum()) - 1  # 0-based index within session

    prior_dates = [d for d in pd.unique(dates) if d < last_date][-sessions:]
    samples: list[float] = []
    for d in prior_dates:
        day = df[dates == d]
        if len(day) > bar_pos:
            samples.append(float(day["volume"].iloc[bar_pos]))

    if not samples:
        return relative_volume(df)
    avg = sum(samples) / len(samples)
    if avg <= 0:
        return 0.0
    return float(df["volume"].iloc[-1]) / avg


def session_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows belonging to the most recent calendar day in the index."""
    if df is None or df.empty:
        return df
    last_date = pd.Index(df.index).normalize()[-1]
    mask = pd.Index(df.index).normalize() == last_date
    return df[mask]


def opening_range(df: pd.DataFrame, minutes: int, timeframe: Timeframe) -> tuple[float, float, int]:
    """Return ``(or_high, or_low, n_or_bars)`` for the current session.

    The opening range spans the first ``minutes`` of the session, converted to a
    whole number of bars given the frame's timeframe.
    """
    session = session_frame(df)
    n_bars = max(1, math.ceil(minutes / bar_minutes(timeframe)))
    head = session.iloc[:n_bars]
    if head.empty:
        return float("nan"), float("nan"), 0
    return float(head["high"].max()), float(head["low"].min()), n_bars
