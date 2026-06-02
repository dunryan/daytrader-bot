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
    """Latest bar's volume divided by the mean of the prior ``window`` bars."""
    if df is None or len(df) < 2:
        return 0.0
    prior = df["volume"].iloc[-(window + 1) : -1]
    avg = float(prior.mean()) if len(prior) else 0.0
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
