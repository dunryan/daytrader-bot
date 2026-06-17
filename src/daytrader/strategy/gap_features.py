"""Opening-gap feature helpers for ORB signals and meta-labeling."""

from __future__ import annotations

import pandas as pd

from daytrader.data.providers.base import Timeframe
from daytrader.strategy.util import session_frame


def compute_gap_features(
    daily: pd.DataFrame | None,
    session_open: float,
) -> dict[str, float]:
    """Gap metrics known at the session open (no look-ahead).

    ``daily`` must contain only sessions *strictly before* today (as the
    backtest engine already supplies). ``session_open`` is today's RTH open.
    """
    if daily is None or daily.empty or session_open <= 0:
        return {}
    prev_close = float(daily["close"].iloc[-1])
    if prev_close <= 0:
        return {}
    gap_pct = (session_open - prev_close) / prev_close * 100.0
    atr = float(daily["atr"].iloc[-1]) if "atr" in daily.columns else 0.0
    gap_norm = abs(session_open - prev_close) / atr if atr > 0 else 0.0
    direction = 1.0 if gap_pct > 0 else (-1.0 if gap_pct < 0 else 0.0)
    return {
        "gap_pct": round(gap_pct, 3),
        "gap_norm": round(gap_norm, 3),
        "gap_direction": direction,
    }


def session_open_price(intraday: pd.DataFrame) -> float | None:
    """First bar open of the current session."""
    session = session_frame(intraday)
    if session is None or session.empty:
        return None
    return float(session["open"].iloc[0])


def gap_from_intraday(
    intraday: pd.DataFrame,
    daily: pd.DataFrame | None,
) -> dict[str, float]:
    """Combine intraday session open with prior daily close."""
    open_px = session_open_price(intraday)
    if open_px is None:
        return {}
    return compute_gap_features(daily, open_px)


def direction_matches_signal(gap_direction: float, direction: str) -> bool:
    """True when long aligns with gap-up (or short with gap-down)."""
    if gap_direction == 0:
        return True
    if direction in ("BUY", "LONG"):
        return gap_direction > 0
    if direction in ("SELL", "SHORT"):
        return gap_direction < 0
    return True
