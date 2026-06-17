"""Session-boundary helpers: regular-trading-hours filtering.

Alpaca (and most vendors) include extended-hours rows in minute bars. Left
unfiltered, those rows corrupt every session-anchored computation downstream:
the opening range starts at 04:00 instead of 09:30, session VWAP is skewed by
thin pre-market prints, and relative-volume baselines average near-zero
overnight bars. All intraday frames are therefore RTH-filtered before
indicator enrichment.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)


def filter_rth(df: pd.DataFrame, tz: ZoneInfo | str = MARKET_TZ) -> pd.DataFrame:
    """Keep only bars whose start timestamp falls within [09:30, 16:00) ET.

    Bar timestamps label the bar *start*, so a 5-minute bar stamped 15:55 is
    the final RTH bar and a 16:00 bar is after-hours. Naive indexes are
    assumed to already be in market time.
    """
    if df is None or df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    local = idx.tz_convert(tz) if idx.tz is not None else idx
    times = local.time
    mask = (times >= RTH_OPEN) & (times < RTH_CLOSE)
    return df[mask]
