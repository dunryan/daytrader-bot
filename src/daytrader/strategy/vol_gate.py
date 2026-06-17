"""Volatility regime gate (VIX / SPY proxy) for day-level trade permission.

Blocks *new* entries on sessions where the prior close indicates an elevated
volatility regime (Aug-2025-style chop). Open positions are still managed
elsewhere — this gate only vetoes signal generation.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from daytrader.config.settings import VolGateConfig
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _prior_row(df: pd.DataFrame, day: pd.Timestamp) -> pd.Series | None:
    """Last row strictly before ``day`` (normalized session date)."""
    if df is None or df.empty:
        return None
    d = df.sort_index()
    idx = pd.Index(d.index).normalize()
    prior = d[idx < day.normalize()]
    if prior.empty:
        return None
    return prior.iloc[-1]


def vix_metrics(vix_daily: pd.DataFrame | None, day: pd.Timestamp) -> dict[str, float]:
    """Prior-session VIX close and its 60-day percentile rank."""
    row = _prior_row(vix_daily, day)
    if row is None:
        return {}
    close = float(row["close"])
    hist = vix_daily.sort_index()
    idx = pd.Index(hist.index).normalize()
    window = hist[idx < day.normalize()].tail(60)["close"].dropna()
    pctile = float((window <= close).mean()) if len(window) >= 20 else 0.5
    return {"vix_prior": round(close, 3), "vix_percentile": round(pctile, 3)}


def spy_vol_metrics(spy_daily: pd.DataFrame | None, day: pd.Timestamp) -> dict[str, float]:
    """SPY ATR% percentile fallback when VIX history is unavailable."""
    row = _prior_row(spy_daily, day)
    if row is None or "atr_pct" not in spy_daily.columns:
        return {}
    hist = spy_daily.sort_index()
    idx = pd.Index(hist.index).normalize()
    window = hist[idx < day.normalize()].tail(60)["atr_pct"].dropna()
    if len(window) < 20:
        return {}
    current = float(row["atr_pct"])
    pctile = float((window <= current).mean())
    return {"spy_atr_pct": round(current, 3), "spy_atr_percentile": round(pctile, 3)}


def day_blocked(
    day: pd.Timestamp,
    config: VolGateConfig,
    vix_daily: pd.DataFrame | None = None,
    spy_daily: pd.DataFrame | None = None,
) -> tuple[bool, dict[str, float]]:
    """Return (blocked, diagnostics) for new entries on ``day``."""
    if config.mode == "off":
        return False, {}

    details: dict[str, float] = {}
    blocked = False
    reason = ""

    vx = vix_metrics(vix_daily, day)
    details.update(vx)
    if vx:
        if config.max_vix > 0 and vx["vix_prior"] > config.max_vix:
            blocked, reason = True, f"VIX {vx['vix_prior']:.1f} > {config.max_vix}"
        elif vx["vix_percentile"] > config.max_vix_percentile:
            blocked, reason = True, (
                f"VIX pctile {vx['vix_percentile']:.2f} > {config.max_vix_percentile:.2f}"
            )

    if not blocked and config.use_spy_proxy and spy_daily is not None:
        sp = spy_vol_metrics(spy_daily, day)
        details.update(sp)
        if sp and sp.get("spy_atr_percentile", 0) > config.max_spy_atr_percentile:
            blocked, reason = True, (
                f"SPY ATR pctile {sp['spy_atr_percentile']:.2f} "
                f"> {config.max_spy_atr_percentile:.2f}"
            )

    if blocked and config.mode == "shadow":
        logger.info("VOL GATE SHADOW: would block %s (%s)", day.date(), reason)
        return False, details
    if blocked:
        logger.debug("VOL GATE: block %s (%s)", day.date(), reason)
    return blocked, details


def blocked_trading_days(
    start: dt.date,
    end: dt.date,
    config: VolGateConfig,
    vix_daily: pd.DataFrame | None = None,
    spy_daily: pd.DataFrame | None = None,
) -> set[pd.Timestamp]:
    """All session dates in [start, end] where new entries are blocked."""
    if config.mode == "off":
        return set()
    blocked: set[pd.Timestamp] = set()
    day = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    while day <= end_ts:
        is_blocked, _ = day_blocked(day, config, vix_daily, spy_daily)
        if is_blocked:
            blocked.add(day.normalize())
        day += pd.Timedelta(days=1)
    return blocked
