"""Market-regime classification for strategy gating.

Breakout strategies (ORB, momentum) and mean-reversion strategies (VWAP
pullback) are opposite bets on range expansion; running both unconditionally
means which one wins a symbol is an ordering accident. The classifier sorts
each symbol's session into one of three regimes:

* ``trend``    — range is expanding and price is one-sided vs VWAP.
* ``balanced`` — two-sided rotation around VWAP (mean-reversion friendly).
* ``quiet``    — compressed volatility; no edge for either family.

Deliberately simple (two thresholds, no fitted model): an auditable rule
beats an opaque one until there is data to justify otherwise.
"""

from __future__ import annotations

from enum import Enum

from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.strategy.util import session_frame


class Regime(str, Enum):
    TREND = "trend"
    BALANCED = "balanced"
    QUIET = "quiet"


def classify(snapshot: MarketSnapshot) -> tuple[Regime, dict[str, float]]:
    """Classify the snapshot's current session; returns (regime, diagnostics)."""
    details: dict[str, float] = {}

    daily = snapshot.frame(Timeframe.DAY)
    intraday = None
    for tf in (Timeframe.MIN_5, Timeframe.MIN_15, Timeframe.MIN_1):
        intraday = snapshot.frame(tf)
        if intraday is not None and not intraday.empty:
            break

    # ── daily ATR context ──────────────────────────────────────
    atr_rank = 0.5
    day_atr = 0.0
    if daily is not None and "atr_pct" in daily.columns:
        hist = daily["atr_pct"].dropna().tail(60)
        if len(hist) >= 20:
            atr_rank = float((hist <= hist.iloc[-1]).mean())
        if "atr" in daily.columns and not daily["atr"].dropna().empty:
            day_atr = float(daily["atr"].dropna().iloc[-1])
    details["atr_percentile"] = round(atr_rank, 3)

    # ── session trend evidence ─────────────────────────────────
    range_ext = 0.0
    vwap_side = 0.5
    if intraday is not None and not intraday.empty:
        session = session_frame(intraday)
        if len(session) >= 3:
            session_range = float(session["high"].max() - session["low"].min())
            if day_atr > 0:
                range_ext = session_range / day_atr
            if "vwap" in session.columns:
                above = float((session["close"] > session["vwap"]).mean())
                vwap_side = max(above, 1.0 - above)
    details["range_extension"] = round(range_ext, 3)
    details["vwap_one_sidedness"] = round(vwap_side, 3)

    if atr_rank <= 0.2 and range_ext < 0.5:
        return Regime.QUIET, details
    if range_ext >= 0.8 and vwap_side >= 0.7:
        return Regime.TREND, details
    return Regime.BALANCED, details
