"""Technical indicators (Module 2).

All functions are pure: they take/return pandas objects and never do I/O, so
they are fully unit-testable against known values. Conventions:

* Input frames carry canonical lowercase columns ``open, high, low, close,
  volume`` indexed by a timezone-aware timestamp.
* EMA/RSI/ATR use the standard Wilder / exponential definitions used by most
  charting platforms so values line up with what a trader sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════
#  Moving averages
# ════════════════════════════════════════════════════════════
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (adjust=False, charting-platform style)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


# ════════════════════════════════════════════════════════════
#  RSI (Wilder)
# ════════════════════════════════════════════════════════════
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 the asset only rose -> RSI 100.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


# ════════════════════════════════════════════════════════════
#  MACD
# ════════════════════════════════════════════════════════════
def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


# ════════════════════════════════════════════════════════════
#  ATR (Wilder) — volatility
# ════════════════════════════════════════════════════════════
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ════════════════════════════════════════════════════════════
#  VWAP — intraday, resets each session
# ════════════════════════════════════════════════════════════
def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price, reset per calendar day.

    Uses the typical price (H+L+C)/3. Index must be datetime; grouping by the
    date component makes each session start fresh, which is how intraday VWAP
    is meant to behave.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    dates = pd.Index(df.index).normalize()
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()
    return (cum_pv / cum_vol.replace(0, np.nan)).rename("vwap")


# ════════════════════════════════════════════════════════════
#  Support / Resistance
# ════════════════════════════════════════════════════════════
def pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    """Classic floor-trader pivot levels from the prior session's HLC."""
    pp = (prev_high + prev_low + prev_close) / 3.0
    return {
        "pivot": pp,
        "r1": 2 * pp - prev_low,
        "s1": 2 * pp - prev_high,
        "r2": pp + (prev_high - prev_low),
        "s2": pp - (prev_high - prev_low),
        "r3": prev_high + 2 * (pp - prev_low),
        "s3": prev_low - 2 * (prev_high - pp),
    }


def swing_levels(df: pd.DataFrame, window: int = 5) -> dict[str, list[float]]:
    """Detect recent swing highs (resistance) and lows (support).

    A bar is a swing high if its high is the max within +/- ``window`` bars
    (and symmetric for lows). Returns sorted, de-duplicated price levels.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    resistance: list[float] = []
    support: list[float] = []
    for i in range(window, n - window):
        hi_win = highs[i - window : i + window + 1]
        lo_win = lows[i - window : i + window + 1]
        if highs[i] == hi_win.max():
            resistance.append(float(highs[i]))
        if lows[i] == lo_win.min():
            support.append(float(lows[i]))
    return {
        "resistance": sorted(set(resistance)),
        "support": sorted(set(support)),
    }


# ════════════════════════════════════════════════════════════
#  Enrichment: append the full indicator set to a frame
# ════════════════════════════════════════════════════════════
def add_indicators(
    df: pd.DataFrame,
    ema_periods: list[int] | None = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    include_vwap: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``df`` with indicator columns appended.

    Columns added: ``ema_<p>`` for each period, ``rsi``, ``atr``, ``atr_pct``,
    ``macd``/``macd_signal``/``macd_hist`` and (if intraday) ``vwap``.
    """
    if df is None or df.empty:
        return df
    ema_periods = ema_periods or [9, 21, 50, 200]
    out = df.copy()

    for p in ema_periods:
        out[f"ema_{p}"] = ema(out["close"], p)

    out["rsi"] = rsi(out["close"], rsi_period)

    out["atr"] = atr(out["high"], out["low"], out["close"], atr_period)
    out["atr_pct"] = (out["atr"] / out["close"]) * 100.0

    fast, slow, signal = macd_params
    out = out.join(macd(out["close"], fast, slow, signal))

    if include_vwap:
        out["vwap"] = vwap(out)

    return out
