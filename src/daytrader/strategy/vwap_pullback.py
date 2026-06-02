"""Strategy A: VWAP Pullback / Reversal.

Idea: in a trending session, price repeatedly pulls back toward VWAP and
resumes. We go long when price is in an uptrend (above the trend EMA) and has
pulled back to within a small ATR-distance of VWAP while holding above it;
symmetric short logic applies in a downtrend.
"""

from __future__ import annotations

import math

from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.strategy.base import Direction, Signal, Strategy


class VwapPullbackStrategy(Strategy):
    name = "vwap_pullback"
    timeframe = Timeframe.MIN_5

    def __init__(self, trend_ema: int = 21, max_distance_from_vwap_atr: float = 0.5) -> None:
        self.trend_ema = trend_ema
        self.max_distance_atr = max_distance_from_vwap_atr

    def evaluate(self, snapshot: MarketSnapshot) -> Signal:
        df = snapshot.frame(self.timeframe)
        if df is None or len(df) < 2:
            return self._hold(snapshot.symbol, "insufficient data")

        last = df.iloc[-1]
        close = float(last["close"])
        vwap = last.get(f"vwap")
        ema_col = f"ema_{self.trend_ema}"
        ema_trend = last.get(ema_col)
        atr = last.get("atr")

        if vwap is None or ema_trend is None or atr is None or any(
            map(lambda v: v is None or math.isnan(float(v)), [vwap, ema_trend, atr])
        ):
            return self._hold(snapshot.symbol, "indicators not ready")
        vwap, ema_trend, atr = float(vwap), float(ema_trend), float(atr)
        if atr <= 0:
            return self._hold(snapshot.symbol, "zero ATR")

        dist_atr = (close - vwap) / atr
        indicators = {"close": close, "vwap": vwap, ema_col: ema_trend, "atr": atr,
                      "dist_vwap_atr": round(dist_atr, 3)}

        uptrend = close > ema_trend
        downtrend = close < ema_trend
        within = abs(dist_atr) <= self.max_distance_atr

        if uptrend and within and close >= vwap:
            confidence = max(0.0, 1.0 - abs(dist_atr) / self.max_distance_atr)
            return Signal(
                symbol=snapshot.symbol,
                strategy=self.name,
                direction=Direction.BUY,
                price=close,
                confidence=round(confidence, 3),
                rationale=(
                    f"VWAP pullback long: uptrend (close>{ema_col}), price {dist_atr:+.2f} ATR "
                    f"from VWAP and holding above it"
                ),
                timeframe=self.timeframe.value,
                indicators=indicators,
                stop_hint=vwap - atr,
                target_hint=close + 2 * atr,
            )

        if downtrend and within and close <= vwap:
            confidence = max(0.0, 1.0 - abs(dist_atr) / self.max_distance_atr)
            return Signal(
                symbol=snapshot.symbol,
                strategy=self.name,
                direction=Direction.SELL,
                price=close,
                confidence=round(confidence, 3),
                rationale=(
                    f"VWAP pullback short: downtrend (close<{ema_col}), price {dist_atr:+.2f} ATR "
                    f"from VWAP and holding below it"
                ),
                timeframe=self.timeframe.value,
                indicators=indicators,
                stop_hint=vwap + atr,
                target_hint=close - 2 * atr,
            )

        return self._hold(snapshot.symbol, "no VWAP pullback setup")
