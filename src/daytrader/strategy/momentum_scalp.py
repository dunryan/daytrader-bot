"""Strategy C: Momentum Scalping.

Idea: a fast EMA crossing the slow EMA *confirmed by a relative-volume spike*
marks an impulse worth scalping. We require the cross to be fresh (it happened
on the latest bar) so we enter the move rather than chase it.
"""

from __future__ import annotations

import math

from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.strategy.base import Direction, Signal, Strategy
from daytrader.strategy.util import relative_volume


class MomentumScalpStrategy(Strategy):
    name = "momentum_scalp"
    timeframe = Timeframe.MIN_5

    def __init__(self, fast_ema: int = 9, slow_ema: int = 21, min_relative_volume: float = 2.0):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.min_rvol = min_relative_volume

    def evaluate(self, snapshot: MarketSnapshot) -> Signal:
        df = snapshot.frame(self.timeframe)
        if df is None or len(df) < 3:
            return self._hold(snapshot.symbol, "insufficient data")

        fast_col, slow_col = f"ema_{self.fast_ema}", f"ema_{self.slow_ema}"
        if fast_col not in df.columns or slow_col not in df.columns:
            return self._hold(snapshot.symbol, "EMA columns missing")

        last, prev = df.iloc[-1], df.iloc[-2]
        vals = [last[fast_col], last[slow_col], prev[fast_col], prev[slow_col], last.get("atr")]
        if any(v is None or math.isnan(float(v)) for v in vals):
            return self._hold(snapshot.symbol, "indicators not ready")

        fast_now, slow_now = float(last[fast_col]), float(last[slow_col])
        fast_prev, slow_prev = float(prev[fast_col]), float(prev[slow_col])
        close = float(last["close"])
        atr = float(last["atr"])
        rvol = relative_volume(df)

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now
        indicators = {
            fast_col: fast_now, slow_col: slow_now,
            "relative_volume": round(rvol, 3), "atr": atr, "close": close,
        }

        if rvol < self.min_rvol:
            return self._hold(snapshot.symbol, f"RVOL {rvol:.2f} < {self.min_rvol}")

        if crossed_up:
            confidence = min(1.0, 0.5 + 0.25 * (rvol - self.min_rvol))
            return Signal(
                symbol=snapshot.symbol, strategy=self.name, direction=Direction.BUY,
                price=close, confidence=round(confidence, 3),
                rationale=(
                    f"Momentum long: EMA{self.fast_ema} crossed above EMA{self.slow_ema} "
                    f"on RVOL {rvol:.2f}x"
                ),
                timeframe=self.timeframe.value, indicators=indicators,
                stop_hint=close - 1.5 * atr, target_hint=close + 2 * atr,
            )

        if crossed_down:
            confidence = min(1.0, 0.5 + 0.25 * (rvol - self.min_rvol))
            return Signal(
                symbol=snapshot.symbol, strategy=self.name, direction=Direction.SELL,
                price=close, confidence=round(confidence, 3),
                rationale=(
                    f"Momentum short: EMA{self.fast_ema} crossed below EMA{self.slow_ema} "
                    f"on RVOL {rvol:.2f}x"
                ),
                timeframe=self.timeframe.value, indicators=indicators,
                stop_hint=close + 1.5 * atr, target_hint=close - 2 * atr,
            )

        return self._hold(snapshot.symbol, "no fresh EMA cross")
