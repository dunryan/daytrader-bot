"""Strategy B: Opening Range Breakout (ORB).

Idea: the high/low of the first N minutes of the session frames the day's
initial balance. A decisive break above that range (ideally on rising volume)
signals a long; a break below signals a short.
"""

from __future__ import annotations

import math

from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.strategy.base import Direction, Signal, Strategy
from daytrader.strategy.util import opening_range, relative_volume, session_frame


class OpeningRangeBreakoutStrategy(Strategy):
    name = "opening_range_breakout"
    timeframe = Timeframe.MIN_5

    def __init__(self, opening_range_minutes: int = 15, volume_confirmation: bool = True) -> None:
        self.opening_range_minutes = opening_range_minutes
        self.volume_confirmation = volume_confirmation

    def evaluate(self, snapshot: MarketSnapshot) -> Signal:
        df = snapshot.frame(self.timeframe)
        if df is None or len(df) < 2:
            return self._hold(snapshot.symbol, "insufficient data")

        session = session_frame(df)
        or_high, or_low, n_or = opening_range(df, self.opening_range_minutes, self.timeframe)
        if n_or == 0 or math.isnan(or_high):
            return self._hold(snapshot.symbol, "opening range not formed")

        # Must be past the opening-range window to trade the break.
        if len(session) <= n_or:
            return self._hold(snapshot.symbol, "still inside opening range window")

        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(last.get("atr")) if last.get("atr") is not None else 0.0
        rvol = relative_volume(df)
        vol_ok = (not self.volume_confirmation) or rvol >= 1.0

        indicators = {
            "close": close, "or_high": or_high, "or_low": or_low,
            "or_bars": n_or, "relative_volume": round(rvol, 3), "atr": atr,
        }
        or_size = max(or_high - or_low, 1e-9)

        if close > or_high and vol_ok:
            confidence = min(1.0, 0.5 + min((close - or_high) / or_size, 0.5))
            return Signal(
                symbol=snapshot.symbol,
                strategy=self.name,
                direction=Direction.BUY,
                price=close,
                confidence=round(confidence, 3),
                rationale=(
                    f"ORB long: broke {self.opening_range_minutes}m high {or_high:.2f} "
                    f"(RVOL {rvol:.2f}x)"
                ),
                timeframe=self.timeframe.value,
                indicators=indicators,
                stop_hint=or_low,
                target_hint=close + or_size,
            )

        if close < or_low and vol_ok:
            confidence = min(1.0, 0.5 + min((or_low - close) / or_size, 0.5))
            return Signal(
                symbol=snapshot.symbol,
                strategy=self.name,
                direction=Direction.SELL,
                price=close,
                confidence=round(confidence, 3),
                rationale=(
                    f"ORB short: broke {self.opening_range_minutes}m low {or_low:.2f} "
                    f"(RVOL {rvol:.2f}x)"
                ),
                timeframe=self.timeframe.value,
                indicators=indicators,
                stop_hint=or_high,
                target_hint=close - or_size,
            )

        return self._hold(snapshot.symbol, "no opening-range breakout")
