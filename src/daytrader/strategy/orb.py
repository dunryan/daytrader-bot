"""Strategy B: Opening Range Breakout (ORB).

Idea: the high/low of the first N minutes of the session frames the day's
initial balance. A break of that range signals directional intent.

Two entry modes:

- ``breakout`` (default): enter as soon as a bar closes beyond the range.
- ``retest``: after a confirmed break, wait for price to pull back and *hold*
  the broken level, then enter at the level with a structural stop at the
  range midpoint and a measured-move target from the level.

A/B result (2026-06, regime-enforced cohorts): retest did not improve
expectancy — PF 0.76 vs 0.74 on gap days with 4x fewer trades (28 vs 117),
and payoff stayed ~0.9-1.0. Successful breaks tend not to retest within the
window, so the retest sample self-selects weaker breaks. Kept as a research
mode; breakout remains the default.

Retest setup rules (long; short is mirrored):
1. First post-range bar that CLOSES above or_high arms the setup.
2. Any subsequent CLOSE below the range midpoint kills it (failed breakout).
3. Entry fires on the first bar whose low touches the level zone
   (or_high + tolerance) while its close holds at/above the level zone,
   within ``retest_max_bars`` of the break. One setup per side per session.
"""

from __future__ import annotations

import math

from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.strategy.base import Direction, Signal, Strategy
from daytrader.strategy.gap_features import (
    compute_gap_features,
    direction_matches_signal,
    session_open_price,
)
from daytrader.strategy.util import opening_range, relative_volume_tod, session_frame


class OpeningRangeBreakoutStrategy(Strategy):
    name = "opening_range_breakout"
    timeframe = Timeframe.MIN_5

    def __init__(
        self,
        opening_range_minutes: int = 15,
        volume_confirmation: bool = True,
        entry_mode: str = "breakout",
        retest_max_bars: int = 12,
        touch_tolerance_atr: float = 0.25,
        require_gap_direction_match: bool = False,
        max_gap_norm: float = 0.0,
        max_entry_minutes_after_open: int = 0,
    ) -> None:
        if entry_mode not in ("breakout", "retest"):
            raise ValueError(f"entry_mode must be 'breakout' or 'retest', got {entry_mode!r}")
        self.opening_range_minutes = opening_range_minutes
        self.volume_confirmation = volume_confirmation
        self.entry_mode = entry_mode
        self.retest_max_bars = retest_max_bars
        self.touch_tolerance_atr = touch_tolerance_atr
        self.require_gap_direction_match = require_gap_direction_match
        self.max_gap_norm = max_gap_norm
        self.max_entry_minutes_after_open = max_entry_minutes_after_open

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

        if self.max_entry_minutes_after_open > 0:
            session_start = session.index[0]
            elapsed_min = (df.index[-1] - session_start).total_seconds() / 60.0
            if elapsed_min > self.max_entry_minutes_after_open:
                return self._hold(snapshot.symbol, "past entry window")

        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(last.get("atr")) if last.get("atr") is not None else 0.0
        rvol = relative_volume_tod(df)
        vol_ok = (not self.volume_confirmation) or rvol >= 1.0

        daily = snapshot.frame(Timeframe.DAY)
        open_px = session_open_price(session)
        gap_feats = compute_gap_features(daily, open_px) if open_px else {}
        if self.max_gap_norm > 0 and gap_feats.get("gap_norm", 0) > self.max_gap_norm:
            return self._hold(snapshot.symbol, "gap exceeds max_gap_norm")

        indicators = {
            "close": close, "or_high": or_high, "or_low": or_low,
            "or_bars": n_or, "relative_volume": round(rvol, 3), "atr": atr,
            "entry_mode": self.entry_mode,
            **gap_feats,
        }
        or_size = max(or_high - or_low, 1e-9)
        or_mid = (or_high + or_low) / 2.0

        if self.entry_mode == "breakout":
            return self._evaluate_breakout(
                snapshot.symbol, close, or_high, or_low, or_size, rvol, vol_ok, indicators
            )
        return self._evaluate_retest(
            snapshot.symbol, session.iloc[n_or:], close, atr,
            or_high, or_low, or_mid, or_size, rvol, vol_ok, indicators,
        )

    # ── legacy chase-the-break entry ─────────────────────────────

    def _evaluate_breakout(
        self, symbol, close, or_high, or_low, or_size, rvol, vol_ok, indicators
    ) -> Signal:
        gap_dir = indicators.get("gap_direction", 0)
        if close > or_high and vol_ok:
            if self.require_gap_direction_match and not direction_matches_signal(
                gap_dir, Direction.BUY.value
            ):
                return self._hold(symbol, "long blocked: gap direction mismatch")
            confidence = min(1.0, 0.5 + min((close - or_high) / or_size, 0.5))
            return Signal(
                symbol=symbol, strategy=self.name, direction=Direction.BUY,
                price=close, confidence=round(confidence, 3),
                rationale=(
                    f"ORB long: broke {self.opening_range_minutes}m high {or_high:.2f} "
                    f"(RVOL {rvol:.2f}x)"
                ),
                timeframe=self.timeframe.value, indicators=indicators,
                stop_hint=or_low, target_hint=close + or_size,
            )
        if close < or_low and vol_ok:
            if self.require_gap_direction_match and not direction_matches_signal(
                gap_dir, Direction.SELL.value
            ):
                return self._hold(symbol, "short blocked: gap direction mismatch")
            confidence = min(1.0, 0.5 + min((or_low - close) / or_size, 0.5))
            return Signal(
                symbol=symbol, strategy=self.name, direction=Direction.SELL,
                price=close, confidence=round(confidence, 3),
                rationale=(
                    f"ORB short: broke {self.opening_range_minutes}m low {or_low:.2f} "
                    f"(RVOL {rvol:.2f}x)"
                ),
                timeframe=self.timeframe.value, indicators=indicators,
                stop_hint=or_high, target_hint=close - or_size,
            )
        return self._hold(symbol, "no opening-range breakout")

    # ── break-then-retest entry ──────────────────────────────────

    def _evaluate_retest(
        self, symbol, post, close, atr,
        or_high, or_low, or_mid, or_size, rvol, vol_ok, indicators,
    ) -> Signal:
        """``post`` = session bars after the opening-range window.

        Stateless: rescans the session every bar and only fires when the
        *current* bar is the first qualifying retest bar, so the signal is
        emitted exactly once per side per session.
        """
        tol = self.touch_tolerance_atr * max(atr, 1e-9)
        gap_dir = indicators.get("gap_direction", 0)
        closes = post["close"].astype(float)
        lows = post["low"].astype(float)
        highs = post["high"].astype(float)
        cur = len(post) - 1  # current bar's position within `post`

        # ── long: first close above or_high arms the setup ──
        broke_up = closes.gt(or_high)
        if broke_up.any():
            b = int(broke_up.argmax())
            if cur > b:
                between = closes.iloc[b + 1: cur]  # bars after break, before now
                invalidated = bool((between < or_mid).any())
                stale = (cur - b) > self.retest_max_bars
                already_retested = bool(
                    (
                        (lows.iloc[b + 1: cur] <= or_high + tol)
                        & (closes.iloc[b + 1: cur] >= or_high - tol)
                    ).any()
                )
                touches_now = (
                    float(lows.iloc[cur]) <= or_high + tol and close >= or_high - tol
                )
                if (
                    not invalidated and not stale and not already_retested
                    and touches_now and vol_ok
                    and (not self.require_gap_direction_match
                         or direction_matches_signal(gap_dir, Direction.BUY.value))
                ):
                    confidence = min(1.0, 0.55 + 0.15 * max(0.0, rvol - 1.0))
                    return Signal(
                        symbol=symbol, strategy=self.name, direction=Direction.BUY,
                        price=close, confidence=round(confidence, 3),
                        rationale=(
                            f"ORB retest long: held {self.opening_range_minutes}m high "
                            f"{or_high:.2f} after break (RVOL {rvol:.2f}x)"
                        ),
                        timeframe=self.timeframe.value, indicators=indicators,
                        stop_hint=or_mid,
                        target_hint=or_high + or_size,
                    )

        # ── short: mirror ──
        broke_dn = closes.lt(or_low)
        if broke_dn.any():
            b = int(broke_dn.argmax())
            if cur > b:
                between = closes.iloc[b + 1: cur]
                invalidated = bool((between > or_mid).any())
                stale = (cur - b) > self.retest_max_bars
                already_retested = bool(
                    (
                        (highs.iloc[b + 1: cur] >= or_low - tol)
                        & (closes.iloc[b + 1: cur] <= or_low + tol)
                    ).any()
                )
                touches_now = (
                    float(highs.iloc[cur]) >= or_low - tol and close <= or_low + tol
                )
                if (
                    not invalidated and not stale and not already_retested
                    and touches_now and vol_ok
                    and (not self.require_gap_direction_match
                         or direction_matches_signal(gap_dir, Direction.SELL.value))
                ):
                    confidence = min(1.0, 0.55 + 0.15 * max(0.0, rvol - 1.0))
                    return Signal(
                        symbol=symbol, strategy=self.name, direction=Direction.SELL,
                        price=close, confidence=round(confidence, 3),
                        rationale=(
                            f"ORB retest short: held {self.opening_range_minutes}m low "
                            f"{or_low:.2f} after break (RVOL {rvol:.2f}x)"
                        ),
                        timeframe=self.timeframe.value, indicators=indicators,
                        stop_hint=or_mid,
                        target_hint=or_low - or_size,
                    )

        return self._hold(symbol, "no qualifying opening-range retest")
