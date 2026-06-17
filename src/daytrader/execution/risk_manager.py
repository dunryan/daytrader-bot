"""RiskManager: position sizing, stops, take-profit, and account guards.

Pure logic (no I/O), so every rule is unit-testable. The engine calls these
helpers; it never inlines risk math itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from daytrader.config.settings import RiskConfig
from daytrader.execution.broker_base import Side


@dataclass
class SizingResult:
    qty: int
    risk_amount: float
    risk_per_share: float
    capped_by: str  # "risk" | "position_cap" | "cash" | "none"


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    # ── stops ──────────────────────────────────────────────────
    def resolve_stop(
        self,
        side: Side,
        entry: float,
        atr: float,
        stop_hint: float | None = None,
        support_levels: list[float] | None = None,
        resistance_levels: list[float] | None = None,
    ) -> float:
        """Determine the initial hard stop.

        Priority: a valid strategy ``stop_hint`` → the configured method
        (``structural`` swing level when available, else ATR) → ATR fallback.
        """
        if stop_hint is not None and self._valid_stop(side, entry, stop_hint):
            return stop_hint

        if self.config.stop_loss.method == "structural":
            level = self._structural_stop(side, entry, support_levels, resistance_levels)
            if level is not None:
                return level

        mult = self.config.stop_loss.atr_multiplier
        return entry - mult * atr if side is Side.BUY else entry + mult * atr

    @staticmethod
    def _valid_stop(side: Side, entry: float, stop: float) -> bool:
        return stop < entry if side is Side.BUY else stop > entry

    @staticmethod
    def _structural_stop(
        side: Side, entry: float,
        support_levels: list[float] | None, resistance_levels: list[float] | None,
    ) -> float | None:
        if side is Side.BUY and support_levels:
            below = [lvl for lvl in support_levels if lvl < entry]
            return max(below) if below else None
        if side is Side.SELL and resistance_levels:
            above = [lvl for lvl in resistance_levels if lvl > entry]
            return min(above) if above else None
        return None

    # ── take-profit ────────────────────────────────────────────
    def take_profit(self, side: Side, entry: float, stop: float) -> float:
        rr = self.config.take_profit.risk_reward_ratio
        risk = abs(entry - stop)
        return entry + rr * risk if side is Side.BUY else entry - rr * risk

    def update_trailing_stop(
        self,
        side: Side,
        current_stop: float,
        peak_price: float,
        atr: float,
        entry: float | None = None,
        initial_risk: float | None = None,
    ) -> float:
        """Ratchet the stop in the trade's favor; never loosens it.

        When ``entry``/``initial_risk`` are provided, the trail only activates
        after the trade has moved at least +1R in its favor; on activation the
        stop jumps to at least breakeven, then trails ``mult * atr`` behind
        the peak. Without them (legacy callers/tests), trailing is immediate.
        """
        if self.config.take_profit.method != "trailing":
            return current_stop
        mult = self.config.take_profit.trailing_atr_multiplier

        if entry is not None and initial_risk is not None and initial_risk > 0:
            favorable = (peak_price - entry) if side is Side.BUY else (entry - peak_price)
            if favorable < initial_risk:
                return current_stop  # not yet +1R; leave the initial stop alone
            if side is Side.BUY:
                return max(current_stop, entry, peak_price - mult * atr)
            return min(current_stop, entry, peak_price + mult * atr)

        if side is Side.BUY:
            return max(current_stop, peak_price - mult * atr)
        return min(current_stop, peak_price + mult * atr)

    # ── sizing ─────────────────────────────────────────────────
    def position_size(
        self,
        equity: float,
        entry: float,
        stop: float,
        available_cash: float | None = None,
        risk_pct: float | None = None,
    ) -> SizingResult:
        """Size so that a stop-out loses at most the risk budget.

        ``risk_pct`` overrides the configured ``max_risk_per_trade_pct`` (used
        by Kelly sizing); it is always clamped to that configured maximum.
        """
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0 or entry <= 0:
            return SizingResult(0, 0.0, risk_per_share, "none")

        effective_pct = self.config.max_risk_per_trade_pct
        if risk_pct is not None:
            effective_pct = min(max(risk_pct, 0.0), self.config.max_risk_per_trade_pct)

        risk_budget = equity * effective_pct / 100.0
        qty = math.floor(risk_budget / risk_per_share)
        capped_by = "risk"

        max_notional = equity * self.config.max_position_size_pct / 100.0
        max_qty_position = math.floor(max_notional / entry)
        if max_qty_position < qty:
            qty, capped_by = max_qty_position, "position_cap"

        if available_cash is not None:
            max_qty_cash = math.floor(available_cash / entry)
            if max_qty_cash < qty:
                qty, capped_by = max_qty_cash, "cash"

        qty = max(qty, 0)
        if qty == 0:
            capped_by = "none"
        return SizingResult(
            qty=qty,
            risk_amount=qty * risk_per_share,
            risk_per_share=risk_per_share,
            capped_by=capped_by,
        )

    # ── Kelly sizing ───────────────────────────────────────────
    def kelly_risk_pct(self, win_rate: float, payoff_ratio: float) -> float:
        """Fractional-Kelly risk percentage, capped at the configured maximum.

        Kelly fraction: f* = p - (1 - p) / b, where p is the win rate and b
        the payoff ratio (avg win / avg loss). Scaled by the configured
        fraction (default quarter-Kelly); a non-positive edge returns 0,
        which suppresses the trade entirely.
        """
        if payoff_ratio <= 0:
            return 0.0
        f_star = win_rate - (1.0 - win_rate) / payoff_ratio
        if f_star <= 0:
            return 0.0
        scaled_pct = f_star * self.config.kelly.fraction * 100.0
        return min(scaled_pct, self.config.max_risk_per_trade_pct)

    # ── account-level guards ───────────────────────────────────
    def drawdown_pct(self, reference_equity: float, current_equity: float) -> float:
        """Percentage decline from ``reference_equity`` (day start or peak)."""
        if reference_equity <= 0:
            return 0.0
        return (reference_equity - current_equity) / reference_equity * 100.0

    def is_drawdown_breached(self, reference_equity: float, current_equity: float) -> bool:
        return self.drawdown_pct(reference_equity, current_equity) >= self.config.max_daily_drawdown_pct

    def can_open_new(self, open_positions: int) -> bool:
        return open_positions < self.config.max_open_positions
