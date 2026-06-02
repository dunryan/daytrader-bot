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
        self, side: Side, current_stop: float, peak_price: float, atr: float
    ) -> float:
        """Ratchet the stop in the trade's favor; never loosens it."""
        if self.config.take_profit.method != "trailing":
            return current_stop
        mult = self.config.take_profit.trailing_atr_multiplier
        if side is Side.BUY:
            return max(current_stop, peak_price - mult * atr)
        return min(current_stop, peak_price + mult * atr)

    # ── sizing ─────────────────────────────────────────────────
    def position_size(
        self, equity: float, entry: float, stop: float, available_cash: float | None = None
    ) -> SizingResult:
        """Size so that a stop-out loses at most ``max_risk_per_trade_pct`` of equity."""
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0 or entry <= 0:
            return SizingResult(0, 0.0, risk_per_share, "none")

        risk_budget = equity * self.config.max_risk_per_trade_pct / 100.0
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

    # ── account-level guards ───────────────────────────────────
    def drawdown_pct(self, starting_equity: float, current_equity: float) -> float:
        if starting_equity <= 0:
            return 0.0
        return (starting_equity - current_equity) / starting_equity * 100.0

    def is_drawdown_breached(self, starting_equity: float, current_equity: float) -> bool:
        return self.drawdown_pct(starting_equity, current_equity) >= self.config.max_daily_drawdown_pct

    def can_open_new(self, open_positions: int) -> bool:
        return open_positions < self.config.max_open_positions
