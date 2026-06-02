"""VirtualBroker — the paper-trading engine used when SIMULATION_MODE is True.

Maintains an isolated virtual cash balance, applies a configurable slippage
model to every fill (buys fill worse/higher, sells fill worse/lower), and
charges optional commission. It is intentionally *stateless about positions* —
the ExecutionEngine + PositionRepository own position bookkeeping so state
survives restarts. The broker only tracks cash and realized P/L.
"""

from __future__ import annotations

from daytrader.execution.broker_base import Broker, FillResult, Side
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


class VirtualBroker(Broker):
    is_simulated = True

    def __init__(
        self,
        starting_cash: float,
        slippage_pct: float = 0.05,
        commission_per_trade: float = 0.0,
    ) -> None:
        self._starting_cash = starting_cash
        self._cash = starting_cash
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        self.realized_pnl = 0.0

    # ── ledger ─────────────────────────────────────────────────
    def get_cash(self) -> float:
        return self._cash

    @property
    def starting_cash(self) -> float:
        return self._starting_cash

    def set_cash(self, cash: float) -> None:
        """Restore cash on rehydration (engine recomputes from DB state)."""
        self._cash = cash

    # ── fills ──────────────────────────────────────────────────
    def apply_slippage(self, ref_price: float, side: Side) -> float:
        """Worst-case slippage: buys pay up, sells receive less."""
        factor = 1.0 + self.slippage_pct / 100.0 if side is Side.BUY else 1.0 - self.slippage_pct / 100.0
        return ref_price * factor

    def fill_market(self, symbol: str, side: Side, qty: float, ref_price: float) -> FillResult:
        fill_price = self.apply_slippage(ref_price, side)
        slippage = abs(fill_price - ref_price) * qty
        commission = self.commission_per_trade
        notional = fill_price * qty

        if side is Side.BUY:
            self._cash -= notional + commission
        else:
            self._cash += notional - commission

        logger.info(
            "[SIM] %s %s x%.4f @ %.4f (ref %.4f, slip $%.2f) | cash=%.2f",
            side.value, symbol, qty, fill_price, ref_price, slippage, self._cash,
        )
        return FillResult(
            symbol=symbol,
            side=side,
            qty=qty,
            fill_price=fill_price,
            slippage=slippage,
            commission=commission,
            broker_order_id=None,
        )

    def record_realized(self, pnl: float) -> None:
        self.realized_pnl += pnl
