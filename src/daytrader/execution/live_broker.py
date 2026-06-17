"""LiveBroker — routes real orders through Alpaca (paper or live endpoint).

Used only when ``SIMULATION_MODE`` is False. The Alpaca SDK is imported lazily.

Execution policy:
* Entries are **marketable limit orders** with a price collar (never blind
  market orders crossing an uncapped spread) submitted as a bracket/OTO so a
  protective stop (and optional take-profit) rests at the broker the moment
  the entry fills. A process crash therefore never leaves a naked position.
* Unfilled entries are cancelled after a short poll window and reported as
  ``None`` so the engine simply skips the trade.
* Trailing is implemented by replacing the resting stop leg.
"""

from __future__ import annotations

import time

from daytrader.execution.broker_base import Broker, FillResult, Side
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


class LiveBroker(Broker):
    is_simulated = False

    def __init__(
        self,
        api_key: str | None,
        secret_key: str | None,
        base_url: str,
        entry_collar_bps: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.paper = "paper" in (base_url or "")
        self.entry_collar_bps = entry_collar_bps
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not (self.api_key and self.secret_key):
            raise RuntimeError("LiveBroker requires ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("alpaca-py not installed; cannot use LiveBroker.") from exc
        self._client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        logger.warning("LiveBroker active (paper=%s) — REAL orders will be sent.", self.paper)
        return self._client

    # ── account ────────────────────────────────────────────────
    def get_cash(self) -> float:
        client = self._ensure_client()
        return float(client.get_account().cash)

    def get_equity(self) -> float:
        client = self._ensure_client()
        return float(client.get_account().equity)

    def list_positions(self) -> dict[str, float] | None:
        client = self._ensure_client()
        try:
            out: dict[str, float] = {}
            for p in client.get_all_positions():
                qty = float(p.qty)
                if "short" in str(getattr(p, "side", "")).lower():
                    qty = -abs(qty)
                out[str(p.symbol)] = qty
            return out
        except Exception:  # noqa: BLE001
            logger.exception("Failed to list broker positions")
            return None

    # ── helpers ────────────────────────────────────────────────
    def _collared_limit(self, side: Side, ref_price: float) -> float:
        collar = self.entry_collar_bps / 10_000.0
        factor = 1.0 + collar if side is Side.BUY else 1.0 - collar
        return round(ref_price * factor, 2)

    def _await_fill_price(self, client, order_id, attempts: int = 10) -> float | None:
        """Poll briefly for the average fill price; ``None`` if unfilled."""
        for _ in range(attempts):
            o = client.get_order_by_id(order_id)
            if getattr(o, "filled_avg_price", None):
                return float(o.filled_avg_price)
            time.sleep(0.5)
        return None

    # ── orders ─────────────────────────────────────────────────
    def fill_market(self, symbol: str, side: Side, qty: float, ref_price: float) -> FillResult:
        """Plain market order — used for exits/flattens only."""
        client = self._ensure_client()
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side is Side.BUY else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        fill_price = self._await_fill_price(client, order.id)
        if fill_price is None:
            logger.warning("Fill price unavailable for order %s; using ref price.", order.id)
            fill_price = ref_price
        return FillResult(
            symbol=symbol, side=side, qty=qty, fill_price=fill_price,
            broker_order_id=str(order.id),
        )

    def submit_bracket(
        self,
        symbol: str,
        side: Side,
        qty: float,
        ref_price: float,
        stop: float,
        target: float | None = None,
    ) -> FillResult | None:
        """Collared limit entry with broker-side stop (and optional TP) legs."""
        client = self._ensure_client()
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side is Side.BUY else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=self._collared_limit(side, ref_price),
            order_class=OrderClass.BRACKET if target is not None else OrderClass.OTO,
            stop_loss=StopLossRequest(stop_price=round(stop, 2)),
            take_profit=(
                TakeProfitRequest(limit_price=round(target, 2)) if target is not None else None
            ),
        )
        order = client.submit_order(request)
        fill_price = self._await_fill_price(client, order.id)
        if fill_price is None:
            # Marketable limit didn't fill inside the collar — walk away.
            try:
                client.cancel_order_by_id(order.id)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to cancel unfilled entry %s", order.id)
            logger.info("Entry %s %s x%.0f unfilled within collar; cancelled.",
                        side.value, symbol, qty)
            return None
        slippage = abs(fill_price - ref_price) * qty
        return FillResult(
            symbol=symbol, side=side, qty=qty, fill_price=fill_price,
            slippage=slippage, broker_order_id=str(order.id),
        )

    def replace_stop(self, symbol: str, new_stop: float) -> bool:
        """Move the resting stop leg for ``symbol`` (trailing-stop ratchet)."""
        client = self._ensure_client()
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest

        try:
            orders = client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            )
            for o in orders:
                if getattr(o, "stop_price", None) is not None:
                    client.replace_order_by_id(
                        o.id, ReplaceOrderRequest(stop_price=round(new_stop, 2))
                    )
                    return True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to replace stop for %s", symbol)
        return False

    def cancel_symbol_orders(self, symbol: str) -> int:
        """Cancel all open orders for ``symbol`` (before a manual exit)."""
        client = self._ensure_client()
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        cancelled = 0
        try:
            orders = client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            )
            for o in orders:
                try:
                    client.cancel_order_by_id(o.id)
                    cancelled += 1
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to cancel order %s for %s", o.id, symbol)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to list open orders for %s", symbol)
        return cancelled
