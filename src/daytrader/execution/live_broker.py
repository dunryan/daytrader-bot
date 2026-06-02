"""LiveBroker — routes real orders through Alpaca (paper or live endpoint).

Used only when ``SIMULATION_MODE`` is False. The Alpaca SDK is imported lazily.
This is a deliberately conservative implementation: it submits market orders
and reads the account; advanced order types are out of scope for now. Treat
"live" against the *paper* endpoint until you have validated the full system.
"""

from __future__ import annotations

import time

from daytrader.execution.broker_base import Broker, FillResult, Side
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


class LiveBroker(Broker):
    is_simulated = False

    def __init__(self, api_key: str | None, secret_key: str | None, base_url: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.paper = "paper" in (base_url or "")
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

    def get_cash(self) -> float:
        client = self._ensure_client()
        return float(client.get_account().cash)

    def fill_market(self, symbol: str, side: Side, qty: float, ref_price: float) -> FillResult:
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
        fill_price = self._await_fill_price(client, order.id, ref_price)
        return FillResult(
            symbol=symbol, side=side, qty=qty, fill_price=fill_price,
            broker_order_id=str(order.id),
        )

    def _await_fill_price(self, client, order_id, ref_price: float, attempts: int = 5) -> float:
        """Poll briefly for the average fill price; fall back to ref price."""
        for _ in range(attempts):
            o = client.get_order_by_id(order_id)
            if getattr(o, "filled_avg_price", None):
                return float(o.filled_avg_price)
            time.sleep(0.5)
        logger.warning("Fill price unavailable for order %s; using ref price.", order_id)
        return ref_price
