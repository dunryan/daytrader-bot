"""ExecutionEngine — turns signals into risk-checked, tracked positions.

Responsibilities:
* Gate everything behind ``SIMULATION_MODE`` by choosing Virtual/Live broker.
* Size each trade (1% risk), set ATR/structural stops and trailing take-profit.
* Persist orders, fills, and positions so state survives restarts.
* Manage open positions every cycle: trail stops, exit on stop/target.
* Enforce the account-level **daily-drawdown kill switch**: flatten and halt.

On startup :meth:`rehydrate` reloads open positions and the day's metrics from
the database, so a reboot never forgets open risk or a latched halt.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from daytrader.config.settings import Settings, get_settings
from daytrader.execution.broker_base import Broker, Side
from daytrader.execution.live_broker import LiveBroker
from daytrader.execution.risk_manager import RiskManager
from daytrader.execution.virtual_broker import VirtualBroker
from daytrader.persistence.database import Database
from daytrader.persistence.repositories import (
    DailyMetricsRepository,
    EquityRepository,
    OrderRepository,
    PositionRepository,
)
from daytrader.strategy.base import Direction, Signal
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class OpenPosition:
    """In-memory mirror of an open DB position, plus trailing state."""

    id: int
    symbol: str
    side: Side  # BUY = long, SELL = short
    qty: float
    entry_price: float
    entry_time: dt.datetime
    stop_loss: float
    take_profit: float
    strategy: str | None
    peak_price: float = field(default=0.0)

    def __post_init__(self) -> None:
        if not self.peak_price:
            self.peak_price = self.entry_price

    @property
    def direction(self) -> str:
        return "LONG" if self.side is Side.BUY else "SHORT"

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.qty if self.side is Side.BUY else (
            self.entry_price - price
        ) * self.qty


def build_broker(settings: Settings) -> Broker:
    """Select the broker implementation from the SIMULATION_MODE toggle."""
    if settings.app.simulation_mode:
        return VirtualBroker(
            starting_cash=settings.risk.starting_equity,
            slippage_pct=settings.risk.slippage_pct,
            commission_per_trade=settings.risk.commission_per_trade,
        )
    logger.warning("SIMULATION_MODE is False — using LiveBroker with REAL order routing.")
    return LiveBroker(
        api_key=settings.secrets.alpaca_api_key,
        secret_key=settings.secrets.alpaca_secret_key,
        base_url=settings.secrets.alpaca_base_url,
    )


class ExecutionEngine:
    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        risk: RiskManager,
        db: Database,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.risk = risk
        self.orders = OrderRepository(db)
        self.positions = PositionRepository(db)
        self.metrics = DailyMetricsRepository(db)
        self.equity = EquityRepository(db)

        self.open_positions: dict[str, OpenPosition] = {}
        self.trade_date: str = dt.date.today().isoformat()
        self.starting_equity: float = settings.risk.starting_equity
        self.halted: bool = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None, db: Database | None = None) -> "ExecutionEngine":
        settings = settings or get_settings()
        db = db or Database(settings.db_url)
        return cls(settings, build_broker(settings), RiskManager(settings.risk), db)

    # ── lifecycle / recovery ───────────────────────────────────
    def start_day(self, trade_date: dt.date | None = None) -> None:
        """Initialize (or resume) a trading day and rehydrate open risk."""
        trade_date = trade_date or dt.date.today()
        self.trade_date = trade_date.isoformat()
        metric = self.metrics.get_or_create(self.trade_date, self.settings.risk.starting_equity)
        self.starting_equity = metric.starting_equity
        self.halted = bool(metric.trading_halted)
        self.rehydrate()
        if self.halted:
            logger.warning("Resuming day %s with trading HALTED (%s)", self.trade_date, metric.halt_reason)

    def rehydrate(self) -> None:
        """Reload open positions from the DB and reconstruct virtual cash."""
        self.open_positions.clear()
        open_rows = self.positions.get_open()
        entry_notional = 0.0
        for row in open_rows:
            side = Side.BUY if row.direction == "LONG" else Side.SELL
            self.open_positions[row.symbol] = OpenPosition(
                id=row.id, symbol=row.symbol, side=side, qty=row.qty,
                entry_price=row.entry_price, entry_time=row.entry_time,
                stop_loss=row.stop_loss or 0.0, take_profit=row.take_profit or 0.0,
                strategy=row.strategy,
            )
            entry_notional += row.qty * row.entry_price * (1 if side is Side.BUY else -1)

        if isinstance(self.broker, VirtualBroker):
            realized_today = sum(
                p.realized_pnl or 0.0 for p in self.positions.get_closed_for_day(self.trade_date)
            )
            self.broker.set_cash(
                self.broker.starting_cash + realized_today - entry_notional
            )
        logger.info("Rehydrated %d open position(s) for %s", len(self.open_positions), self.trade_date)

    # ── equity ─────────────────────────────────────────────────
    def equity_value(self, prices: dict[str, float]) -> float:
        eq = self.broker.get_cash()
        for pos in self.open_positions.values():
            price = prices.get(pos.symbol, pos.entry_price)
            eq += pos.qty * price * (1 if pos.side is Side.BUY else -1)
        return eq

    # ── opening trades ─────────────────────────────────────────
    def process_signals(self, signals: list[Signal], prices: dict[str, float]) -> list[OpenPosition]:
        opened: list[OpenPosition] = []
        if self.halted:
            logger.info("Trading halted; ignoring %d signal(s).", len(signals))
            return opened

        equity = self.equity_value(prices)
        for sig in signals:
            if not sig.is_actionable:
                continue
            if sig.symbol in self.open_positions:
                continue  # one position per symbol
            if not self.risk.can_open_new(len(self.open_positions)):
                logger.info("Max open positions reached; skipping %s", sig.symbol)
                break

            pos = self._open_from_signal(sig, prices, equity)
            if pos is not None:
                opened.append(pos)
                equity = self.equity_value(prices)
        return opened

    def _open_from_signal(self, sig: Signal, prices: dict[str, float], equity: float) -> OpenPosition | None:
        side = Side.BUY if sig.direction is Direction.BUY else Side.SELL
        entry = sig.price or prices.get(sig.symbol)
        if not entry or entry <= 0:
            logger.warning("No entry price for %s; skipping", sig.symbol)
            return None
        atr = float(sig.indicators.get("atr") or 0.0)

        stop = self.risk.resolve_stop(side, entry, atr, stop_hint=sig.stop_hint)
        if not self.risk._valid_stop(side, entry, stop):
            logger.warning("Invalid stop for %s (entry=%.2f stop=%.2f); skipping", sig.symbol, entry, stop)
            return None
        take_profit = sig.target_hint or self.risk.take_profit(side, entry, stop)

        sizing = self.risk.position_size(equity, entry, stop, available_cash=self.broker.get_cash())
        if sizing.qty <= 0:
            logger.info("Sizing yielded 0 shares for %s (capped_by=%s); skipping", sig.symbol, sizing.capped_by)
            return None

        fill = self.broker.fill_market(sig.symbol, side, sizing.qty, entry)

        order_id = self.orders.create(
            signal_id=None, trade_date=self.trade_date, symbol=sig.symbol,
            side=side.value, order_type="MARKET", qty=sizing.qty,
            stop_loss=stop, take_profit=take_profit, risk_amount=sizing.risk_amount,
            status="FILLED", broker_order_id=fill.broker_order_id,
            is_simulated=self.broker.is_simulated,
        )
        self.orders.add_fill(order_id, sig.symbol, side.value, sizing.qty,
                             fill.fill_price, fill.slippage, fill.commission)

        pos_id = self.positions.open_position(
            trade_date=self.trade_date, symbol=sig.symbol, strategy=sig.strategy,
            direction="LONG" if side is Side.BUY else "SHORT", qty=sizing.qty,
            entry_price=fill.fill_price, entry_time=fill.timestamp,
            stop_loss=stop, take_profit=take_profit, is_simulated=self.broker.is_simulated,
        )
        pos = OpenPosition(
            id=pos_id, symbol=sig.symbol, side=side, qty=sizing.qty,
            entry_price=fill.fill_price, entry_time=fill.timestamp,
            stop_loss=stop, take_profit=take_profit, strategy=sig.strategy,
        )
        self.open_positions[sig.symbol] = pos
        logger.info("OPENED %s %s x%d @ %.2f stop=%.2f tp=%.2f (%s)",
                    pos.direction, sig.symbol, sizing.qty, fill.fill_price, stop, take_profit, sig.strategy)
        self._recompute_daily(self.equity_value(prices))
        return pos

    # ── managing open trades ───────────────────────────────────
    def manage_positions(self, prices: dict[str, float]) -> list[str]:
        """Trail stops and exit positions that hit stop/target. Returns closed symbols."""
        closed: list[str] = []
        for symbol, pos in list(self.open_positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            atr = abs(pos.entry_price - pos.stop_loss) / max(self.settings.risk.stop_loss.atr_multiplier, 1e-9)

            # Update peak and trailing stop.
            if pos.side is Side.BUY:
                pos.peak_price = max(pos.peak_price, price)
            else:
                pos.peak_price = min(pos.peak_price, price)
            new_stop = self.risk.update_trailing_stop(pos.side, pos.stop_loss, pos.peak_price, atr)
            if new_stop != pos.stop_loss:
                pos.stop_loss = new_stop
                self.positions.update_stops(pos.id, new_stop, None)

            reason = self._exit_reason(pos, price)
            if reason:
                self._close_position(pos, price, reason)
                closed.append(symbol)
        if closed:
            self._recompute_daily(self.equity_value(prices))
        return closed

    @staticmethod
    def _exit_reason(pos: OpenPosition, price: float) -> str | None:
        if pos.side is Side.BUY:
            if price <= pos.stop_loss:
                return "SL"
            if price >= pos.take_profit:
                return "TP"
        else:
            if price >= pos.stop_loss:
                return "SL"
            if price <= pos.take_profit:
                return "TP"
        return None

    def _close_position(self, pos: OpenPosition, price: float, reason: str) -> None:
        exit_side = pos.side.opposite
        fill = self.broker.fill_market(pos.symbol, exit_side, pos.qty, price)
        if pos.side is Side.BUY:
            realized = (fill.fill_price - pos.entry_price) * pos.qty
        else:
            realized = (pos.entry_price - fill.fill_price) * pos.qty
        realized -= fill.commission
        pnl_pct = realized / (pos.entry_price * pos.qty) * 100.0 if pos.entry_price else 0.0

        self.positions.close_position(
            pos.id, exit_price=fill.fill_price, exit_time=fill.timestamp,
            exit_reason=reason, realized_pnl=realized, pnl_percent=pnl_pct,
        )
        if isinstance(self.broker, VirtualBroker):
            self.broker.record_realized(realized)
        self.open_positions.pop(pos.symbol, None)
        logger.info("CLOSED %s %s @ %.2f (%s) pnl=%.2f (%.2f%%)",
                    pos.direction, pos.symbol, fill.fill_price, reason, realized, pnl_pct)

    def flatten_all(self, prices: dict[str, float], reason: str) -> None:
        for pos in list(self.open_positions.values()):
            price = prices.get(pos.symbol, pos.entry_price)
            self._close_position(pos, price, reason)
        self._recompute_daily(self.equity_value(prices))

    def end_of_day_flatten(self, prices: dict[str, float]) -> None:
        if self.open_positions:
            logger.info("End-of-day flatten of %d position(s)", len(self.open_positions))
        self.flatten_all(prices, "EOD")

    # ── kill switch ────────────────────────────────────────────
    def check_kill_switch(self, prices: dict[str, float]) -> bool:
        """Flatten and halt for the day if the drawdown limit is breached."""
        equity = self.equity_value(prices)
        self._recompute_daily(equity)
        if self.halted:
            return True
        if self.risk.is_drawdown_breached(self.starting_equity, equity):
            dd = self.risk.drawdown_pct(self.starting_equity, equity)
            reason = f"Daily drawdown {dd:.2f}% >= limit {self.settings.risk.max_daily_drawdown_pct}%"
            logger.critical("KILL SWITCH: %s — flattening and halting.", reason)
            self.flatten_all(prices, "KILL_SWITCH")
            self.halted = True
            self.metrics.set_halted(self.trade_date, reason)
            return True
        return False

    # ── metrics ────────────────────────────────────────────────
    def _recompute_daily(self, current_equity: float) -> None:
        closed = self.positions.get_closed_for_day(self.trade_date)
        wins = [p for p in closed if (p.realized_pnl or 0) > 0]
        losses = [p for p in closed if (p.realized_pnl or 0) < 0]
        gross_profit = sum(p.realized_pnl for p in wins) if wins else 0.0
        gross_loss = sum(p.realized_pnl for p in losses) if losses else 0.0
        realized = sum(p.realized_pnl or 0.0 for p in closed)
        total = len(closed)
        win_rate = (len(wins) / total * 100.0) if total else None
        # Undefined when there are no losses yet (avoid storing inf in SQLite).
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

        existing = self.metrics.get(self.trade_date)
        peak = max(existing.peak_equity or current_equity, current_equity) if existing else current_equity
        max_dd = self.risk.drawdown_pct(self.starting_equity, current_equity)
        prior_dd = (existing.max_drawdown_pct if existing else 0.0) or 0.0

        self.metrics.update(
            self.trade_date,
            ending_equity=current_equity,
            realized_pnl=realized,
            peak_equity=peak,
            max_drawdown_pct=max(prior_dd, max_dd),
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=profit_factor,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
        )
        self.equity.add_point(self.trade_date, current_equity)
