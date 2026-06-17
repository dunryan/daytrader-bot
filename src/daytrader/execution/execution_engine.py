"""ExecutionEngine — turns signals into risk-checked, tracked positions.

Responsibilities:
* Gate everything behind ``SIMULATION_MODE`` by choosing Virtual/Live broker.
* Rank actionable signals (confidence x risk/reward x meta-probability) and
  allocate capital best-first instead of arrival-order.
* Size each trade (fixed-fractional risk, optionally Kelly-scaled), set
  ATR/structural stops, and place broker-side protective orders in live mode.
* Persist orders, fills, and positions so state survives restarts.
* Manage open positions every cycle: trail stops (after +1R) using the ATR
  captured at entry, exit on stop/target, mirror trailing to the broker.
* Enforce the account-level **daily-drawdown kill switch** measured from the
  intraday peak: flatten and halt.

On startup :meth:`rehydrate` reloads open positions and the day's metrics from
the database (and reconciles against the broker's actual positions in live
mode), so a reboot never forgets open risk or a latched halt.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

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
    take_profit: float | None
    strategy: str | None
    atr_at_entry: float = 0.0
    initial_risk: float = 0.0
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
        entry_collar_bps=settings.risk.entry_collar_bps,
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
        #: Optional fallback quote source (set by the Application) used when
        #: flattening symbols whose price is missing from the cycle snapshot.
        self.quote_fn: Callable[[list[str]], dict[str, float]] | None = None

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

        # Anchor the day's equity to the REAL account in live mode; the
        # config seed is only meaningful for the virtual ledger.
        day_equity = self.settings.risk.starting_equity
        if not self.broker.is_simulated:
            try:
                day_equity = self.broker.get_equity()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Could not fetch live account equity; falling back to configured seed."
                )

        metric = self.metrics.get_or_create(self.trade_date, day_equity)
        self.starting_equity = metric.starting_equity
        self.halted = bool(metric.trading_halted)
        self.rehydrate()
        if self.halted:
            logger.warning("Resuming day %s with trading HALTED (%s)", self.trade_date, metric.halt_reason)

    def rehydrate(self) -> None:
        """Reload open positions from the DB and reconstruct virtual cash.

        In live mode the DB view is reconciled against the broker's actual
        positions: rows the broker no longer holds are closed out locally,
        and unknown broker positions are flagged loudly for manual action.
        """
        self.open_positions.clear()
        open_rows = self.positions.get_open()
        entry_notional = 0.0
        for row in open_rows:
            side = Side.BUY if row.direction == "LONG" else Side.SELL
            initial_risk = row.initial_risk or 0.0
            if initial_risk <= 0 and row.stop_loss:
                initial_risk = abs(row.entry_price - row.stop_loss)
            self.open_positions[row.symbol] = OpenPosition(
                id=row.id, symbol=row.symbol, side=side, qty=row.qty,
                entry_price=row.entry_price, entry_time=row.entry_time,
                stop_loss=row.stop_loss or 0.0, take_profit=row.take_profit,
                strategy=row.strategy,
                atr_at_entry=row.atr_at_entry or 0.0,
                initial_risk=initial_risk,
            )
            entry_notional += row.qty * row.entry_price * (1 if side is Side.BUY else -1)

        if isinstance(self.broker, VirtualBroker):
            realized_today = sum(
                p.realized_pnl or 0.0 for p in self.positions.get_closed_for_day(self.trade_date)
            )
            self.broker.set_cash(
                self.broker.starting_cash + realized_today - entry_notional
            )
        else:
            self._reconcile_with_broker()

        logger.info("Rehydrated %d open position(s) for %s", len(self.open_positions), self.trade_date)

    def _reconcile_with_broker(self) -> None:
        """Cross-check DB open positions against the broker's reality."""
        broker_positions = self.broker.list_positions()
        if broker_positions is None:
            logger.warning("Broker position reconciliation unavailable; trusting DB state.")
            return

        for symbol, pos in list(self.open_positions.items()):
            expected = pos.qty if pos.side is Side.BUY else -pos.qty
            actual = broker_positions.get(symbol, 0.0)
            if abs(actual - expected) > 1e-6:
                logger.critical(
                    "RECONCILE: DB says %s %s x%.2f but broker holds %.2f — "
                    "closing the DB row (exit likely filled while offline).",
                    pos.direction, symbol, pos.qty, actual,
                )
                # Best estimate of the offline exit: the resting stop level.
                est_exit = pos.stop_loss or pos.entry_price
                self._record_close(pos, est_exit, "RECONCILE")
                self.open_positions.pop(symbol, None)

        for symbol, qty in broker_positions.items():
            if symbol not in self.open_positions and abs(qty) > 1e-6:
                logger.critical(
                    "RECONCILE: broker holds %s x%.2f UNKNOWN to the DB — "
                    "manual intervention required (not auto-managed).",
                    symbol, qty,
                )

    # ── equity ─────────────────────────────────────────────────
    def equity_value(self, prices: dict[str, float]) -> float:
        eq = self.broker.get_cash()
        for pos in self.open_positions.values():
            price = prices.get(pos.symbol, pos.entry_price)
            eq += pos.qty * price * (1 if pos.side is Side.BUY else -1)
        return eq

    # ── opening trades ─────────────────────────────────────────
    @staticmethod
    def _signal_score(sig: Signal, default_rr: float) -> float:
        """Allocation rank: confidence x expected R:R x meta-probability."""
        rr = default_rr
        if sig.price and sig.stop_hint is not None and sig.target_hint is not None:
            risk = abs(sig.price - sig.stop_hint)
            reward = abs(sig.target_hint - sig.price)
            if risk > 0:
                rr = reward / risk
        meta = sig.indicators.get("meta_prob")
        meta = float(meta) if meta is not None else 1.0
        return (sig.confidence or 0.0) * rr * meta

    def process_signals(self, signals: list[Signal], prices: dict[str, float]) -> list[OpenPosition]:
        opened: list[OpenPosition] = []
        if self.halted:
            logger.info("Trading halted; ignoring %d signal(s).", len(signals))
            return opened

        default_rr = self.settings.risk.take_profit.risk_reward_ratio
        ranked = sorted(
            (s for s in signals if s.is_actionable),
            key=lambda s: self._signal_score(s, default_rr),
            reverse=True,
        )

        equity = self.equity_value(prices)
        for sig in ranked:
            if sig.symbol in self.open_positions:
                continue  # one position per symbol
            if not self.risk.can_open_new(len(self.open_positions)):
                logger.info("Max open positions reached; skipping %s", sig.symbol)
                break  # list is sorted: everything after ranks lower

            pos = self._open_from_signal(sig, prices, equity)
            if pos is not None:
                opened.append(pos)
                equity = self.equity_value(prices)
        return opened

    def _risk_pct_for(self, strategy: str | None) -> float | None:
        """Kelly-scaled risk %% for a strategy, or None for the fixed default."""
        kelly = self.settings.risk.kelly
        if not kelly.enabled or not strategy:
            return None
        rows = self.positions.get_closed_for_strategy(strategy, limit=kelly.lookback_trades)
        if len(rows) < kelly.min_trades:
            return None
        wins = [p.realized_pnl for p in rows if (p.realized_pnl or 0) > 0]
        losses = [p.realized_pnl for p in rows if (p.realized_pnl or 0) < 0]
        if not wins or not losses:
            return None
        win_rate = len(wins) / len(rows)
        payoff = (sum(wins) / len(wins)) / abs(sum(losses) / len(losses))
        pct = self.risk.kelly_risk_pct(win_rate, payoff)
        if pct <= 0:
            logger.info(
                "Kelly sizing suppressed %s: measured edge non-positive "
                "(win_rate=%.2f payoff=%.2f over %d trades)",
                strategy, win_rate, payoff, len(rows),
            )
        return pct

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

        # Trailing and fixed take-profit are mutually exclusive: a fixed cap
        # under a trailing regime truncates the right tail the trail exists
        # to capture.
        if self.settings.risk.take_profit.method == "trailing":
            take_profit: float | None = None
        else:
            take_profit = sig.target_hint or self.risk.take_profit(side, entry, stop)

        sizing = self.risk.position_size(
            equity, entry, stop,
            available_cash=self.broker.get_cash(),
            risk_pct=self._risk_pct_for(sig.strategy),
        )
        if sizing.qty <= 0:
            logger.info("Sizing yielded 0 shares for %s (capped_by=%s); skipping", sig.symbol, sizing.capped_by)
            return None

        fill = self.broker.submit_bracket(sig.symbol, side, sizing.qty, entry, stop, take_profit)
        if fill is None:
            logger.info("Entry for %s not filled; skipping.", sig.symbol)
            return None

        initial_risk = abs(fill.fill_price - stop)
        order_id = self.orders.create(
            signal_id=None, trade_date=self.trade_date, symbol=sig.symbol,
            side=side.value, order_type="BRACKET", qty=sizing.qty,
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
            atr_at_entry=atr, initial_risk=initial_risk,
        )
        pos = OpenPosition(
            id=pos_id, symbol=sig.symbol, side=side, qty=sizing.qty,
            entry_price=fill.fill_price, entry_time=fill.timestamp,
            stop_loss=stop, take_profit=take_profit, strategy=sig.strategy,
            atr_at_entry=atr, initial_risk=initial_risk,
        )
        self.open_positions[sig.symbol] = pos
        logger.info("OPENED %s %s x%d @ %.2f stop=%.2f tp=%s (%s)",
                    pos.direction, sig.symbol, sizing.qty, fill.fill_price, stop,
                    f"{take_profit:.2f}" if take_profit else "trail", sig.strategy)
        self._recompute_daily(self.equity_value(prices))
        return pos

    # ── managing open trades ───────────────────────────────────
    def manage_positions(self, prices: dict[str, float]) -> list[str]:
        """Trail stops and exit positions that hit stop/target. Returns closed symbols."""
        closed: list[str] = []
        closed.extend(self._sync_broker_exits(prices))

        for symbol, pos in list(self.open_positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            # Volatility captured at entry; legacy rows fall back to deriving
            # it from the (ATR-method) stop distance.
            atr = pos.atr_at_entry
            if atr <= 0:
                atr = abs(pos.entry_price - pos.stop_loss) / max(
                    self.settings.risk.stop_loss.atr_multiplier, 1e-9
                )

            # Update peak and trailing stop (activates after +1R).
            if pos.side is Side.BUY:
                pos.peak_price = max(pos.peak_price, price)
            else:
                pos.peak_price = min(pos.peak_price, price)
            new_stop = self.risk.update_trailing_stop(
                pos.side, pos.stop_loss, pos.peak_price, atr,
                entry=pos.entry_price, initial_risk=pos.initial_risk,
            )
            if new_stop != pos.stop_loss:
                pos.stop_loss = new_stop
                self.positions.update_stops(pos.id, new_stop, None)
                if not self.broker.is_simulated:
                    if not self.broker.replace_stop(symbol, new_stop):
                        logger.warning("Could not move broker stop for %s to %.2f", symbol, new_stop)

            reason = self._exit_reason(pos, price)
            if reason:
                self._close_position(pos, price, reason)
                closed.append(symbol)
        if closed:
            self._recompute_daily(self.equity_value(prices))
        return closed

    def _sync_broker_exits(self, prices: dict[str, float]) -> list[str]:
        """Detect positions the broker already exited (resting stop/TP filled)."""
        if self.broker.is_simulated or not self.open_positions:
            return []
        broker_positions = self.broker.list_positions()
        if broker_positions is None:
            return []
        closed: list[str] = []
        for symbol, pos in list(self.open_positions.items()):
            if abs(broker_positions.get(symbol, 0.0)) < 1e-6:
                est_exit = prices.get(symbol) or pos.stop_loss or pos.entry_price
                logger.info("Broker-side exit detected for %s; recording close.", symbol)
                self._record_close(pos, est_exit, "BROKER_EXIT")
                self.open_positions.pop(symbol, None)
                closed.append(symbol)
        return closed

    @staticmethod
    def _exit_reason(pos: OpenPosition, price: float) -> str | None:
        if pos.side is Side.BUY:
            if price <= pos.stop_loss:
                return "SL"
            if pos.take_profit is not None and price >= pos.take_profit:
                return "TP"
        else:
            if price >= pos.stop_loss:
                return "SL"
            if pos.take_profit is not None and price <= pos.take_profit:
                return "TP"
        return None

    def _record_close(self, pos: OpenPosition, exit_price: float, reason: str) -> None:
        """Persist a close that already happened (no new broker order)."""
        if pos.side is Side.BUY:
            realized = (exit_price - pos.entry_price) * pos.qty
        else:
            realized = (pos.entry_price - exit_price) * pos.qty
        pnl_pct = realized / (pos.entry_price * pos.qty) * 100.0 if pos.entry_price else 0.0
        self.positions.close_position(
            pos.id, exit_price=exit_price, exit_time=dt.datetime.now(dt.timezone.utc),
            exit_reason=reason, realized_pnl=realized, pnl_percent=pnl_pct,
        )

    def _close_position(self, pos: OpenPosition, price: float, reason: str) -> None:
        exit_side = pos.side.opposite
        if not self.broker.is_simulated:
            # Lift the resting bracket legs first or the exit double-fills.
            try:
                self.broker.cancel_symbol_orders(pos.symbol)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to cancel resting orders for %s", pos.symbol)
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
        prices = dict(prices)
        missing = [s for s in self.open_positions if s not in prices]
        if missing and self.quote_fn is not None:
            try:
                prices.update(self.quote_fn(missing))
            except Exception:  # noqa: BLE001
                logger.exception("Quote fallback failed for %s", missing)
        for pos in list(self.open_positions.values()):
            price = prices.get(pos.symbol)
            if price is None:
                logger.critical(
                    "No price available to flatten %s; using entry price %.2f — "
                    "recorded P&L for this exit is unreliable.",
                    pos.symbol, pos.entry_price,
                )
                price = pos.entry_price
            self._close_position(pos, price, reason)
        self._recompute_daily(self.equity_value(prices))

    def end_of_day_flatten(self, prices: dict[str, float]) -> None:
        if self.open_positions:
            logger.info("End-of-day flatten of %d position(s)", len(self.open_positions))
        self.flatten_all(prices, "EOD")

    # ── kill switch ────────────────────────────────────────────
    def check_kill_switch(self, prices: dict[str, float]) -> bool:
        """Flatten and halt for the day if the peak-to-trough drawdown limit is breached."""
        equity = self.equity_value(prices)
        self._recompute_daily(equity)
        if self.halted:
            return True
        peak = self._peak_equity(equity)
        if self.risk.is_drawdown_breached(peak, equity):
            dd = self.risk.drawdown_pct(peak, equity)
            reason = (
                f"Drawdown {dd:.2f}% from intraday peak {peak:.2f} >= "
                f"limit {self.settings.risk.max_daily_drawdown_pct}%"
            )
            logger.critical("KILL SWITCH: %s — flattening and halting.", reason)
            self.flatten_all(prices, "KILL_SWITCH")
            self.halted = True
            self.metrics.set_halted(self.trade_date, reason)
            return True
        return False

    def _peak_equity(self, current_equity: float) -> float:
        existing = self.metrics.get(self.trade_date)
        persisted_peak = (existing.peak_equity if existing else None) or 0.0
        return max(persisted_peak, self.starting_equity, current_equity)

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

        peak = self._peak_equity(current_equity)
        max_dd = self.risk.drawdown_pct(peak, current_equity)
        existing = self.metrics.get(self.trade_date)
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
