"""Typed CRUD repositories.

Every other module goes through these instead of touching the ORM/SQL
directly. Each repository takes a :class:`Database` and opens short-lived
transactional sessions per operation. More repositories (orders, fills,
positions, metrics) are added as the corresponding modules are built.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Sequence

from sqlalchemy import delete, select

from daytrader.persistence.database import Database
from daytrader.persistence.models import (
    DailyMetric,
    EquityPoint,
    Fill,
    Order,
    Position,
    Scan,
    Signal,
    WatchlistItem,
)
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


class WatchlistRepository:
    """CRUD for the per-day watchlist produced by Module 1."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def replace_for_day(self, trade_date: str, items: Sequence[dict[str, Any]]) -> int:
        """Atomically replace the watchlist for ``trade_date``.

        Pre-market research is idempotent: re-running it for the same day
        clears the previous list and inserts the fresh one. Returns the number
        of rows written.
        """
        with self.db.session() as s:
            s.execute(delete(WatchlistItem).where(WatchlistItem.trade_date == trade_date))
            rows = [
                WatchlistItem(
                    trade_date=trade_date,
                    symbol=item["symbol"],
                    reason=item.get("reason"),
                    gap_percent=item.get("gap_percent"),
                    relative_volume=item.get("relative_volume"),
                    avg_daily_volume=item.get("avg_daily_volume"),
                    sentiment_score=item.get("sentiment_score"),
                    rank=item.get("rank"),
                )
                for item in items
            ]
            s.add_all(rows)
            logger.info("Wrote %d watchlist items for %s", len(rows), trade_date)
            return len(rows)

    def get_for_day(self, trade_date: str) -> list[WatchlistItem]:
        with self.db.session() as s:
            stmt = (
                select(WatchlistItem)
                .where(WatchlistItem.trade_date == trade_date)
                .order_by(WatchlistItem.rank.asc().nulls_last())
            )
            return list(s.scalars(stmt).all())

    def symbols_for_day(self, trade_date: str) -> list[str]:
        return [item.symbol for item in self.get_for_day(trade_date)]


class ScanRepository:
    """Append-only audit log of raw screener snapshots."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, trade_date: str, symbol: str, metrics: dict[str, Any]) -> None:
        with self.db.session() as s:
            s.add(
                Scan(
                    trade_date=trade_date,
                    symbol=symbol,
                    metrics_json=json.dumps(metrics, default=str),
                )
            )

    def record_many(self, trade_date: str, scans: Sequence[tuple[str, dict[str, Any]]]) -> int:
        with self.db.session() as s:
            rows = [
                Scan(trade_date=trade_date, symbol=sym, metrics_json=json.dumps(m, default=str))
                for sym, m in scans
            ]
            s.add_all(rows)
            return len(rows)

    def get_for_day(self, trade_date: str) -> list[Scan]:
        with self.db.session() as s:
            stmt = select(Scan).where(Scan.trade_date == trade_date)
            return list(s.scalars(stmt).all())


class SignalRepository:
    """Append-only log of strategy signals produced by Module 3."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        trade_date: str,
        symbol: str,
        strategy: str,
        direction: str,
        price_at_signal: float | None = None,
        confidence: float | None = None,
        rationale: str | None = None,
        indicators: dict[str, Any] | None = None,
    ) -> int:
        """Persist a single signal; returns its row id."""
        with self.db.session() as s:
            row = Signal(
                trade_date=trade_date,
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                price_at_signal=price_at_signal,
                confidence=confidence,
                rationale=rationale,
                indicators_json=json.dumps(indicators or {}, default=str),
            )
            s.add(row)
            s.flush()
            return row.id

    def mark_acted_on(self, signal_id: int) -> None:
        with self.db.session() as s:
            row = s.get(Signal, signal_id)
            if row is not None:
                row.acted_on = True

    def get_for_day(self, trade_date: str, direction: str | None = None) -> list[Signal]:
        with self.db.session() as s:
            stmt = select(Signal).where(Signal.trade_date == trade_date)
            if direction is not None:
                stmt = stmt.where(Signal.direction == direction)
            stmt = stmt.order_by(Signal.created_at.asc())
            return list(s.scalars(stmt).all())


class OrderRepository:
    """Order intents and their fills (Module 4)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, **kwargs: Any) -> int:
        with self.db.session() as s:
            order = Order(**kwargs)
            s.add(order)
            s.flush()
            return order.id

    def add_fill(
        self,
        order_id: int,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        slippage: float = 0.0,
        commission: float = 0.0,
        status: str = "FILLED",
    ) -> int:
        with self.db.session() as s:
            fill = Fill(
                order_id=order_id,
                symbol=symbol,
                side=side,
                qty=qty,
                fill_price=fill_price,
                slippage=slippage,
                commission=commission,
            )
            s.add(fill)
            order = s.get(Order, order_id)
            if order is not None:
                order.status = status
            s.flush()
            return fill.id

    def get_for_day(self, trade_date: str) -> list[Order]:
        with self.db.session() as s:
            return list(s.scalars(select(Order).where(Order.trade_date == trade_date)).all())


class PositionRepository:
    """Open/closed round-trips. Open rows rehydrate AppState on restart."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def open_position(self, **kwargs: Any) -> int:
        with self.db.session() as s:
            pos = Position(status="OPEN", **kwargs)
            s.add(pos)
            s.flush()
            return pos.id

    def update_stops(self, position_id: int, stop_loss: float | None, take_profit: float | None) -> None:
        with self.db.session() as s:
            pos = s.get(Position, position_id)
            if pos is not None:
                if stop_loss is not None:
                    pos.stop_loss = stop_loss
                if take_profit is not None:
                    pos.take_profit = take_profit

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_time: dt.datetime,
        exit_reason: str,
        realized_pnl: float,
        pnl_percent: float,
    ) -> None:
        with self.db.session() as s:
            pos = s.get(Position, position_id)
            if pos is None:
                return
            pos.exit_price = exit_price
            pos.exit_time = exit_time
            pos.exit_reason = exit_reason
            pos.realized_pnl = realized_pnl
            pos.pnl_percent = pnl_percent
            pos.status = "CLOSED"

    def get_open(self) -> list[Position]:
        with self.db.session() as s:
            return list(s.scalars(select(Position).where(Position.status == "OPEN")).all())

    def get_for_day(self, trade_date: str) -> list[Position]:
        with self.db.session() as s:
            stmt = select(Position).where(Position.trade_date == trade_date)
            return list(s.scalars(stmt).all())

    def get_closed_for_day(self, trade_date: str) -> list[Position]:
        with self.db.session() as s:
            stmt = select(Position).where(
                Position.trade_date == trade_date, Position.status == "CLOSED"
            )
            return list(s.scalars(stmt).all())


class DailyMetricsRepository:
    """One row per trading day; backs the drawdown kill switch and reports."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_or_create(self, trade_date: str, starting_equity: float) -> DailyMetric:
        with self.db.session() as s:
            row = s.get(DailyMetric, trade_date)
            if row is None:
                row = DailyMetric(
                    trade_date=trade_date,
                    starting_equity=starting_equity,
                    peak_equity=starting_equity,
                    ending_equity=starting_equity,
                )
                s.add(row)
                s.flush()
            s.expunge(row)
            return row

    def get(self, trade_date: str) -> DailyMetric | None:
        with self.db.session() as s:
            row = s.get(DailyMetric, trade_date)
            if row is not None:
                s.expunge(row)
            return row

    def update(self, trade_date: str, **fields: Any) -> None:
        with self.db.session() as s:
            row = s.get(DailyMetric, trade_date)
            if row is None:
                return
            for key, value in fields.items():
                setattr(row, key, value)

    def set_halted(self, trade_date: str, reason: str) -> None:
        self.update(trade_date, trading_halted=True, halt_reason=reason)

    def mark_report_generated(self, trade_date: str) -> None:
        self.update(trade_date, report_generated=True)


class EquityRepository:
    """Intraday equity snapshots for drawdown tracking and report charts."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add_point(self, trade_date: str, equity: float, timestamp: dt.datetime | None = None) -> None:
        with self.db.session() as s:
            s.add(
                EquityPoint(
                    trade_date=trade_date,
                    equity=equity,
                    timestamp=timestamp or dt.datetime.now(dt.timezone.utc),
                )
            )

    def get_for_day(self, trade_date: str) -> list[EquityPoint]:
        with self.db.session() as s:
            stmt = (
                select(EquityPoint)
                .where(EquityPoint.trade_date == trade_date)
                .order_by(EquityPoint.timestamp.asc())
            )
            return list(s.scalars(stmt).all())
