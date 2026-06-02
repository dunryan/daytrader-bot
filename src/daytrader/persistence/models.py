"""SQLAlchemy ORM models (Module 5 schema).

These map 1:1 to the schema documented in the README. Open ``Position`` rows
and the current day's ``DailyMetric`` are what rehydrate ``AppState`` on
restart, giving the "never forget open positions" guarantee.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ════════════════════════════════════════════════════════════
#  WATCHLIST — output of Module 1, per trading day
# ════════════════════════════════════════════════════════════
class WatchlistItem(Base):
    __tablename__ = "watchlist"
    __table_args__ = (UniqueConstraint("trade_date", "symbol", name="uq_watchlist_date_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    gap_percent: Mapped[float | None] = mapped_column(Float)
    relative_volume: Mapped[float | None] = mapped_column(Float)
    avg_daily_volume: Mapped[int | None] = mapped_column(Integer)
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    rank: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<WatchlistItem {self.trade_date} {self.symbol} rank={self.rank}>"


# ════════════════════════════════════════════════════════════
#  SCANS — raw screener snapshots (audit / replay)
# ════════════════════════════════════════════════════════════
class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    scanned_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


# ════════════════════════════════════════════════════════════
#  SIGNALS — output of Module 3
# ════════════════════════════════════════════════════════════
class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    strategy: Mapped[str] = mapped_column(String(40), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY/SELL/HOLD
    price_at_signal: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text)
    indicators_json: Mapped[str | None] = mapped_column(Text)
    acted_on: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    orders: Mapped[list["Order"]] = relationship(back_populates="signal")


# ════════════════════════════════════════════════════════════
#  ORDERS — intent submitted to broker (sim or live)
# ════════════════════════════════════════════════════════════
class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"))
    trade_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY/SELL
    order_type: Mapped[str] = mapped_column(String(12), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    limit_price: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)
    risk_amount: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="NEW")
    broker_order_id: Mapped[str | None] = mapped_column(String(64))
    is_simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    signal: Mapped["Signal | None"] = relationship(back_populates="orders")
    fills: Mapped[list["Fill"]] = relationship(back_populates="order")


# ════════════════════════════════════════════════════════════
#  FILLS — actual executions (entry & exit legs)
# ════════════════════════════════════════════════════════════
class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    slippage: Mapped[float | None] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    filled_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    order: Mapped["Order"] = relationship(back_populates="fills")


# ════════════════════════════════════════════════════════════
#  POSITIONS — open & closed round-trips (rehydrate on restart)
# ════════════════════════════════════════════════════════════
class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    strategy: Mapped[str | None] = mapped_column(String(40))
    direction: Mapped[str] = mapped_column(String(8), nullable=False)  # LONG/SHORT
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_time: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float)
    exit_time: Mapped[dt.datetime | None] = mapped_column(DateTime)
    exit_reason: Mapped[str | None] = mapped_column(String(16))
    realized_pnl: Mapped[float | None] = mapped_column(Float)
    pnl_percent: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(8), nullable=False, default="OPEN", index=True)
    is_simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


# ════════════════════════════════════════════════════════════
#  DAILY_METRICS — one row per trading day (drawdown guard + report)
# ════════════════════════════════════════════════════════════
class DailyMetric(Base):
    __tablename__ = "daily_metrics"

    trade_date: Mapped[str] = mapped_column(String(10), primary_key=True)
    starting_equity: Mapped[float] = mapped_column(Float, nullable=False)
    ending_equity: Mapped[float | None] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    peak_equity: Mapped[float | None] = mapped_column(Float)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float | None] = mapped_column(Float)
    profit_factor: Mapped[float | None] = mapped_column(Float)
    gross_profit: Mapped[float] = mapped_column(Float, default=0.0)
    gross_loss: Mapped[float] = mapped_column(Float, default=0.0)
    trading_halted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    halt_reason: Mapped[str | None] = mapped_column(Text)
    report_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


# ════════════════════════════════════════════════════════════
#  EQUITY_CURVE — periodic equity snapshots
# ════════════════════════════════════════════════════════════
class EquityPoint(Base):
    __tablename__ = "equity_curve"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
