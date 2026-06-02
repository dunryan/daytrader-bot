"""Performance metrics computed from a day's closed positions (pure)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from daytrader.persistence.models import Position


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float          # %
    gross_profit: float
    gross_loss: float        # negative
    net_pnl: float
    profit_factor: float | None
    avg_win: float
    avg_loss: float
    expectancy: float        # average $ per trade
    largest_win: float
    largest_loss: float
    return_pct: float        # net P/L vs starting equity

    @property
    def profit_factor_str(self) -> str:
        if self.profit_factor is None:
            return "n/a"
        return f"{self.profit_factor:.2f}"


def compute_performance(
    closed: Sequence[Position], starting_equity: float
) -> PerformanceMetrics:
    """Aggregate closed round-trips into a performance summary."""
    pnls = [float(p.realized_pnl or 0.0) for p in closed]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]

    gross_profit = sum(wins)
    gross_loss = sum(losses)
    net = sum(pnls)
    total = len(pnls)

    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None
    win_rate = (len(wins) / total * 100.0) if total else 0.0
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    expectancy = (net / total) if total else 0.0
    return_pct = (net / starting_equity * 100.0) if starting_equity else 0.0

    return PerformanceMetrics(
        total_trades=total,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_pnl=net,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        largest_win=max(wins) if wins else 0.0,
        largest_loss=min(losses) if losses else 0.0,
        return_pct=return_pct,
    )
