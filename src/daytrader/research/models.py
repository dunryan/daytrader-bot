"""Data structures used by the research engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TickerMetrics:
    """Pre-market metrics computed for a single ticker."""

    symbol: str
    last_close: float
    prev_close: float
    day_open: float
    gap_percent: float
    current_volume: float
    avg_daily_volume: float
    relative_volume: float
    atr_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WatchlistCandidate:
    """A ticker that passed the screen, enriched with sentiment + a reason."""

    symbol: str
    metrics: TickerMetrics
    sentiment_score: float | None = None
    score: float = 0.0
    rank: int | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)

    def to_watchlist_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "reason": self.reason_text,
            "gap_percent": round(self.metrics.gap_percent, 4),
            "relative_volume": round(self.metrics.relative_volume, 4),
            "avg_daily_volume": int(self.metrics.avg_daily_volume),
            "sentiment_score": self.sentiment_score,
            "rank": self.rank,
        }
