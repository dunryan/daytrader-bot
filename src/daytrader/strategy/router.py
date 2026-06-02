"""Strategy & Watchlist Router (Module 3).

Builds the set of *enabled* strategies from config, runs them across the
snapshots of watchlist symbols, and persists the actionable signals (with full
rationale + indicator metadata) to the ``signals`` table for the audit trail
and for Module 4 to consume.
"""

from __future__ import annotations

import datetime as dt

from daytrader.config.settings import Settings, get_settings
from daytrader.data.data_engine import MarketSnapshot
from daytrader.persistence.database import Database
from daytrader.persistence.repositories import SignalRepository
from daytrader.strategy.base import Signal, Strategy
from daytrader.strategy.momentum_scalp import MomentumScalpStrategy
from daytrader.strategy.orb import OpeningRangeBreakoutStrategy
from daytrader.strategy.vwap_pullback import VwapPullbackStrategy
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_strategies(settings: Settings) -> list[Strategy]:
    """Instantiate strategies whose config toggle is enabled."""
    cfg = settings.strategies
    strategies: list[Strategy] = []

    if cfg.vwap_pullback.enabled:
        t = cfg.vwap_pullback
        strategies.append(
            VwapPullbackStrategy(
                trend_ema=int(getattr(t, "trend_ema", 21)),
                max_distance_from_vwap_atr=float(getattr(t, "max_distance_from_vwap_atr", 0.5)),
            )
        )
    if cfg.opening_range_breakout.enabled:
        t = cfg.opening_range_breakout
        strategies.append(
            OpeningRangeBreakoutStrategy(
                opening_range_minutes=int(getattr(t, "opening_range_minutes", 15)),
                volume_confirmation=bool(getattr(t, "volume_confirmation", True)),
            )
        )
    if cfg.momentum_scalp.enabled:
        t = cfg.momentum_scalp
        strategies.append(
            MomentumScalpStrategy(
                fast_ema=int(getattr(t, "fast_ema", 9)),
                slow_ema=int(getattr(t, "slow_ema", 21)),
                min_relative_volume=float(getattr(t, "min_relative_volume", 2.0)),
            )
        )

    logger.info("Enabled strategies: %s", [s.name for s in strategies] or "none")
    return strategies


class StrategyRouter:
    """Routes snapshots through enabled strategies and persists signals."""

    def __init__(
        self,
        strategies: list[Strategy],
        signal_repo: SignalRepository | None = None,
    ) -> None:
        self.strategies = strategies
        self.signal_repo = signal_repo

    @classmethod
    def from_settings(cls, settings: Settings | None = None, db: Database | None = None) -> "StrategyRouter":
        settings = settings or get_settings()
        db = db or Database(settings.db_url)
        return cls(build_strategies(settings), SignalRepository(db))

    def evaluate_snapshot(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Run every enabled strategy on one snapshot; return actionable signals."""
        signals: list[Signal] = []
        for strat in self.strategies:
            try:
                sig = strat.evaluate(snapshot)
            except Exception:  # noqa: BLE001 - one strategy must not kill the loop
                logger.exception("Strategy %s failed on %s", strat.name, snapshot.symbol)
                continue
            if sig.is_actionable:
                signals.append(sig)
            else:
                logger.debug("%s/%s HOLD: %s", snapshot.symbol, strat.name, sig.rationale)
        return signals

    def evaluate(
        self,
        snapshots: dict[str, MarketSnapshot],
        trade_date: dt.date | None = None,
        persist: bool = True,
    ) -> list[Signal]:
        """Evaluate all snapshots; persist and return the actionable signals."""
        trade_date = trade_date or dt.date.today()
        date_str = trade_date.isoformat()
        all_signals: list[Signal] = []

        for symbol, snap in snapshots.items():
            for sig in self.evaluate_snapshot(snap):
                all_signals.append(sig)
                logger.info("SIGNAL %s %s %s | %s", sig.direction.value, symbol, sig.strategy, sig.rationale)
                if persist and self.signal_repo is not None:
                    self.signal_repo.record(
                        trade_date=date_str,
                        symbol=sig.symbol,
                        strategy=sig.strategy,
                        direction=sig.direction.value,
                        price_at_signal=sig.price,
                        confidence=sig.confidence,
                        rationale=sig.rationale,
                        indicators=sig.indicators,
                    )

        logger.info("Router produced %d actionable signals for %s", len(all_signals), date_str)
        return all_signals
