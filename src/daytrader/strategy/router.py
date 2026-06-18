"""Strategy & Watchlist Router (Module 3).

Builds the set of *enabled* strategies from config, runs them across the
snapshots of watchlist symbols (gated by the market-regime classifier),
applies the meta-label quality filter, and persists every actionable signal
(with full rationale, indicator metadata, and meta-probability) to the
``signals`` table for the audit trail and for Module 4 to consume.

Pipeline per snapshot:  regime gate -> strategies -> meta filter.
Blocked signals are still persisted (``acted_on`` stays False) so filter
performance can be evaluated offline.
"""

from __future__ import annotations

import datetime as dt

from daytrader.config.settings import RegimeFilterConfig, Settings, get_settings
from daytrader.data.data_engine import MarketSnapshot
from daytrader.ml.meta_label import SignalFilter
from daytrader.persistence.database import Database
from daytrader.persistence.repositories import SignalRepository
from daytrader.strategy.base import Signal, Strategy
from daytrader.strategy.momentum_scalp import MomentumScalpStrategy
from daytrader.strategy.orb import OpeningRangeBreakoutStrategy
from daytrader.strategy.regime import classify
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
                entry_mode=str(getattr(t, "entry_mode", "breakout")),
                retest_max_bars=int(getattr(t, "retest_max_bars", 12)),
                touch_tolerance_atr=float(getattr(t, "touch_tolerance_atr", 0.25)),
                require_gap_direction_match=bool(getattr(t, "require_gap_direction_match", False)),
                max_gap_norm=float(getattr(t, "max_gap_norm", 0.0)),
                max_entry_minutes_after_open=int(getattr(t, "max_entry_minutes_after_open", 0)),
                min_breakout_rvol=float(getattr(t, "min_breakout_rvol", 0.0)),
                min_or_width_pct=float(getattr(t, "min_or_width_pct", 0.0)),
                max_or_width_atr=float(getattr(t, "max_or_width_atr", 0.0)),
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
        regime_config: RegimeFilterConfig | None = None,
        signal_filter: SignalFilter | None = None,
    ) -> None:
        self.strategies = strategies
        self.signal_repo = signal_repo
        self.regime_config = regime_config or RegimeFilterConfig()
        self.signal_filter = signal_filter

    @classmethod
    def from_settings(cls, settings: Settings | None = None, db: Database | None = None) -> "StrategyRouter":
        settings = settings or get_settings()
        db = db or Database(settings.db_url)
        return cls(
            build_strategies(settings),
            SignalRepository(db),
            regime_config=settings.strategies.regime_filter,
            signal_filter=SignalFilter.from_settings(settings),
        )

    # ── regime gating ──────────────────────────────────────────
    def _regime_allows(self, regime: str | None, strategy_name: str) -> bool:
        """Whether ``strategy_name`` is permitted in ``regime`` (None = no gating)."""
        if regime is None:
            return True
        allowed = self.regime_config.allowed.get(strategy_name)
        if allowed is None:
            return True  # unmapped strategies are never gated
        return regime in allowed

    def evaluate_snapshot(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Run every enabled strategy on one snapshot; return actionable signals."""
        regime: str | None = None
        regime_details: dict[str, float] = {}
        if self.regime_config.mode != "off":
            try:
                regime_enum, regime_details = classify(snapshot)
                regime = regime_enum.value
            except Exception:  # noqa: BLE001
                logger.exception("Regime classification failed for %s", snapshot.symbol)

        signals: list[Signal] = []
        for strat in self.strategies:
            allowed = self._regime_allows(regime, strat.name)
            if not allowed and self.regime_config.mode == "enforce":
                logger.debug(
                    "%s/%s blocked by regime gate (%s)", snapshot.symbol, strat.name, regime
                )
                continue
            try:
                sig = strat.evaluate(snapshot)
            except Exception:  # noqa: BLE001 - one strategy must not kill the loop
                logger.exception("Strategy %s failed on %s", strat.name, snapshot.symbol)
                continue
            if not sig.is_actionable:
                logger.debug("%s/%s HOLD: %s", snapshot.symbol, strat.name, sig.rationale)
                continue
            if regime is not None:
                sig.indicators["regime"] = regime
                sig.indicators.update(regime_details)
                if not allowed:  # shadow mode: annotate + log, let through
                    sig.indicators["regime_block"] = True
                    logger.info(
                        "REGIME SHADOW: would block %s %s/%s (regime=%s)",
                        sig.direction.value, snapshot.symbol, strat.name, regime,
                    )
            signals.append(sig)
        return signals

    def evaluate(
        self,
        snapshots: dict[str, MarketSnapshot],
        trade_date: dt.date | None = None,
        persist: bool = True,
    ) -> list[Signal]:
        """Evaluate all snapshots; persist and return the surviving signals."""
        trade_date = trade_date or dt.date.today()
        date_str = trade_date.isoformat()
        survivors: list[Signal] = []

        for symbol, snap in snapshots.items():
            for sig in self.evaluate_snapshot(snap):
                passed = self.signal_filter.passes(sig) if self.signal_filter else True
                logger.info(
                    "SIGNAL %s %s %s | %s%s",
                    sig.direction.value, symbol, sig.strategy, sig.rationale,
                    "" if passed else " [META-BLOCKED]",
                )
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
                        meta_prob=sig.indicators.get("meta_prob"),
                    )
                if passed:
                    survivors.append(sig)

        logger.info("Router produced %d actionable signals for %s", len(survivors), date_str)
        return survivors
