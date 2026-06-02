"""Application wiring + daily lifecycle stages.

The :class:`Application` is the dependency-injection container: it builds one
shared database, market-data provider, and the six module engines, then exposes
the lifecycle stage methods the scheduler calls (pre-market research, the
trading cycle, and market-close reporting). Each stage is wrapped so a failure
is logged but never crashes the long-running service.
"""

from __future__ import annotations

import datetime as dt

from daytrader.config.market_calendar import MarketClock
from daytrader.config.settings import Settings, get_settings
from daytrader.data.data_engine import DataEngine
from daytrader.data.providers import get_provider
from daytrader.data.providers.base import Timeframe
from daytrader.execution.execution_engine import ExecutionEngine
from daytrader.persistence.database import Database
from daytrader.reporting.report_engine import ReportEngine
from daytrader.research.research_engine import ResearchEngine
from daytrader.research.sentiment import get_sentiment_provider
from daytrader.strategy.router import StrategyRouter
from daytrader.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def _parse_timeframes(values: list[str]) -> list[Timeframe]:
    out: list[Timeframe] = []
    for v in values:
        try:
            out.append(Timeframe(v))
        except ValueError:
            logger.warning("Ignoring unknown timeframe %r", v)
    return out or [Timeframe.MIN_5, Timeframe.DAY]


class Application:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.clock = MarketClock(self.settings.app.timezone)
        self.db = Database(self.settings.db_url)

        provider = get_provider(self.settings)
        self.provider = provider
        self.data_engine = DataEngine(provider, self.settings.indicators)
        self.research = ResearchEngine(
            self.settings, provider, get_sentiment_provider(self.settings), self.db
        )
        self.router = StrategyRouter.from_settings(self.settings, self.db)
        self.execution = ExecutionEngine.from_settings(self.settings, self.db)
        self.report = ReportEngine(self.settings, self.db, self.data_engine)

        self.timeframes = _parse_timeframes(self.settings.data.timeframes)
        # Restrict per-cycle snapshots to the intraday + daily frames we need.
        self.cycle_timeframes = [
            tf for tf in self.timeframes if tf in (Timeframe.MIN_1, Timeframe.MIN_5, Timeframe.MIN_15, Timeframe.DAY)
        ] or [Timeframe.MIN_5, Timeframe.DAY]

    # ── lifecycle ──────────────────────────────────────────────
    def on_start(self) -> None:
        """Boot: rehydrate state for today (or resume a latched halt)."""
        mode = "SIMULATION" if self.settings.app.simulation_mode else "LIVE"
        logger.info("Starting daytrader-bot [%s] for %s", mode, self.clock.now().date())
        self.execution.start_day(self.clock.now().date())

    def premarket_research(self) -> list[str]:
        if not self.clock.is_trading_day():
            logger.info("Not a trading day; skipping pre-market research.")
            return []
        try:
            self.execution.start_day(self.clock.now().date())
            candidates = self.research.run(self.clock.now().date())
            return [c.symbol for c in candidates]
        except Exception:  # noqa: BLE001
            logger.exception("Pre-market research failed")
            return []

    def trading_cycle(self) -> None:
        """One scan/evaluate/execute/manage pass. Safe to call on an interval."""
        now = self.clock.now()
        if not self.clock.is_market_open(now):
            return
        try:
            symbols = self.research.watchlist_repo.symbols_for_day(now.date().isoformat())
            if not symbols:
                logger.debug("No watchlist symbols for %s", now.date())
                return

            snapshots = self.data_engine.build_snapshots(symbols, self.cycle_timeframes, as_of=now)
            prices = {
                sym: snap.latest_price()
                for sym, snap in snapshots.items()
                if snap.latest_price() is not None
            }

            # Account guard first: a kill-switch breach flattens and halts.
            if self.execution.check_kill_switch(prices):
                return

            signals = self.router.evaluate(snapshots, trade_date=now.date())
            self.execution.process_signals(signals, prices)
            self.execution.manage_positions(prices)
        except Exception:  # noqa: BLE001
            logger.exception("Trading cycle failed")

    def market_close(self) -> None:
        """Flatten everything for the day, then generate + email the report."""
        now = self.clock.now()
        try:
            symbols = self.research.watchlist_repo.symbols_for_day(now.date().isoformat())
            prices: dict[str, float] = {}
            if symbols:
                snapshots = self.data_engine.build_snapshots(symbols, self.cycle_timeframes, as_of=now)
                prices = {
                    s: snap.latest_price() for s, snap in snapshots.items() if snap.latest_price()
                }
            self.execution.end_of_day_flatten(prices)
        except Exception:  # noqa: BLE001
            logger.exception("End-of-day flatten failed")

        try:
            self.report.generate(now.date())
        except Exception:  # noqa: BLE001
            logger.exception("Report generation failed")


def build_application(settings: Settings | None = None) -> Application:
    settings = settings or get_settings()
    setup_logging(settings.app.log_level)
    return Application(settings)
