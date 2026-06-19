"""Module 1 orchestrator: build the day's watchlist before the open.

Pipeline:
    resolve universe -> screen (gap/RVOL/liquidity) -> record scan audit
    -> enrich survivors with news sentiment -> score & rank
    -> truncate to max size -> persist to the ``watchlist`` table.

Runnable standalone for ad-hoc research::

    python -m daytrader.research.research_engine
"""

from __future__ import annotations

import datetime as dt

from daytrader.config.settings import Settings, get_settings
from daytrader.data.providers import get_provider
from daytrader.data.providers.base import MarketDataProvider
from daytrader.persistence.database import Database
from daytrader.persistence.repositories import ScanRepository, WatchlistRepository
from daytrader.research.models import TickerMetrics, WatchlistCandidate
from daytrader.research.screener import Screener
from daytrader.research.sentiment import SentimentProvider, get_sentiment_provider
from daytrader.research.universe import resolve_universe
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


class ResearchEngine:
    """Coordinates screener + sentiment into a persisted daily watchlist."""

    def __init__(
        self,
        settings: Settings,
        provider: MarketDataProvider,
        sentiment: SentimentProvider,
        db: Database,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.sentiment = sentiment
        self.watchlist_repo = WatchlistRepository(db)
        self.scan_repo = ScanRepository(db)
        self.screener = Screener(
            provider,
            settings.research.filters,
            premarket_cutoff=settings.schedule.premarket_research,
        )

    # ── public API ─────────────────────────────────────────────
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ResearchEngine":
        settings = settings or get_settings()
        provider = get_provider(settings)
        sentiment = get_sentiment_provider(settings)
        db = Database(settings.db_url)
        return cls(settings, provider, sentiment, db)

    def run(self, trade_date: dt.date | None = None) -> list[WatchlistCandidate]:
        """Run the full research pipeline and persist the watchlist."""
        trade_date = trade_date or dt.date.today()
        date_str = trade_date.isoformat()
        logger.info("─── Research run for %s ───", date_str)

        if not self.provider.is_available():
            logger.error(
                "Market-data provider unavailable (missing Alpaca credentials?). "
                "Aborting research run; no watchlist written."
            )
            return []

        symbols = resolve_universe(self.settings.research.universe)
        passed, all_metrics = self.screener.screen(symbols)

        self._record_scans(date_str, all_metrics)

        candidates = [WatchlistCandidate(symbol=m.symbol, metrics=m) for m in passed]
        self._enrich_sentiment(candidates)
        self._score_and_rank(candidates)

        top = candidates[: self.settings.research.max_watchlist_size]
        for rank, cand in enumerate(top, start=1):
            cand.rank = rank

        self.watchlist_repo.replace_for_day(date_str, [c.to_watchlist_row() for c in top])
        self._log_summary(top)
        return top

    # ── steps ──────────────────────────────────────────────────
    def _record_scans(self, date_str: str, metrics: list[TickerMetrics]) -> None:
        if not metrics:
            return
        self.scan_repo.record_many(date_str, [(m.symbol, m.to_dict()) for m in metrics])

    def _enrich_sentiment(self, candidates: list[WatchlistCandidate]) -> None:
        cfg = self.settings.research.sentiment
        if not cfg.enabled:
            return
        for cand in candidates:
            result = self.sentiment.score_symbol(cand.symbol, cfg.lookback_hours, cfg.min_articles)
            cand.sentiment_score = result.score
            if result.score is not None:
                cand.reasons.append(f"news {result.label} ({result.article_count} art.)")

    def _score_and_rank(self, candidates: list[WatchlistCandidate]) -> None:
        """Composite score: RVOL + gap magnitude + sentiment tie-breaker."""
        for cand in candidates:
            m = cand.metrics
            sentiment = cand.sentiment_score or 0.0
            cand.score = m.relative_volume + abs(m.gap_percent) / 10.0 + 0.5 * sentiment
            # Prepend the screener's qualifying reasons.
            from daytrader.research.screener import passes_filters

            _, reasons = passes_filters(m, self.settings.research.filters)
            cand.reasons = reasons + cand.reasons
        candidates.sort(key=lambda c: c.score, reverse=True)

    def _log_summary(self, top: list[WatchlistCandidate]) -> None:
        if not top:
            logger.info("Watchlist empty — no symbols passed the screen today.")
            return
        logger.info("Watchlist (%d):", len(top))
        for c in top:
            logger.info("  #%-2d %-6s | %s", c.rank, c.symbol, c.reason_text)


def main() -> None:  # pragma: no cover - manual entrypoint
    settings = get_settings()
    from daytrader.utils.logging_setup import setup_logging

    setup_logging(settings.app.log_level)
    engine = ResearchEngine.from_settings(settings)
    engine.run()


if __name__ == "__main__":  # pragma: no cover
    main()
