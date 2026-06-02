"""Module 6 orchestrator: build and email the end-of-day report.

Pulls the day's positions, watchlist, signals, and metrics from the database,
renders annotated charts (entry/stop/exit) for every traded symbol, compiles a
ReportLab PDF, and emails it. Designed to be called by the scheduler at the
configured ``report_time`` but is also runnable standalone for any date.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from daytrader.config.settings import PROJECT_ROOT, Settings, get_settings
from daytrader.data.data_engine import DataEngine
from daytrader.data.providers import get_provider
from daytrader.data.providers.base import Timeframe
from daytrader.persistence.database import Database
from daytrader.persistence.models import Position
from daytrader.persistence.repositories import (
    DailyMetricsRepository,
    EquityRepository,
    PositionRepository,
    SignalRepository,
    WatchlistRepository,
)
from daytrader.reporting import plotter
from daytrader.reporting.emailer import EmailSender
from daytrader.reporting.metrics import compute_performance
from daytrader.reporting.pdf_builder import TradeNarrative, build_report
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


class ReportEngine:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        data_engine: DataEngine | None = None,
        emailer: EmailSender | None = None,
    ) -> None:
        self.settings = settings
        self.positions = PositionRepository(db)
        self.watchlist = WatchlistRepository(db)
        self.signals = SignalRepository(db)
        self.metrics = DailyMetricsRepository(db)
        self.equity = EquityRepository(db)
        self.data_engine = data_engine
        self.emailer = emailer or EmailSender(settings.secrets)

    @classmethod
    def from_settings(cls, settings: Settings | None = None, db: Database | None = None) -> "ReportEngine":
        settings = settings or get_settings()
        db = db or Database(settings.db_url)
        data_engine = None
        provider = get_provider(settings)
        if provider.is_available():
            data_engine = DataEngine(provider, settings.indicators)
        return cls(settings, db, data_engine)

    # ── output paths ───────────────────────────────────────────
    def _output_dir(self, date_str: str) -> Path:
        base = Path(self.settings.reporting.output_dir)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        out = base / date_str
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ── narrative ──────────────────────────────────────────────
    def _build_narrative(
        self, pos: Position, watch_reason: str | None, signal_rationales: list[str]
    ) -> str:
        parts: list[str] = []
        if watch_reason:
            parts.append(f"Selected pre-market because: {watch_reason}.")
        if signal_rationales:
            parts.append("Signal(s): " + " | ".join(signal_rationales) + ".")
        entry = f"{pos.entry_price:.2f}" if pos.entry_price else "n/a"
        exit_ = f"{pos.exit_price:.2f}" if pos.exit_price else "open"
        parts.append(
            f"Entered {pos.direction} {int(pos.qty)} share(s) @ {entry}; "
            f"exited @ {exit_} ({pos.exit_reason or 'n/a'}) for "
            f"${pos.realized_pnl or 0:,.2f} ({pos.pnl_percent or 0:.2f}%)."
        )
        return " ".join(parts)

    # ── main ───────────────────────────────────────────────────
    def generate(self, trade_date: dt.date | None = None, send_email: bool | None = None) -> Path:
        trade_date = trade_date or dt.date.today()
        date_str = trade_date.isoformat()
        send_email = self.settings.reporting.send_email if send_email is None else send_email
        out_dir = self._output_dir(date_str)
        logger.info("Generating report for %s", date_str)

        closed = self.positions.get_closed_for_day(date_str)
        watchlist_items = self.watchlist.get_for_day(date_str)
        watch_reason = {w.symbol: w.reason for w in watchlist_items}
        watchlist_rows = [
            {"rank": w.rank, "symbol": w.symbol, "reason": w.reason} for w in watchlist_items
        ]

        signals = self.signals.get_for_day(date_str)
        rationales: dict[str, list[str]] = {}
        for sig in signals:
            rationales.setdefault(sig.symbol, []).append(f"{sig.strategy}: {sig.rationale}")

        metric = self.metrics.get(date_str)
        starting = metric.starting_equity if metric else self.settings.risk.starting_equity
        ending = (metric.ending_equity if metric and metric.ending_equity is not None else starting)
        perf = compute_performance(closed, starting)

        # Equity curve chart.
        equity_chart = plotter.plot_equity_curve(
            self.equity.get_for_day(date_str), out_dir / "equity_curve.png"
        )

        # Per-trade charts + narrative.
        narratives: list[TradeNarrative] = []
        for pos in closed:
            chart_path = self._chart_for(pos, trade_date, out_dir)
            narratives.append(
                TradeNarrative(
                    symbol=pos.symbol,
                    text=self._build_narrative(pos, watch_reason.get(pos.symbol), rationales.get(pos.symbol, [])),
                    chart_path=chart_path,
                )
            )

        pdf_path = build_report(
            out_dir / f"daily_report_{date_str}.pdf",
            trade_date=date_str,
            perf=perf,
            starting_equity=starting,
            ending_equity=ending,
            watchlist_rows=watchlist_rows,
            narratives=narratives,
            equity_chart=equity_chart,
            halted=bool(metric.trading_halted) if metric else False,
            halt_reason=metric.halt_reason if metric else None,
        )

        if send_email:
            self._send(date_str, perf, pdf_path, narratives)
        self.metrics.mark_report_generated(date_str)
        return pdf_path

    def _chart_for(self, pos: Position, trade_date: dt.date, out_dir: Path) -> Path | None:
        if self.data_engine is None:
            return None
        try:
            as_of = dt.datetime.combine(trade_date, dt.time(23, 59), tzinfo=dt.timezone.utc)
            frames = self.data_engine.fetch_enriched([pos.symbol], Timeframe.MIN_5, as_of=as_of)
            df = frames.get(pos.symbol)
            if df is None:
                return None
            return plotter.plot_trade(pos, df, out_dir / f"{pos.symbol}.png")
        except Exception:  # noqa: BLE001
            logger.exception("Chart generation failed for %s", pos.symbol)
            return None

    def _send(self, date_str, perf, pdf_path, narratives) -> None:  # noqa: ANN001
        subject = f"[daytrader-bot] Daily Report {date_str} — Net {perf.net_pnl:+,.2f} ({perf.return_pct:+.2f}%)"
        body = (
            f"Daily trading report for {date_str}.\n\n"
            f"Net P/L: ${perf.net_pnl:,.2f} ({perf.return_pct:+.2f}%)\n"
            f"Trades: {perf.total_trades} | Win rate: {perf.win_rate:.1f}% | "
            f"Profit factor: {perf.profit_factor_str}\n\n"
            f"See the attached PDF for charts and selection rationale.\n"
        )
        attachments = [pdf_path]
        if self.settings.reporting.attach_individual_charts:
            attachments += [n.chart_path for n in narratives if n.chart_path]
        self.emailer.send(subject, body, attachments)


def main() -> None:  # pragma: no cover - manual entrypoint
    from daytrader.utils.logging_setup import setup_logging

    settings = get_settings()
    setup_logging(settings.app.log_level)
    ReportEngine.from_settings(settings).generate()


if __name__ == "__main__":  # pragma: no cover
    main()
