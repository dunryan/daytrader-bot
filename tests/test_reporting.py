"""Tests for Module 6: metrics, emailer, and end-to-end PDF generation."""

from __future__ import annotations

import datetime as dt

from daytrader.config.settings import Secrets, Settings
from daytrader.persistence.models import Position
from daytrader.persistence.repositories import (
    DailyMetricsRepository,
    EquityRepository,
    PositionRepository,
    SignalRepository,
    WatchlistRepository,
)
from daytrader.reporting.emailer import EmailSender, build_message
from daytrader.reporting.metrics import compute_performance
from daytrader.reporting.report_engine import ReportEngine


# ── metrics ──────────────────────────────────────────────────
def _pos(pnl: float) -> Position:
    return Position(
        trade_date="2026-06-01", symbol="X", direction="LONG", qty=10,
        entry_price=100.0, entry_time=dt.datetime.now(dt.timezone.utc),
        realized_pnl=pnl, status="CLOSED", is_simulated=True,
    )


def test_compute_performance_basic():
    closed = [_pos(300), _pos(-100), _pos(200), _pos(-50)]
    perf = compute_performance(closed, starting_equity=100_000)
    assert perf.total_trades == 4
    assert perf.winning_trades == 2
    assert perf.losing_trades == 2
    assert perf.win_rate == 50.0
    assert perf.gross_profit == 500
    assert perf.gross_loss == -150
    assert perf.net_pnl == 350
    assert round(perf.profit_factor, 3) == round(500 / 150, 3)
    assert round(perf.return_pct, 4) == 0.35


def test_compute_performance_no_losses_profit_factor_none():
    perf = compute_performance([_pos(100), _pos(50)], starting_equity=10_000)
    assert perf.profit_factor is None
    assert perf.profit_factor_str == "n/a"


def test_compute_performance_empty():
    perf = compute_performance([], starting_equity=100_000)
    assert perf.total_trades == 0
    assert perf.win_rate == 0.0
    assert perf.net_pnl == 0.0


# ── emailer ──────────────────────────────────────────────────
def test_build_message_with_attachment(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    msg = build_message("a@x.com", "b@y.com", "Subj", "Body", [pdf])
    assert msg["From"] == "a@x.com"
    assert msg["To"] == "b@y.com"
    assert msg["Subject"] == "Subj"
    attachments = [p for p in msg.iter_attachments()]
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "report.pdf"


def test_email_sender_not_configured_skips():
    sender = EmailSender(Secrets(_env_file=None))  # ignore any developer .env
    assert sender.is_configured is False
    assert sender.send("s", "b", []) is False


# ── report engine end-to-end ─────────────────────────────────
def _seed_day(db, date_str="2026-06-01"):
    WatchlistRepository(db).replace_for_day(
        date_str,
        [{"symbol": "AAPL", "reason": "gap-up 3.2%; RVOL 2.1x", "rank": 1, "avg_daily_volume": 5_000_000}],
    )
    SignalRepository(db).record(
        date_str, "AAPL", "vwap_pullback", "BUY",
        price_at_signal=100.0, confidence=0.8, rationale="VWAP pullback long",
    )
    positions = PositionRepository(db)
    pid = positions.open_position(
        trade_date=date_str, symbol="AAPL", strategy="vwap_pullback", direction="LONG",
        qty=100, entry_price=100.0, entry_time=dt.datetime(2026, 6, 1, 14, 0, tzinfo=dt.timezone.utc),
        stop_loss=98.0, take_profit=104.0, is_simulated=True,
    )
    positions.close_position(
        pid, exit_price=104.0, exit_time=dt.datetime(2026, 6, 1, 15, 0, tzinfo=dt.timezone.utc),
        exit_reason="TP", realized_pnl=400.0, pnl_percent=4.0,
    )
    DailyMetricsRepository(db).get_or_create(date_str, 100_000.0)
    DailyMetricsRepository(db).update(date_str, ending_equity=100_400.0, realized_pnl=400.0)
    EquityRepository(db).add_point(date_str, 100_000.0, dt.datetime(2026, 6, 1, 14, 0, tzinfo=dt.timezone.utc))
    EquityRepository(db).add_point(date_str, 100_400.0, dt.datetime(2026, 6, 1, 15, 0, tzinfo=dt.timezone.utc))


def test_report_engine_generates_pdf(db, tmp_path):
    settings = Settings()
    settings.reporting.output_dir = str(tmp_path / "reports")
    settings.reporting.send_email = False
    _seed_day(db)

    # data_engine=None -> per-trade charts skipped (no network); equity chart still rendered.
    engine = ReportEngine(settings, db, data_engine=None, emailer=EmailSender(Secrets()))
    pdf_path = engine.generate(dt.date(2026, 6, 1), send_email=False)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000  # a real PDF, not empty
    # Report marked generated in metrics.
    assert DailyMetricsRepository(db).get("2026-06-01").report_generated is True
