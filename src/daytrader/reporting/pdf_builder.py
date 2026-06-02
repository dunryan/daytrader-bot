"""Daily performance report PDF assembly (ReportLab)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from daytrader.reporting.metrics import PerformanceMetrics
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TradeNarrative:
    """A traded symbol and the human-readable reason it was selected/traded."""

    symbol: str
    text: str
    chart_path: Path | None = None


def _money(x: float) -> str:
    return f"${x:,.2f}"


def build_report(
    out_path: Path | str,
    trade_date: str,
    perf: PerformanceMetrics,
    starting_equity: float,
    ending_equity: float,
    watchlist_rows: list[dict],
    narratives: list[TradeNarrative],
    equity_chart: Path | None = None,
    halted: bool = False,
    halt_reason: str | None = None,
) -> Path:
    """Compile the daily report PDF and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )
    story: list = []

    # ── Header ──
    story.append(Paragraph(f"Daily Trading Report — {trade_date}", styles["Title"]))
    mode = "SIMULATION (paper)" if True else "LIVE"
    story.append(Paragraph(f"Mode: {mode}", styles["Normal"]))
    if halted:
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f'<font color="red"><b>TRADING HALTED:</b> {halt_reason or "kill switch"}</font>',
            styles["Normal"],
        ))
    story.append(Spacer(1, 12))

    # ── Summary table ──
    net_color = colors.green if perf.net_pnl >= 0 else colors.red
    summary = [
        ["Metric", "Value", "Metric", "Value"],
        ["Net P/L", _money(perf.net_pnl), "Total Trades", str(perf.total_trades)],
        ["Return", f"{perf.return_pct:.2f}%", "Win Rate", f"{perf.win_rate:.1f}%"],
        ["Gross Profit", _money(perf.gross_profit), "Profit Factor", perf.profit_factor_str],
        ["Gross Loss", _money(perf.gross_loss), "Expectancy/Trade", _money(perf.expectancy)],
        ["Avg Win", _money(perf.avg_win), "Avg Loss", _money(perf.avg_loss)],
        ["Largest Win", _money(perf.largest_win), "Largest Loss", _money(perf.largest_loss)],
        ["Starting Equity", _money(starting_equity), "Ending Equity", _money(ending_equity)],
    ]
    table = Table(summary, colWidths=[1.4 * inch, 1.6 * inch, 1.4 * inch, 1.6 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("TEXTCOLOR", (1, 1), (1, 1), net_color),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(Paragraph("Performance Summary", styles["Heading2"]))
    story.append(table)
    story.append(Spacer(1, 14))

    # ── Equity curve ──
    if equity_chart and Path(equity_chart).exists():
        story.append(Paragraph("Equity Curve", styles["Heading2"]))
        story.append(Image(str(equity_chart), width=7.0 * inch, height=2.45 * inch))
        story.append(Spacer(1, 12))

    # ── Watchlist ──
    if watchlist_rows:
        story.append(Paragraph("Watchlist & Selection Rationale", styles["Heading2"]))
        wl = [["#", "Symbol", "Why selected"]]
        for row in watchlist_rows:
            wl.append([str(row.get("rank", "")), row.get("symbol", ""), row.get("reason", "")])
        wl_table = Table(wl, colWidths=[0.4 * inch, 0.9 * inch, 5.7 * inch])
        wl_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(wl_table)
        story.append(Spacer(1, 12))

    # ── Per-trade narrative + charts ──
    if narratives:
        story.append(PageBreak())
        story.append(Paragraph("Trades Executed", styles["Heading2"]))
        for n in narratives:
            story.append(Paragraph(f"<b>{n.symbol}</b>", styles["Heading3"]))
            story.append(Paragraph(n.text, styles["Normal"]))
            story.append(Spacer(1, 6))
            if n.chart_path and Path(n.chart_path).exists():
                story.append(Image(str(n.chart_path), width=7.0 * inch, height=3.5 * inch))
            story.append(Spacer(1, 14))
    else:
        story.append(Paragraph("No trades were executed today.", styles["Normal"]))

    doc.build(story)
    logger.info("Report PDF written to %s", out_path)
    return out_path
