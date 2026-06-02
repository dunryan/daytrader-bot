"""Chart rendering for the daily report (Matplotlib, headless).

Uses the non-interactive ``Agg`` backend so it runs fine in a background
service with no display. Each traded symbol gets a price chart with the entry,
stop, take-profit, and exit visually annotated.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from daytrader.persistence.models import EquityPoint, Position  # noqa: E402
from daytrader.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)


def plot_trade(position: Position, df: pd.DataFrame, out_path: Path | str) -> Path | None:
    """Render a price chart for one position with entry/stop/TP/exit marked.

    ``df`` is the symbol's intraday OHLCV (optionally with a ``vwap`` column).
    Returns the output path, or ``None`` if there's no data to plot.
    """
    out_path = Path(out_path)
    if df is None or df.empty:
        logger.warning("No price data to chart for %s", position.symbol)
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["close"], color="#1f77b4", linewidth=1.2, label="Close")
    if "vwap" in df.columns:
        ax.plot(df.index, df["vwap"], color="#9467bd", linewidth=1.0, linestyle="--", label="VWAP")

    # Horizontal reference lines.
    if position.stop_loss:
        ax.axhline(position.stop_loss, color="#d62728", linestyle=":", linewidth=1.0, label="Stop")
    if position.take_profit:
        ax.axhline(position.take_profit, color="#2ca02c", linestyle=":", linewidth=1.0, label="Target")

    # Entry / exit markers.
    if position.entry_time is not None and position.entry_price:
        ax.scatter([position.entry_time], [position.entry_price], marker="^", s=120,
                   color="#2ca02c", zorder=5, label=f"Entry {position.entry_price:.2f}")
    if position.exit_time is not None and position.exit_price:
        color = "#2ca02c" if (position.realized_pnl or 0) >= 0 else "#d62728"
        ax.scatter([position.exit_time], [position.exit_price], marker="v", s=120,
                   color=color, zorder=5, label=f"Exit {position.exit_price:.2f}")

    pnl = position.realized_pnl or 0.0
    ax.set_title(
        f"{position.symbol} — {position.direction} {int(position.qty)} sh "
        f"[{position.strategy or 'n/a'}]  P/L ${pnl:,.2f} ({position.pnl_percent or 0:.2f}%)"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_equity_curve(points: list[EquityPoint], out_path: Path | str) -> Path | None:
    """Render the intraday equity curve."""
    out_path = Path(out_path)
    if not points:
        return None
    times = [p.timestamp for p in points]
    equity = [p.equity for p in points]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(times, equity, color="#1f77b4", linewidth=1.4)
    ax.fill_between(times, equity, min(equity), alpha=0.1, color="#1f77b4")
    ax.set_title("Intraday Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    if isinstance(times[0], dt.datetime):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
