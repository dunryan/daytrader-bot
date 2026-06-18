#!/usr/bin/env python3
"""Phase 0 paper-sim gate status — one command for weekly review.

Reads ``data/trading.db`` and prints watchlist, signals, fills, cumulative
closed trades, rolling PF, and distance to the 30-trade / PF parity gates.

Usage:
    python scripts/sim_gate_status.py
    python scripts/sim_gate_status.py --date 2026-06-09
    python scripts/sim_gate_status.py --since 2026-06-01   # trades after deploy
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daytrader.config.settings import get_settings  # noqa: E402
from daytrader.persistence.database import Database  # noqa: E402
from daytrader.persistence.repositories import (  # noqa: E402
    PositionRepository,
    SignalRepository,
    WatchlistRepository,
)
from daytrader.reporting.metrics import compute_performance  # noqa: E402
from daytrader.strategy.pf_gate import evaluate_pf_gate, profit_factor  # noqa: E402
from daytrader.utils.logging_setup import setup_logging  # noqa: E402

# Keep in sync with docs/sim_validation.md
GATE_MIN_TRADES = 30
BACKTEST_PF = 1.16
PF_TOLERANCE = 0.15
SIM_PF_FLOOR = BACKTEST_PF - PF_TOLERANCE  # 1.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper sim Phase 0 gate status")
    p.add_argument("--date", default=None, help="session date YYYY-MM-DD (default: today ET)")
    p.add_argument("--since", default=None, help="only count closed trades on/after this date")
    p.add_argument("--db", default=None, help="override SQLite path (default: config app.db_path)")
    return p.parse_args()


def _today_et() -> dt.date:
    try:
        from zoneinfo import ZoneInfo

        return dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001
        return dt.date.today()


def _fmt_pf(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def _filter_closed(positions, since: dt.date | None):  # noqa: ANN001
    if since is None:
        return list(positions)
    return [p for p in positions if p.trade_date >= since.isoformat()]


def main() -> None:
    args = parse_args()
    setup_logging("ERROR")
    settings = get_settings()
    trade_date = args.date or _today_et().isoformat()
    since = dt.date.fromisoformat(args.since) if args.since else None

    db_path = Path(args.db) if args.db else Path(settings.app.db_path)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    db = Database(f"sqlite:///{db_path}")
    watchlist = WatchlistRepository(db)
    signals = SignalRepository(db)
    positions = PositionRepository(db)

    wl = watchlist.get_for_day(trade_date)
    day_sigs = signals.get_for_day(trade_date)
    actionable = [s for s in day_sigs if s.direction in ("BUY", "SELL")]
    acted = [s for s in actionable if s.acted_on]
    meta_scored = [s for s in actionable if s.meta_prob is not None]
    meta_below = [s for s in meta_scored if s.meta_prob < settings.ml.meta_filter.threshold]

    all_closed = positions.get_recent_closed(limit=500)
    if since:
        all_closed = _filter_closed(all_closed, since)
    closed_today = [p for p in all_closed if p.trade_date == trade_date]
    orb_closed = [p for p in all_closed if p.strategy == "opening_range_breakout"]

    cum = compute_performance(all_closed, settings.risk.starting_equity)
    orb_cum = compute_performance(orb_closed, settings.risk.starting_equity) if orb_closed else None

    lookback = settings.strategies.pf_gate.lookback_trades
    recent_pnls = [float(p.realized_pnl or 0.0) for p in all_closed[-lookback:]]
    rolling_pf = profit_factor(recent_pnls) if recent_pnls else None
    _, pf_details = evaluate_pf_gate(
        [float(p.realized_pnl or 0.0) for p in all_closed],
        settings.strategies.pf_gate,
    )

    trades_needed = max(0, GATE_MIN_TRADES - cum.total_trades)
    pf_ok = cum.profit_factor is not None and cum.profit_factor >= SIM_PF_FLOOR
    trades_ok = cum.total_trades >= GATE_MIN_TRADES
    gate_pass = trades_ok and pf_ok

    print("=" * 72)
    print(f"Phase 0 sim gate status  |  session {trade_date}")
    print(f"DB: {db_path}")
    if since:
        print(f"Closed trades counted since: {since.isoformat()}")
    print("=" * 72)

    print(f"\n--- Today ({trade_date}) ---")
    print(f"  Watchlist:     {len(wl)} symbol(s)")
    for w in wl[:10]:
        rvol = f"{w.relative_volume:.2f}x" if w.relative_volume is not None else "n/a"
        print(f"    #{w.rank or '?':>2} {w.symbol:<6} RVOL {rvol}  {w.reason or ''}")
    if len(wl) > 10:
        print(f"    ... +{len(wl) - 10} more")

    print(f"  Signals:       {len(actionable)} actionable ({len(day_sigs)} total incl. HOLD)")
    print(f"  Acted on:      {len(acted)} (filled or attempted entry)")
    print(f"  Closed today:  {len(closed_today)} round-trip(s)")
    if meta_scored:
        avg_meta = sum(s.meta_prob for s in meta_scored) / len(meta_scored)
        print(
            f"  Meta (shadow): {len(meta_scored)} scored, avg P(win)={avg_meta:.3f}, "
            f"{len(meta_below)} below threshold {settings.ml.meta_filter.threshold}"
        )
    else:
        print("  Meta (shadow): no scored signals today")

    print("\n--- Cumulative (closed round-trips) ---")
    print(f"  Total closed:  {cum.total_trades}")
    print(f"  Net P/L:       ${cum.net_pnl:,.2f}")
    print(f"  PF:            {_fmt_pf(cum.profit_factor)}  (backtest ref {BACKTEST_PF:.2f})")
    print(f"  Expectancy:    ${cum.expectancy:,.2f}/trade")
    print(f"  Win rate:      {cum.win_rate:.1f}%")
    if orb_cum and orb_cum.total_trades != cum.total_trades:
        print(
            f"  ORB only:      {orb_cum.total_trades} trades, PF {_fmt_pf(orb_cum.profit_factor)}, "
            f"exp ${orb_cum.expectancy:,.2f}"
        )

    print(f"\n--- Rolling PF (last {lookback} closed, pf_gate lookback) ---")
    print(f"  n={len(recent_pnls)}  PF={_fmt_pf(rolling_pf)}")
    if pf_details.get("pf_gate_trades", 0) >= settings.strategies.pf_gate.min_trades:
        shadow_pf = pf_details.get("pf_gate_pf")
        would_block = shadow_pf is not None and shadow_pf < settings.strategies.pf_gate.min_pf
        mode = settings.strategies.pf_gate.mode
        print(
            f"  PF gate ({mode}): PF={shadow_pf:.2f} vs min {settings.strategies.pf_gate.min_pf:.2f} "
            f"-> {'would block' if would_block else 'ok'}"
        )
    else:
        print(
            f"  PF gate ({settings.strategies.pf_gate.mode}): "
            f"need {settings.strategies.pf_gate.min_trades} trades "
            f"(have {int(pf_details.get('pf_gate_trades', 0))})"
        )

    print("\n--- Phase 0 gates (see docs/sim_validation.md) ---")
    print(f"  [{'PASS' if trades_ok else 'WAIT'}] Min trades:  {cum.total_trades}/{GATE_MIN_TRADES}"
          + (f"  ({trades_needed} to go)" if trades_needed else ""))
    print(
        f"  [{'PASS' if pf_ok else 'WAIT'}] Sim PF floor:  {_fmt_pf(cum.profit_factor)} "
        f"(need >= {SIM_PF_FLOOR:.2f} = backtest {BACKTEST_PF:.2f} - {PF_TOLERANCE:.2f})"
    )
    print(f"\n  Overall Phase 0: {'PASS - ready for Phase 5 prep review' if gate_pass else 'IN PROGRESS'}")

    if closed_today:
        print("\n--- Closed today ---")
        for p in closed_today:
            print(
                f"  {p.symbol:<6} {p.direction:<5} ${p.realized_pnl or 0:,.2f}  "
                f"{p.exit_reason or ''}  ({p.strategy or '?'})"
            )


if __name__ == "__main__":
    main()
