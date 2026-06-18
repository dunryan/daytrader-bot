#!/usr/bin/env python3
"""Sweep ORB min_breakout_rvol on a fixed window (clean + slippage stress).

Loads bar data once, replays each RVOL threshold twice, prints a ranked table.
Use to pick a sweet spot without over-fitting a large parameter grid.

Example:
    python scripts/sweep_orb_rvol.py \\
        --symbols COIN,SMCI,PLTR,TSLA,AMD,MU,NVDA,AVGO,MSTR,META \\
        --start 2025-01-02 --end 2026-06-05 \\
        --feed sip --cache-dir data/backtest_cache_sip

PowerShell one-liner (no script):
    foreach ($r in 1.5,1.6,1.75,1.85,2.0) {
        Write-Host "=== RVOL $r ==="
        .\\.venv\\Scripts\\python.exe backtest.py ... --orb-min-breakout-rvol $r 2>&1 |
            Select-String "trades=|PF="
    }
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daytrader.backtest.data_store import BarStore  # noqa: E402
from daytrader.backtest.engine import BacktestEngine, gap_eligible_days  # noqa: E402
from daytrader.backtest.metrics import compute_metrics  # noqa: E402
from daytrader.config.settings import get_settings  # noqa: E402
from daytrader.data.data_engine import DataEngine  # noqa: E402
from daytrader.data.providers import get_provider  # noqa: E402
from daytrader.data.providers.base import Timeframe  # noqa: E402
from daytrader.strategy.router import build_strategies  # noqa: E402
from daytrader.strategy.vol_gate import blocked_trading_days  # noqa: E402
from daytrader.utils.logging_setup import setup_logging  # noqa: E402


@dataclass
class SweepRow:
    rvol: float
    trades: int
    pf_clean: float | None
    exp_clean: float | None
    pf_stress: float | None
    exp_stress: float | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep ORB min_breakout_rvol")
    p.add_argument("--symbols", required=True, help="comma-separated symbols")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--rvol", default="1.5,1.6,1.75,1.85,2.0",
                   help="comma-separated min_breakout_rvol values")
    p.add_argument("--orb-entry-window", type=int, default=90)
    p.add_argument("--feed", choices=("iex", "sip"), default="sip")
    p.add_argument("--cache-dir", default="data/backtest_cache_sip")
    p.add_argument("--spread-bps-clean", type=float, default=4.0)
    p.add_argument("--atr-impact-clean", type=float, default=0.05)
    p.add_argument("--spread-bps-stress", type=float, default=8.0)
    p.add_argument("--atr-impact-stress", type=float, default=0.10)
    p.add_argument("--min-trades", type=int, default=80,
                   help="minimum trades for ranking (avoid tiny samples)")
    p.add_argument("--min-pf-clean", type=float, default=1.10,
                   help="minimum clean PF to rank a row")
    return p.parse_args()


def _run_once(
    settings,
    intraday,
    daily,
    eligible_days,
    blocked,
    spread_bps: float,
    atr_impact: float,
    min_breakout_rvol: float,
) -> dict:
    s = copy.deepcopy(settings)
    s.ml.meta_filter.mode = "off"
    s.strategies.pf_gate.mode = "off"
    s.strategies.regime_filter.mode = "enforce"
    s.strategies.vol_gate.mode = "off"
    s.strategies.opening_range_breakout.enabled = True
    s.strategies.vwap_pullback.enabled = False
    s.strategies.momentum_scalp.enabled = False
    s.strategies.opening_range_breakout.min_breakout_rvol = min_breakout_rvol
    s.strategies.opening_range_breakout.max_entry_minutes_after_open = settings.strategies.opening_range_breakout.max_entry_minutes_after_open

    strategies = build_strategies(s)
    engine = BacktestEngine(
        s, strategies,
        spread_bps=spread_bps,
        atr_impact_coeff=atr_impact,
        signal_filter=None,
    )
    result = engine.run(
        intraday, daily, eligible_days=eligible_days, blocked_days=blocked or None
    )
    return compute_metrics(result.trades, result.equity_curve)


def _fmt_pf(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _fmt_exp(value: float | None) -> str:
    return f"${value:.2f}" if value is not None else "n/a"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    setup_logging("ERROR")

    settings.data.feed = args.feed
    settings.strategies.opening_range_breakout.max_entry_minutes_after_open = args.orb_entry_window

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    rvols = [float(x.strip()) for x in args.rvol.split(",") if x.strip()]
    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(args.end).replace(
        hour=23, minute=59, tzinfo=dt.timezone.utc
    )

    orb = settings.strategies.opening_range_breakout
    max_gap_norm = float(getattr(orb, "max_gap_norm", 0) or 0)
    max_gap_pct = float(getattr(orb, "max_gap_pct", 0) or 0)
    min_gap = settings.research.filters.min_gap_percent

    provider = get_provider(settings)
    store = BarStore(args.cache_dir)
    data_engine = DataEngine(provider, settings.indicators, rth_only=settings.data.rth_only)
    feed = settings.data.feed

    print(f"Loading {len(symbols)} symbols ({feed}) {args.start} -> {args.end} ...")
    raw_5m = store.get(provider, symbols, Timeframe.MIN_5, start, end, feed=feed)
    gate_symbols = list(dict.fromkeys(symbols + ["SPY"]))
    raw_day = store.get(
        provider, gate_symbols, Timeframe.DAY, start - dt.timedelta(days=400), end, feed=feed
    )
    vix_sym = settings.strategies.vol_gate.vix_symbol.upper()
    vix_day = store.get(
        provider, [vix_sym], Timeframe.DAY, start - dt.timedelta(days=400), end, feed=feed
    ).get(vix_sym)

    intraday = {s: data_engine.enrich(df, Timeframe.MIN_5) for s, df in raw_5m.items()}
    daily = {s: data_engine.enrich(df, Timeframe.DAY) for s, df in raw_day.items() if s in symbols}
    spy_daily = daily.get("SPY") or raw_day.get("SPY")
    if spy_daily is not None and "SPY" not in daily:
        spy_daily = data_engine.enrich(spy_daily, Timeframe.DAY)

    eligible_days = {
        s: gap_eligible_days(
            raw_day.get(s), min_gap, max_gap_pct=max_gap_pct, max_gap_norm=max_gap_norm
        )
        for s in symbols if s in raw_day
    }
    blocked = blocked_trading_days(
        start.date(), end.date(), settings.strategies.vol_gate,
        vix_daily=vix_day, spy_daily=spy_daily,
    )

    rows: list[SweepRow] = []
    for rvol in rvols:
        print(f"  RVOL {rvol:.2f} ...", flush=True)
        clean = _run_once(
            settings, intraday, daily, eligible_days, blocked,
            args.spread_bps_clean, args.atr_impact_clean, rvol,
        )
        stress = _run_once(
            settings, intraday, daily, eligible_days, blocked,
            args.spread_bps_stress, args.atr_impact_stress, rvol,
        )
        rows.append(SweepRow(
            rvol=rvol,
            trades=int(clean["total_trades"]),
            pf_clean=clean.get("profit_factor"),
            exp_clean=clean.get("expectancy"),
            pf_stress=stress.get("profit_factor"),
            exp_stress=stress.get("expectancy"),
        ))

    print()
    print("=" * 78)
    print(f"ORB min_breakout_rvol sweep  |  {args.start} -> {args.end}  |  feed={feed}")
    print(f"clean: spread={args.spread_bps_clean}bps atr_impact={args.atr_impact_clean}")
    print(f"stress: spread={args.spread_bps_stress}bps atr_impact={args.atr_impact_stress}")
    print("=" * 78)
    header = f"{'RVOL':>6}  {'trades':>6}  {'PF_clean':>8}  {'exp_clean':>10}  {'PF_stress':>9}  {'exp_stress':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.rvol:>6.2f}  {row.trades:>6}  {_fmt_pf(row.pf_clean):>8}  "
            f"{_fmt_exp(row.exp_clean):>10}  {_fmt_pf(row.pf_stress):>9}  "
            f"{_fmt_exp(row.exp_stress):>10}"
        )

    eligible = [
        r for r in rows
        if r.trades >= args.min_trades
        and r.pf_clean is not None
        and r.pf_clean >= args.min_pf_clean
    ]
    if eligible:
        best = max(
            eligible,
            key=lambda r: (r.pf_stress or 0.0, r.pf_clean or 0.0, r.trades),
        )
        print()
        print(
            f"Recommended (max PF_stress, PF_clean>={args.min_pf_clean}, trades>={args.min_trades}): "
            f"min_breakout_rvol={best.rvol:.2f}"
        )
        print(f"  clean  PF={_fmt_pf(best.pf_clean)}  exp={_fmt_exp(best.exp_clean)}  trades={best.trades}")
        print(f"  stress PF={_fmt_pf(best.pf_stress)}  exp={_fmt_exp(best.exp_stress)}")
    else:
        print()
        print("No row passed min-trades / min-PF-clean filters — widen grid or relax thresholds.")


if __name__ == "__main__":
    main()
