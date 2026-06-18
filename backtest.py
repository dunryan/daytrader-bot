#!/usr/bin/env python3
"""Backtest CLI — replay strategies over historical bars.

Usage:
    python backtest.py --symbols-file config/backtest_universe.yaml --universe-set training \\
        --start 2021-01-02 --end 2026-06-05 --train-end 2025-12-31 --walk-forward \\
        --regime enforce --gap-days-only --vol-gate enforce \\
        --export-labels data/backtest/labels_train.parquet

Promotion gates (in order): positive expectancy after costs, profit factor,
per-period stability, drawdown tolerance — then paper, then live.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import yaml

# Allow running directly from a checkout (src/ layout) without install.
SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daytrader.backtest.data_store import BarStore  # noqa: E402
from daytrader.backtest.engine import BacktestEngine, gap_eligible_days  # noqa: E402
from daytrader.backtest.screener_parity import (  # noqa: E402
    intersect_eligible_days,
    parse_cutoff_time,
    premarket_rvol_eligible_days,
)
from daytrader.backtest.labels import export_labels, signals_to_frame  # noqa: E402
from daytrader.backtest.metrics import (  # noqa: E402
    compute_metrics,
    format_report,
    metrics_by_period,
    metrics_by_strategy,
)
from daytrader.config.settings import get_settings  # noqa: E402
from daytrader.data.data_engine import DataEngine  # noqa: E402
from daytrader.ml.meta_label import SignalFilter  # noqa: E402
from daytrader.data.providers import get_provider  # noqa: E402
from daytrader.data.providers.base import Timeframe  # noqa: E402
from daytrader.strategy.router import build_strategies  # noqa: E402
from daytrader.strategy.vol_gate import blocked_trading_days  # noqa: E402
from daytrader.utils.logging_setup import setup_logging  # noqa: E402

KNOWN_STRATEGIES = ("vwap_pullback", "opening_range_breakout", "momentum_scalp")


def load_universe_symbols(path: Path, universe_set: str) -> list[str]:
    """Load symbol lists from a YAML universe file or plain-text ticker list."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text) or {}
        if universe_set == "training":
            symbols = data.get("training", [])
        elif universe_set == "holdout":
            symbols = data.get("holdout", [])
        elif universe_set == "all":
            symbols = list(data.get("training", [])) + list(data.get("holdout", []))
        else:
            raise SystemExit(f"Unknown --universe-set {universe_set!r}")
    else:
        symbols = [
            line.strip().upper()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [str(s).upper() for s in symbols if s and str(s).upper().isalpha() or str(s).replace(".", "").isalnum()]


def parse_date(value: str) -> dt.date:
    return dt.datetime.fromisoformat(value).date()


def split_trades(trades, cutoff: dt.date):  # noqa: ANN001
    train = [t for t in trades if t.entry_time.date() <= cutoff]
    oos = [t for t in trades if t.entry_time.date() > cutoff]
    return train, oos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="daytrader-bot backtester")
    sym = parser.add_mutually_exclusive_group(required=True)
    sym.add_argument("--symbols", help="comma-separated symbols")
    sym.add_argument("--symbols-file", help="YAML universe file or plain ticker list")
    parser.add_argument("--universe-set", choices=("training", "holdout", "all"),
                        default="training",
                        help="which list to load from --symbols-file (default: training)")
    parser.add_argument("--exclude-symbols", default=None,
                        help="comma-separated symbols to drop")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--train-end", default=None,
                        help="YYYY-MM-DD split: print in-sample vs OOS metrics")
    parser.add_argument("--strategies", default=None,
                        help=f"comma-separated subset of {KNOWN_STRATEGIES}")
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--atr-impact", type=float, default=0.05)
    parser.add_argument("--feed", choices=("iex", "sip"), default=None,
                        help="override data.feed for this run (Phase 3 SIP A/B)")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--gap-days-only", action="store_true")
    parser.add_argument(
        "--premarket-rvol",
        action="store_true",
        help="require premarket RVOL >= screener threshold by cutoff (default 07:00 ET)",
    )
    parser.add_argument(
        "--min-premarket-rvol",
        type=float,
        default=None,
        help="override research.filters.min_relative_volume for --premarket-rvol",
    )
    parser.add_argument(
        "--premarket-cutoff",
        default=None,
        help="ET HH:MM to measure cumulative premarket volume (default: schedule or 07:00)",
    )
    parser.add_argument("--min-gap-pct", type=float, default=None)
    parser.add_argument("--max-gap-pct", type=float, default=None,
                        help="reject exhaustion gaps above this |gap| %% (default: config or off)")
    parser.add_argument("--max-gap-norm", type=float, default=None,
                        help="reject gaps above this ATR multiple (default: config or off)")
    parser.add_argument("--regime", choices=("off", "shadow", "enforce"), default=None)
    parser.add_argument("--vol-gate", choices=("off", "shadow", "enforce"), default=None,
                        help="override strategies.vol_gate.mode")
    parser.add_argument("--pf-gate", choices=("off", "shadow", "enforce"), default=None,
                        help="override strategies.pf_gate.mode (rolling PF deployment gate)")
    parser.add_argument("--pf-lookback", type=int, default=None,
                        help="rolling PF gate lookback trades")
    parser.add_argument("--pf-min", type=float, default=None,
                        help="rolling PF gate minimum profit factor")
    parser.add_argument("--meta-filter", choices=("off", "shadow", "enforce"), default=None,
                        help="meta-label filter mode (off for label export)")
    parser.add_argument("--meta-threshold", type=float, default=None,
                        help="meta-label P(win) threshold")
    parser.add_argument("--meta-model", default=None,
                        help="path to meta-label model pickle")
    parser.add_argument("--orb-entry-window", type=int, default=None,
                        help="ORB max minutes after 09:30 for new entries (0=off)")
    parser.add_argument("--orb-min-breakout-rvol", type=float, default=None,
                        help="ORB min breakout bar RVOL (overrides volume_confirmation 1.0x)")
    parser.add_argument("--orb-min-width-pct", type=float, default=None,
                        help="ORB min opening-range width as %% of session open (0=off)")
    parser.add_argument("--orb-max-width-atr", type=float, default=None,
                        help="ORB max opening-range width as multiple of daily ATR (0=off)")
    parser.add_argument("--tp-method", choices=("trailing", "fixed"), default=None)
    parser.add_argument("--tp-rr", type=float, default=None)
    parser.add_argument("--trail-atr-mult", type=float, default=None)
    parser.add_argument("--stop-atr-mult", type=float, default=None)
    parser.add_argument("--orb-entry-mode", choices=("breakout", "retest"), default=None)
    parser.add_argument("--export-labels", default=None)
    parser.add_argument("--export-labels-oos", default=None,
                        help="optional second export for trades after --train-end")
    parser.add_argument("--cache-dir", default="data/backtest_cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.app.log_level)

    if args.feed:
        settings.data.feed = args.feed
        print(f"data feed: {args.feed}")

    if args.regime:
        settings.strategies.regime_filter.mode = args.regime
        print(f"regime filter mode: {args.regime}")

    if args.vol_gate:
        settings.strategies.vol_gate.mode = args.vol_gate
        print(f"vol gate mode: {args.vol_gate}")

    if args.pf_gate:
        settings.strategies.pf_gate.mode = args.pf_gate
        print(f"pf gate mode: {args.pf_gate}")
    if args.pf_lookback is not None:
        settings.strategies.pf_gate.lookback_trades = args.pf_lookback
    if args.pf_min is not None:
        settings.strategies.pf_gate.min_pf = args.pf_min
    pf = settings.strategies.pf_gate
    if pf.mode != "off":
        print(f"pf gate: lookback={pf.lookback_trades} min_pf={pf.min_pf} min_trades={pf.min_trades}")

    if args.meta_filter:
        settings.ml.meta_filter.mode = args.meta_filter
        print(f"meta filter mode: {args.meta_filter}")
    if args.meta_threshold is not None:
        settings.ml.meta_filter.threshold = args.meta_threshold
    if args.meta_model:
        settings.ml.meta_filter.model_path = args.meta_model
    meta = settings.ml.meta_filter
    if meta.mode != "off":
        print(f"meta filter: threshold={meta.threshold} model={meta.model_path}")
    if args.export_labels and meta.mode == "enforce":
        print("WARNING: --export-labels with meta-filter enforce skews training data; using off for this run.")
        settings.ml.meta_filter.mode = "off"
        meta = settings.ml.meta_filter

    orb = settings.strategies.opening_range_breakout
    if args.orb_entry_window is not None:
        orb.max_entry_minutes_after_open = args.orb_entry_window
        print(f"ORB entry window: {args.orb_entry_window} min after open")
    elif getattr(orb, "max_entry_minutes_after_open", 0) > 0:
        print(f"ORB entry window: {orb.max_entry_minutes_after_open} min after open (config)")

    if args.orb_min_breakout_rvol is not None:
        orb.min_breakout_rvol = args.orb_min_breakout_rvol
    if args.orb_min_width_pct is not None:
        orb.min_or_width_pct = args.orb_min_width_pct
    if args.orb_max_width_atr is not None:
        orb.max_or_width_atr = args.orb_max_width_atr
    hv = float(getattr(orb, "min_breakout_rvol", 0) or 0)
    min_w = float(getattr(orb, "min_or_width_pct", 0) or 0)
    max_w = float(getattr(orb, "max_or_width_atr", 0) or 0)
    if hv > 0 or min_w > 0 or max_w > 0:
        print(f"ORB filters: min_breakout_rvol={hv or 'off'} min_or_width_pct={min_w or 'off'} "
              f"max_or_width_atr={max_w or 'off'}")

    if args.max_gap_norm is not None:
        orb.max_gap_norm = args.max_gap_norm
    elif getattr(orb, "max_gap_norm", 0) > 0:
        pass
    max_gap_norm = float(getattr(orb, "max_gap_norm", 0) or 0)
    max_gap_pct = args.max_gap_pct
    if max_gap_pct is None:
        max_gap_pct = float(getattr(orb, "max_gap_pct", 0) or 0)

    if args.tp_method:
        settings.risk.take_profit.method = args.tp_method
    if args.tp_rr is not None:
        settings.risk.take_profit.risk_reward_ratio = args.tp_rr
    if args.trail_atr_mult is not None:
        settings.risk.take_profit.trailing_atr_multiplier = args.trail_atr_mult
    if args.stop_atr_mult is not None:
        settings.risk.stop_loss.atr_multiplier = args.stop_atr_mult
    tp = settings.risk.take_profit
    print(f"exits: method={tp.method} rr={tp.risk_reward_ratio} "
          f"trail_atr={tp.trailing_atr_multiplier} stop_atr={settings.risk.stop_loss.atr_multiplier}")

    if args.orb_entry_mode:
        settings.strategies.opening_range_breakout.entry_mode = args.orb_entry_mode
        print(f"ORB entry mode: {args.orb_entry_mode}")

    if args.strategies:
        chosen = {s.strip() for s in args.strategies.split(",") if s.strip()}
        unknown = chosen - set(KNOWN_STRATEGIES)
        if unknown:
            raise SystemExit(f"Unknown strategies: {sorted(unknown)}")
        for name in KNOWN_STRATEGIES:
            getattr(settings.strategies, name).enabled = name in chosen
    strategies = build_strategies(settings)
    if not strategies:
        raise SystemExit("No strategies enabled.")

    if args.symbols_file:
        symbols = load_universe_symbols(Path(args.symbols_file), args.universe_set)
        print(f"universe ({args.universe_set}): {len(symbols)} symbols from {args.symbols_file}")
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.exclude_symbols:
        excluded = {s.strip().upper() for s in args.exclude_symbols.split(",") if s.strip()}
        symbols = [s for s in symbols if s not in excluded]
        print(f"excluded: {', '.join(sorted(excluded))} -> {len(symbols)} symbol(s) remain")
        if not symbols:
            raise SystemExit("No symbols left after --exclude-symbols.")

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(args.end).replace(
        hour=23, minute=59, tzinfo=dt.timezone.utc
    )
    train_end = parse_date(args.train_end) if args.train_end else None

    provider = get_provider(settings)
    store = BarStore(args.cache_dir)
    data_engine = DataEngine(provider, settings.indicators, rth_only=settings.data.rth_only)

    gate_symbols = list(dict.fromkeys(symbols + ["SPY"]))
    vix_sym = settings.strategies.vol_gate.vix_symbol.upper()
    print(f"Loading {len(symbols)} symbol(s): 5m bars {args.start} -> {args.end} ...")
    feed = settings.data.feed
    raw_5m = store.get(provider, symbols, Timeframe.MIN_5, start, end, feed=feed)
    raw_day = store.get(
        provider, gate_symbols, Timeframe.DAY, start - dt.timedelta(days=400), end, feed=feed
    )
    vix_day = store.get(
        provider, [vix_sym], Timeframe.DAY, start - dt.timedelta(days=400), end, feed=feed
    ).get(vix_sym)

    intraday = {s: data_engine.enrich(df, Timeframe.MIN_5) for s, df in raw_5m.items()}
    daily = {s: data_engine.enrich(df, Timeframe.DAY) for s, df in raw_day.items() if s in symbols}
    spy_daily = daily.get("SPY") or raw_day.get("SPY")
    if spy_daily is not None and "SPY" not in daily:
        spy_daily = data_engine.enrich(spy_daily, Timeframe.DAY)

    missing = [s for s in symbols if s not in intraday]
    if missing:
        print(f"WARNING: no intraday data for {missing}")
    if not intraday:
        raise SystemExit("No data to backtest.")

    eligible_days = None
    if args.gap_days_only:
        min_gap = args.min_gap_pct or settings.research.filters.min_gap_percent
        eligible_days = {
            s: gap_eligible_days(
                raw_day.get(s), min_gap, max_gap_pct=max_gap_pct, max_gap_norm=max_gap_norm
            )
            for s in symbols if s in raw_day
        }
        total = sum(len(v) for v in eligible_days.values())
        print(f"gap-days-only: |gap|>={min_gap:.1f}% max_gap={max_gap_pct or 'off'} "
              f"max_norm={max_gap_norm or 'off'} -> {total} eligible symbol-days")

    if args.premarket_rvol:
        min_rvol = args.min_premarket_rvol or settings.research.filters.min_relative_volume
        cutoff_str = args.premarket_cutoff or settings.schedule.premarket_research
        cutoff = parse_cutoff_time(cutoff_str)
        pm_eligible = {
            s: premarket_rvol_eligible_days(raw_5m.get(s), raw_day.get(s), min_rvol, cutoff)
            for s in symbols if s in raw_day
        }
        prev_total = sum(len(v) for v in eligible_days.values()) if eligible_days else None
        eligible_days = intersect_eligible_days(eligible_days, pm_eligible, symbols)
        after = sum(len(v) for v in eligible_days.values())
        extra = f" (was {prev_total} before RVOL gate)" if prev_total is not None else ""
        print(
            f"premarket-rvol: >= {min_rvol:.2f}x by {cutoff_str} ET "
            f"(extended 5m bars) -> {after} eligible symbol-days{extra}"
        )

    blocked = blocked_trading_days(
        start.date(), end.date(), settings.strategies.vol_gate,
        vix_daily=vix_day, spy_daily=spy_daily,
    )
    if settings.strategies.vol_gate.mode != "off":
        print(f"vol gate ({settings.strategies.vol_gate.mode}): {len(blocked)} blocked session-days")

    signal_filter = (
        SignalFilter.from_settings(settings)
        if settings.ml.meta_filter.mode != "off"
        else None
    )
    engine = BacktestEngine(
        settings, strategies,
        starting_cash=args.cash,
        spread_bps=args.spread_bps,
        atr_impact_coeff=args.atr_impact,
        signal_filter=signal_filter,
    )
    result = engine.run(
        intraday, daily, eligible_days=eligible_days, blocked_days=blocked or None
    )

    if train_end:
        train_trades, oos_trades = split_trades(result.trades, train_end)
        print("\n" + "=" * 64)
        print(f"IN-SAMPLE (through {train_end})")
        print("=" * 64)
        print(format_report(
            compute_metrics(train_trades, result.equity_curve),
            metrics_by_strategy(train_trades),
            metrics_by_period(train_trades, result.equity_curve) if args.walk_forward else None,
        ))
        print("\n" + "=" * 64)
        print(f"OUT-OF-SAMPLE (after {train_end})")
        print("=" * 64)
        print(format_report(
            compute_metrics(oos_trades, result.equity_curve),
            metrics_by_strategy(oos_trades),
            metrics_by_period(oos_trades, result.equity_curve) if args.walk_forward else None,
        ))
    else:
        overall = compute_metrics(result.trades, result.equity_curve)
        per_strategy = metrics_by_strategy(result.trades)
        per_period = (
            metrics_by_period(result.trades, result.equity_curve) if args.walk_forward else None
        )
        print(format_report(overall, per_strategy, per_period))

    filled = sum(1 for s in result.signals if s.filled)
    print(f"signals: {len(result.signals)} generated, {filled} filled")

    if args.export_labels:
        if train_end:
            train_sigs = [
                s for s in result.signals
                if s.filled and s.label is not None and parse_date(s.trade_date) <= train_end
            ]
            n = export_labels(train_sigs, args.export_labels)
            print(f"labels (in-sample): {n} rows -> {args.export_labels}")
            oos_path = args.export_labels_oos or str(
                Path(args.export_labels).with_name(
                    Path(args.export_labels).stem + "_oos" + Path(args.export_labels).suffix
                )
            )
            oos_sigs = [
                s for s in result.signals
                if s.filled and s.label is not None and parse_date(s.trade_date) > train_end
            ]
            n_oos = export_labels(oos_sigs, oos_path)
            print(f"labels (OOS): {n_oos} rows -> {oos_path}")
            if n:
                print("Next: python scripts/train_meta_model.py --labels", args.export_labels)
        else:
            n = export_labels(result.signals, args.export_labels)
            print(f"labels: {n} rows -> {args.export_labels}")
            if n:
                print("Next: python scripts/train_meta_model.py --labels", args.export_labels)


if __name__ == "__main__":
    main()
