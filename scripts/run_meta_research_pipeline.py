#!/usr/bin/env python3
"""Expanded-universe meta-label research pipeline (SIP + TOD premarket RVOL).

Steps:
  1. SIP backtest on training+holdout symbols (gap-days ∩ TOD premarket RVOL)
  2. Export in-sample / OOS label parquet files (symbol batches to limit RAM)
  3. Train meta model with holdout symbols excluded from fit; report purged CV + holdout AUC

Keeps live sim on core-10 + shadow mode — this is offline research only.

Usage:
    python scripts/run_meta_research_pipeline.py
    python scripts/run_meta_research_pipeline.py --skip-backtest   # retrain only
    python scripts/run_meta_research_pipeline.py --batch-size 8
    python scripts/run_meta_research_pipeline.py --resume-batches
    python scripts/run_meta_research_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "config" / "backtest_universe.yaml"
DEFAULT_LABELS = ROOT / "data" / "backtest" / "labels_expanded_sip_tod.parquet"
DEFAULT_MODEL = ROOT / "data" / "models" / "meta_label_expanded_oos.pkl"
HOLDOUT_AUC_GATE = 0.55


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run expanded meta-label research pipeline")
    p.add_argument("--symbols-file", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--universe-set", default="all", choices=("training", "holdout", "all"))
    p.add_argument("--start", default="2021-01-04",
                   help="backtest start (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-06-05")
    p.add_argument("--train-end", default="2024-12-31",
                   help="in-sample label cutoff; OOS labels written alongside")
    p.add_argument("--eval-start", default="2025-01-01",
                   help="holdout-symbol AUC eval window start")
    p.add_argument("--feed", default="sip", choices=("sip", "iex"))
    p.add_argument("--cache-dir", default="data/backtest_cache_sip")
    p.add_argument("--min-premarket-rvol", type=float, default=1.35)
    p.add_argument("--premarket-cutoff", default="07:00")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--model-out", default=str(DEFAULT_MODEL))
    p.add_argument("--min-rows", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8,
                   help="symbols per backtest replay (0 = single run for full universe)")
    p.add_argument("--resume-batches", action="store_true",
                   help="skip batch exports that already exist on disk")
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_universe_symbols(path: Path, universe_set: str) -> list[str]:
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
            raise SystemExit(f"Unknown universe-set {universe_set!r}")
    else:
        symbols = [
            line.strip().upper()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [
        str(s).upper()
        for s in symbols
        if s and (str(s).upper().isalpha() or str(s).replace(".", "").isalnum())
    ]


def load_holdout_symbols(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [str(s).upper() for s in data.get("holdout", [])]


def chunk_symbols(symbols: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0 or len(symbols) <= batch_size:
        return [symbols]
    return [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]


def merge_parquet_files(paths: list[Path], out: Path) -> int:
    frames = [pd.read_parquet(p) for p in paths if p.exists() and p.stat().st_size > 0]
    if not frames:
        if out.exists():
            out.unlink()
        return 0
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df)


def run_step(cmd: list[str], dry_run: bool) -> None:
    print("\n>>", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def backtest_cmd(
    py: str,
    args: argparse.Namespace,
    symbols: list[str],
    export_labels: Path,
) -> list[str]:
    return [
        py, "backtest.py",
        "--symbols", ",".join(symbols),
        "--start", args.start,
        "--end", args.end,
        "--train-end", args.train_end,
        "--feed", args.feed,
        "--cache-dir", args.cache_dir,
        "--gap-days-only",
        "--premarket-rvol",
        "--premarket-rvol-mode", "tod",
        "--premarket-cutoff", args.premarket_cutoff,
        "--min-premarket-rvol", str(args.min_premarket_rvol),
        "--regime", "enforce",
        "--vol-gate", "off",
        "--pf-gate", "off",
        "--meta-filter", "off",
        "--strategies", "opening_range_breakout",
        "--walk-forward",
        "--export-labels", str(export_labels),
    ]


def run_batched_backtest(
    py: str,
    args: argparse.Namespace,
    symbols: list[str],
    labels: Path,
) -> None:
    batches = chunk_symbols(symbols, args.batch_size)
    batch_dir = labels.parent / f"{labels.stem}_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    in_parts: list[Path] = []
    oos_parts: list[Path] = []
    oos_path = labels.with_name(f"{labels.stem}_oos{labels.suffix}")

    print(f"Backtest batches: {len(batches)} x up to {args.batch_size or len(symbols)} symbols")
    for idx, batch in enumerate(batches, start=1):
        part = batch_dir / f"{labels.stem}_batch{idx:02d}{labels.suffix}"
        part_oos = batch_dir / f"{labels.stem}_batch{idx:02d}_oos{labels.suffix}"
        in_parts.append(part)
        oos_parts.append(part_oos)

        if args.resume_batches and part.exists() and part_oos.exists():
            print(f"  batch {idx}/{len(batches)}: skip (exists) {', '.join(batch)}")
            continue

        print(f"  batch {idx}/{len(batches)}: {', '.join(batch)}")
        run_step(backtest_cmd(py, args, batch, part), args.dry_run)

    if args.dry_run:
        return

    n_in = merge_parquet_files(in_parts, labels)
    n_oos = merge_parquet_files(oos_parts, oos_path)
    print(f"Merged labels: {n_in} in-sample -> {labels}")
    print(f"Merged labels: {n_oos} OOS -> {oos_path}")


def main() -> int:
    args = parse_args()
    universe = Path(args.symbols_file)
    symbols = load_universe_symbols(universe, args.universe_set)
    holdout = load_holdout_symbols(universe)
    labels = Path(args.labels)
    labels.parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    batches = chunk_symbols(symbols, args.batch_size)
    print("=" * 72)
    print("Meta research pipeline (expanded universe, shadow-only output)")
    print(f"  universe: {universe.name} ({args.universe_set}) -> {len(symbols)} symbols")
    print(f"  batches:  {len(batches)} x {args.batch_size or len(symbols)} symbols")
    print(f"  window:   {args.start} -> {args.end}  train-end={args.train_end}")
    print(f"  feed:     {args.feed}  premarket RVOL>={args.min_premarket_rvol} TOD @ {args.premarket_cutoff}")
    print(f"  holdout:  {', '.join(holdout)} (excluded from model fit)")
    print(f"  labels:   {labels}")
    print(f"  model:    {args.model_out}")
    print("=" * 72)

    if not args.skip_backtest:
        if len(batches) == 1:
            run_step(backtest_cmd(py, args, symbols, labels), args.dry_run)
        else:
            run_batched_backtest(py, args, symbols, labels)

    if not args.skip_train:
        if not args.dry_run and not labels.exists():
            print(f"ERROR: labels not found at {labels} — run backtest first.")
            return 1
        run_step(
            [
                py, "scripts/train_meta_model.py",
                "--labels", str(labels),
                "--out", str(args.model_out),
                "--train-end", args.train_end,
                "--eval-start", args.eval_start,
                "--holdout-symbols", ",".join(holdout),
                "--purged-cv",
                "--min-rows", str(args.min_rows),
            ],
            args.dry_run,
        )

    print()
    print("Done. Next steps:")
    print(f"  1. Review purged CV AUC and holdout OOS AUC (gate: >={HOLDOUT_AUC_GATE})")
    print(f"  2. If promising, point sim shadow model at {args.model_out}:")
    print("       ml.meta_filter.model_path: ...  (keep mode: shadow)")
    print("  3. Do NOT switch to enforce until sim + holdout PF beat structural stack alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
