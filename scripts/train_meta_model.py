#!/usr/bin/env python3
"""Train the meta-label signal classifier from a backtest label export.

Usage:
    python scripts/train_meta_model.py --labels data/backtest/labels_train.parquet \\
        --out data/models/meta_label.pkl --purged-cv \\
        --holdout-symbols AAPL,INTC,QCOM,SNOW,U

Reports time-ordered (purged) cross-validated AUC before saving.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from daytrader.ml.meta_label import MetaLabelModel  # noqa: E402


def load_labels(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _to_ts(row: pd.Series) -> pd.Timestamp:
    if pd.notna(row.get("exit_time")):
        return pd.Timestamp(row["exit_time"])
    return pd.Timestamp(row["trade_date"]) + pd.Timedelta(hours=16)


def purged_train_test_split(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    embargo: dt.timedelta = dt.timedelta(days=1),
) -> tuple[np.ndarray, np.ndarray]:
    """Remove training rows whose label window overlaps the test block (+ embargo)."""
    test_start = df.iloc[test_idx]["timestamp"].map(pd.Timestamp).min()
    test_end = df.iloc[test_idx].apply(_to_ts, axis=1).max()
    embargo_end = test_end + embargo
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_idx] = False
    for pos in range(len(df)):
        if not train_mask[pos]:
            continue
        row = df.iloc[pos]
        entry = pd.Timestamp(row["timestamp"])
        exit_ts = _to_ts(row)
        if exit_ts >= test_start and entry <= embargo_end:
            train_mask[pos] = False
    train_idx = np.flatnonzero(train_mask)
    return train_idx, test_idx


def purged_cv_aucs(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, n_splits: int = 4) -> list[float]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    n = len(df)
    fold_size = max(n // (n_splits + 1), 1)
    aucs: list[float] = []
    for k in range(n_splits):
        test_start = (k + 1) * fold_size
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            break
        test_idx = np.arange(test_start, test_end)
        train_idx, test_idx = purged_train_test_split(df, test_idx)
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue
        if len(set(y[test_idx])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05)
        clf.fit(X[train_idx], y[train_idx])
        aucs.append(float(roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx])[:, 1])))
    return aucs


def time_series_cv_aucs(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, n_splits: int = 4) -> list[float]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    aucs: list[float] = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        if len(set(y[test_idx])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05)
        clf.fit(X[train_idx], y[train_idx])
        aucs.append(float(roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx])[:, 1])))
    return aucs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the meta-label model")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="data/models/meta_label.pkl")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--train-end", default=None, help="YYYY-MM-DD inclusive train cutoff")
    parser.add_argument("--eval-start", default=None, help="YYYY-MM-DD OOS eval start (optional report)")
    parser.add_argument("--holdout-symbols", default=None,
                        help="comma-separated symbols excluded from training (eval only)")
    parser.add_argument("--purged-cv", action="store_true",
                        help="use purged + embargo CV instead of vanilla TimeSeriesSplit")
    args = parser.parse_args()

    df = load_labels(Path(args.labels))
    if args.holdout_symbols:
        holdout = {s.strip().upper() for s in args.holdout_symbols.split(",") if s.strip()}
        eval_df = df[df["symbol"].isin(holdout)]
        df = df[~df["symbol"].isin(holdout)]
        print(f"holdout symbols excluded from training: {sorted(holdout)} ({len(eval_df)} rows held out)")
    else:
        eval_df = pd.DataFrame()

    if args.train_end:
        cutoff = dt.datetime.fromisoformat(args.train_end).date()
        df = df[pd.to_datetime(df["trade_date"]).dt.date <= cutoff]
        print(f"train-end filter: {len(df)} rows through {cutoff}")

    feature_cols = sorted(c for c in df.columns if c.startswith("feat_"))
    if not feature_cols or "label" not in df.columns:
        raise SystemExit("Labels file must contain feat_* columns and a 'label' column.")
    if len(df) < args.min_rows:
        raise SystemExit(
            f"Only {len(df)} labeled signals (< {args.min_rows}). Generate more backtest data."
        )

    order = df["timestamp"].astype(str).argsort(kind="stable") if "timestamp" in df.columns else None
    if order is not None:
        df = df.iloc[order].reset_index(drop=True)

    X = df[feature_cols].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()
    pos_rate = y.mean()
    print(f"{len(df)} signals | {len(feature_cols)} features | base win rate {pos_rate:.1%}")

    if args.purged_cv:
        aucs = purged_cv_aucs(df, X, y)
        cv_name = "Purged CV AUC"
    else:
        aucs = time_series_cv_aucs(df, X, y)
        cv_name = "Time-ordered CV AUC"

    if aucs:
        mean_auc = sum(aucs) / len(aucs)
        print(f"{cv_name}: {mean_auc:.3f} (folds: {[f'{a:.3f}' for a in aucs]})")
        if mean_auc < 0.53:
            print("WARNING: AUC barely above coin-flip — do not deploy in enforce mode.")

    if not eval_df.empty and args.eval_start:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        eval_df = eval_df[
            pd.to_datetime(eval_df["trade_date"]).dt.date >= dt.datetime.fromisoformat(args.eval_start).date()
        ]
        if len(eval_df) >= 30 and len(set(eval_df["label"])) > 1:
            final_pre = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05)
            final_pre.fit(X, y)
            X_eval = eval_df[feature_cols].astype(float).to_numpy()
            y_eval = eval_df["label"].astype(int).to_numpy()
            probs = final_pre.predict_proba(X_eval)[:, 1]
            print(f"Holdout OOS AUC ({args.eval_start}+): {roc_auc_score(y_eval, probs):.3f} "
                  f"({len(eval_df)} rows)")

    from sklearn.ensemble import HistGradientBoostingClassifier

    final = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05)
    final.fit(X, y)
    runtime_names = [c.removeprefix("feat_") for c in feature_cols]
    MetaLabelModel(final, runtime_names).save(args.out)
    print(f"Model saved to {args.out}")
    print("Deploy in SHADOW mode first (ml.meta_filter.mode: shadow).")


if __name__ == "__main__":
    main()
