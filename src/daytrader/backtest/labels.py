"""Export labeled signal datasets for meta-label model training.

Each row is one *filled* signal from a backtest replay: feature columns
(prefixed ``feat_``, produced by the same encoder used at inference time) plus
a binary ``label`` (1 = the resulting round-trip was profitable). Unfilled
signals carry no outcome and are excluded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from daytrader.backtest.engine import SignalEvent
from daytrader.ml.meta_label import encode_features
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


def signals_to_frame(signals: list[SignalEvent]) -> pd.DataFrame:
    """Flatten labeled signal events into a training DataFrame."""
    rows: list[dict] = []
    for ev in signals:
        if not ev.filled or ev.label is None:
            continue
        feats = encode_features(ev.strategy, ev.direction, ev.confidence, ev.indicators)
        row = {
            "trade_date": ev.trade_date,
            "timestamp": ev.timestamp,
            "exit_time": ev.exit_time,
            "symbol": ev.symbol,
            "strategy": ev.strategy,
            "direction": ev.direction,
            "label": int(ev.label),
        }
        row.update({f"feat_{k}": v for k, v in feats.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    # Missing features (strategy-specific indicators) stay NaN; the
    # HistGradientBoosting trainer handles NaN natively.
    return df


def export_labels(signals: list[SignalEvent], path: str | Path) -> int:
    """Write the labeled dataset to parquet (or csv by extension)."""
    df = signals_to_frame(signals)
    if df.empty:
        logger.warning("No labeled signals to export (no filled trades?).")
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    logger.info("Exported %d labeled signals to %s", len(df), path)
    return len(df)
