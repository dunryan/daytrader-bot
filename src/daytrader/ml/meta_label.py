"""Meta-label signal filter.

A secondary ML quality gate over primary strategy signals (meta-labeling):
the strategies decide *direction*, the classifier estimates *P(profitable)*
from the signal's own features, and low-probability signals are filtered out.

Lifecycle:
* Train offline from backtest label exports (``scripts/train_meta_model.py``).
* Deploy in ``shadow`` mode first: every signal is scored and the would-block
  decisions are logged, but nothing is filtered. Promote to ``enforce`` only
  after the shadow cohort shows the filter earns its keep.
* With no model file on disk the filter is inert regardless of mode.

scikit-learn is imported lazily so the trading service runs without it until
a model actually exists.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

from daytrader.config.settings import MetaFilterConfig, Settings
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ── feature encoding (shared by training and inference) ───────
def encode_features(
    strategy: str,
    direction: str,
    confidence: float | None,
    indicators: dict[str, Any],
) -> dict[str, float]:
    """Symbol-agnostic feature vector for one signal.

    Absolute price levels (close, VWAP, EMAs, OR bounds) are excluded — only
    normalized/relative quantities generalize across symbols.
    """
    feats: dict[str, float] = {"confidence": float(confidence or 0.0)}

    close = indicators.get("close")
    atr = indicators.get("atr")
    try:
        if atr is not None and close:
            feats["atr_pct"] = float(atr) / float(close) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    for key in (
        "relative_volume",
        "dist_vwap_atr",
        "or_bars",
        "atr_percentile",
        "range_extension",
        "vwap_one_sidedness",
        "gap_pct",
        "gap_norm",
        "gap_direction",
        "vix_prior",
        "vix_percentile",
    ):
        value = indicators.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            feats[key] = value

    feats[f"strategy={strategy}"] = 1.0
    feats[f"direction={direction}"] = 1.0
    regime = indicators.get("regime")
    if regime:
        feats[f"regime={regime}"] = 1.0
    return feats


class MetaLabelModel:
    """A trained classifier plus the feature columns it expects."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names

    def predict_proba(self, feats: dict[str, float]) -> float:
        import numpy as np

        row = np.array(
            [[feats.get(name, np.nan) for name in self.feature_names]], dtype=float
        )
        return float(self.model.predict_proba(row)[0, 1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"model": self.model, "features": self.feature_names}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "MetaLabelModel | None":
        path = Path(path)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return cls(payload["model"], payload["features"])


class SignalFilter:
    """Runtime filter applying a :class:`MetaLabelModel` to signals."""

    def __init__(self, config: MetaFilterConfig, model: MetaLabelModel | None = None) -> None:
        self.config = config
        self.model = model
        if config.mode != "off" and model is None:
            logger.warning(
                "Meta filter mode is %r but no model found at %s — filter inert.",
                config.mode, config.model_path,
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> "SignalFilter":
        cfg = settings.ml.meta_filter
        model = None
        if cfg.mode != "off":
            try:
                model = MetaLabelModel.load(cfg.model_path)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load meta-label model from %s", cfg.model_path)
        return cls(cfg, model)

    @property
    def active(self) -> bool:
        return self.config.mode != "off" and self.model is not None

    def score(self, signal) -> float | None:  # noqa: ANN001 - Signal (avoids import cycle)
        """Score one signal, annotate ``indicators['meta_prob']``, return prob."""
        if not self.active:
            return None
        try:
            feats = encode_features(
                signal.strategy, signal.direction.value, signal.confidence, signal.indicators
            )
            prob = self.model.predict_proba(feats)
        except Exception:  # noqa: BLE001
            logger.exception("Meta scoring failed for %s/%s", signal.symbol, signal.strategy)
            return None
        signal.indicators["meta_prob"] = round(prob, 4)
        return prob

    def passes(self, signal) -> bool:  # noqa: ANN001
        """Whether the signal survives the filter (shadow mode always passes)."""
        prob = self.score(signal)
        if prob is None:
            return True
        below = prob < self.config.threshold
        if below and self.config.mode == "shadow":
            logger.info(
                "META SHADOW: would block %s %s/%s (P=%.3f < %.2f)",
                signal.direction.value, signal.symbol, signal.strategy,
                prob, self.config.threshold,
            )
            return True
        if below:
            logger.info(
                "META BLOCK: %s %s/%s (P=%.3f < %.2f)",
                signal.direction.value, signal.symbol, signal.strategy,
                prob, self.config.threshold,
            )
            return False
        return True
