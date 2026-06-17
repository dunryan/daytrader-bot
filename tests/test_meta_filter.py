"""Tests for the meta-label SignalFilter (off/shadow/enforce + feature encoding)."""

from __future__ import annotations

from daytrader.config.settings import MetaFilterConfig
from daytrader.ml.meta_label import MetaLabelModel, SignalFilter, encode_features
from daytrader.strategy.base import Direction, Signal


class _StubModel:
    """Deterministic stand-in for a trained classifier."""

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def predict_proba(self, rows):  # noqa: ANN001
        import numpy as np

        return np.array([[1.0 - self.prob, self.prob]] * len(rows))


def _signal(confidence: float = 0.8) -> Signal:
    return Signal(
        symbol="AAPL", strategy="vwap_pullback", direction=Direction.BUY,
        price=100.0, confidence=confidence, rationale="test", timeframe="5m",
        indicators={"atr": 1.0, "close": 100.0, "dist_vwap_atr": 0.2, "regime": "balanced"},
    )


def _filter(mode: str, prob: float, threshold: float = 0.6) -> SignalFilter:
    cfg = MetaFilterConfig(mode=mode, threshold=threshold)
    model = MetaLabelModel(_StubModel(prob), ["confidence", "atr_pct", "dist_vwap_atr"])
    return SignalFilter(cfg, model)


def test_encode_features_excludes_absolute_prices():
    feats = encode_features("orb", "BUY", 0.7, {
        "close": 412.5, "or_high": 415.0, "or_low": 410.0, "atr": 2.0,
        "relative_volume": 1.8, "or_bars": 3,
    })
    assert "close" not in feats and "or_high" not in feats
    assert round(feats["atr_pct"], 4) == round(2.0 / 412.5 * 100, 4)
    assert feats["relative_volume"] == 1.8
    assert feats["strategy=orb"] == 1.0
    assert feats["direction=BUY"] == 1.0


def test_off_mode_is_inert():
    f = SignalFilter(MetaFilterConfig(mode="off"), model=None)
    sig = _signal()
    assert f.passes(sig) is True
    assert "meta_prob" not in sig.indicators


def test_missing_model_is_inert_even_in_enforce():
    f = SignalFilter(MetaFilterConfig(mode="enforce", threshold=0.99), model=None)
    assert f.passes(_signal()) is True


def test_shadow_mode_scores_but_never_blocks():
    f = _filter("shadow", prob=0.2)
    sig = _signal()
    assert f.passes(sig) is True
    assert sig.indicators["meta_prob"] == 0.2


def test_enforce_blocks_below_threshold():
    f = _filter("enforce", prob=0.4, threshold=0.6)
    assert f.passes(_signal()) is False


def test_enforce_passes_above_threshold():
    f = _filter("enforce", prob=0.75, threshold=0.6)
    sig = _signal()
    assert f.passes(sig) is True
    assert sig.indicators["meta_prob"] == 0.75


def test_model_save_load_roundtrip(tmp_path):
    model = MetaLabelModel(_StubModel(0.7), ["confidence"])
    path = tmp_path / "models" / "meta.pkl"
    model.save(path)
    loaded = MetaLabelModel.load(path)
    assert loaded is not None
    assert loaded.feature_names == ["confidence"]
    assert loaded.predict_proba({"confidence": 0.5}) == 0.7


def test_load_missing_model_returns_none(tmp_path):
    assert MetaLabelModel.load(tmp_path / "nope.pkl") is None
