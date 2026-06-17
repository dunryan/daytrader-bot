"""Tests for gap feature helpers."""

from __future__ import annotations

import pandas as pd

from daytrader.strategy.gap_features import (
    compute_gap_features,
    direction_matches_signal,
)


def test_compute_gap_features():
    daily = pd.DataFrame(
        {"open": [100], "close": [100], "atr": [2.0]},
        index=pd.date_range("2026-06-01", periods=1, tz="UTC"),
    )
    feats = compute_gap_features(daily, session_open=103.0)
    assert feats["gap_pct"] == 3.0
    assert feats["gap_norm"] == 1.5
    assert feats["gap_direction"] == 1.0


def test_direction_matches_signal():
    assert direction_matches_signal(1.0, "BUY") is True
    assert direction_matches_signal(1.0, "SELL") is False
    assert direction_matches_signal(-1.0, "SELL") is True
