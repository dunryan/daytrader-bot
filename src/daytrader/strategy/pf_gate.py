"""Rolling profit-factor deployment gate.

Blocks *new* entries when recent closed-trade performance falls below a
threshold (e.g. Aug-2025-style chop). Open positions are still managed
elsewhere — this gate only vetoes signal generation / entries.

Evaluated at the start of each session from the last ``lookback_trades``
closed round-trips (causal: no same-day peeking).
"""

from __future__ import annotations

from daytrader.config.settings import PfGateConfig
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


def profit_factor(pnls: list[float]) -> float:
    """Gross wins / gross losses; inf when no losses, 1.0 when flat."""
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p <= 0))
    if losses == 0:
        return float("inf") if wins > 0 else 1.0
    return wins / losses


def evaluate_pf_gate(
    closed_pnls: list[float],
    config: PfGateConfig,
) -> tuple[bool, dict[str, float]]:
    """Return ``(block_new_entries, diagnostics)`` for the upcoming session."""
    if config.mode == "off":
        return False, {}

    recent = closed_pnls[-config.lookback_trades :]
    n = len(recent)
    details: dict[str, float] = {"pf_gate_trades": float(n)}
    if n < config.min_trades:
        return False, details

    pf = profit_factor(recent)
    details["pf_gate_pf"] = round(pf, 4)
    blocked = pf < config.min_pf
    if blocked and config.mode == "shadow":
        logger.info(
            "PF GATE SHADOW: would block (PF=%.2f < %.2f, n=%d)",
            pf, config.min_pf, n,
        )
        return False, details
    if blocked:
        logger.debug("PF GATE: block (PF=%.2f < %.2f, n=%d)", pf, config.min_pf, n)
    return blocked, details
