"""Resolve the configured universe into a concrete list of symbols.

Supports:
* An explicit list of tickers in config (used verbatim).
* Named universes ``sp500`` / ``nasdaq100`` backed by a *static snapshot*
  shipped below.

NOTE: the named-universe lists are a curated static snapshot (large-cap, liquid
names), not a live index membership feed. For production accuracy, replace
:func:`_load_named_universe` with a maintained constituents source. This keeps
Module 1 fully functional and dependency-free in the meantime.
"""

from __future__ import annotations

from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Curated, liquid large-caps. Deliberately a representative subset, not the full
# index. Replace with a maintained source for exact membership.
_SP500_SNAPSHOT = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "AVGO", "HD", "CVX", "MRK",
    "ABBV", "COST", "PEP", "ADBE", "KO", "BAC", "CRM", "MCD", "AMD", "NFLX",
    "TMO", "CSCO", "ACN", "ABT", "LIN", "DIS", "WFC", "DHR", "INTC", "VZ",
    "TXN", "PM", "QCOM", "INTU", "NKE", "AMGN", "IBM", "GE", "CAT", "BA",
]

_NASDAQ100_SNAPSHOT = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "PEP", "COST",
    "ADBE", "NFLX", "AMD", "CSCO", "INTC", "QCOM", "INTU", "TXN", "AMGN", "HON",
    "BKNG", "SBUX", "GILD", "MDLZ", "ADI", "PYPL", "REGN", "VRTX", "LRCX", "MU",
]

_NAMED = {
    "sp500": _SP500_SNAPSHOT,
    "nasdaq100": _NASDAQ100_SNAPSHOT,
}


def resolve_universe(universe: list[str]) -> list[str]:
    """Expand a config ``universe`` value into a deduped symbol list.

    Entries matching a named universe (case-insensitive) expand to its
    snapshot; any other entry is treated as a literal ticker symbol.
    """
    symbols: list[str] = []
    seen: set[str] = set()
    for entry in universe:
        key = entry.strip().lower()
        members = _NAMED.get(key)
        if members is not None:
            candidates = members
        else:
            candidates = [entry.strip().upper()]
        for sym in candidates:
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
    logger.info("Resolved universe %s -> %d symbols", universe, len(symbols))
    return symbols
