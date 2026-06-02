"""Module 5: persistence layer (SQLAlchemy over SQLite).

The database is the single source of truth and survives restarts. Models map
1:1 to the schema in the README; repositories provide typed CRUD so no other
module writes raw SQL.
"""

from daytrader.persistence.database import Database
from daytrader.persistence.models import (
    Base,
    DailyMetric,
    EquityPoint,
    Fill,
    Order,
    Position,
    Scan,
    Signal,
    WatchlistItem,
)

__all__ = [
    "Database",
    "Base",
    "WatchlistItem",
    "Scan",
    "Signal",
    "Order",
    "Fill",
    "Position",
    "DailyMetric",
    "EquityPoint",
]
