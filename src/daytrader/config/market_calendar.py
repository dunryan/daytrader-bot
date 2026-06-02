"""Market calendar / clock helpers.

Wraps ``pandas_market_calendars`` (NYSE) to answer "is the market open?",
"what is today's session?", and to expose timezone-aware *now*. Falls back to
a simple weekday/After-hours heuristic if the calendar package is unavailable,
so the rest of the system keeps working in minimal environments.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)

try:  # optional dependency, graceful fallback
    import pandas_market_calendars as mcal

    _HAS_MCAL = True
except Exception:  # noqa: BLE001
    _HAS_MCAL = False


@dataclass(frozen=True)
class Session:
    """A single trading session's open/close in the configured timezone."""

    date: dt.date
    open: dt.datetime
    close: dt.datetime

    def contains(self, moment: dt.datetime) -> bool:
        return self.open <= moment <= self.close


class MarketClock:
    """Timezone-aware market clock for the NYSE calendar."""

    def __init__(self, timezone: str = "America/New_York", exchange: str = "NYSE") -> None:
        self.tz = ZoneInfo(timezone)
        self.exchange = exchange
        self._calendar = mcal.get_calendar(exchange) if _HAS_MCAL else None
        if not _HAS_MCAL:
            logger.warning(
                "pandas_market_calendars not installed; using weekday heuristic "
                "for market hours (holidays will not be detected)."
            )

    def now(self) -> dt.datetime:
        """Current time in the market timezone."""
        return dt.datetime.now(self.tz)

    def session_for(self, date: dt.date | None = None) -> Session | None:
        """Return the trading session for ``date`` (today if None).

        Returns ``None`` when the market is closed that day (weekend/holiday).
        """
        date = date or self.now().date()
        if self._calendar is not None:
            sched = self._calendar.schedule(start_date=date, end_date=date)
            if sched.empty:
                return None
            row = sched.iloc[0]
            return Session(
                date=date,
                open=row["market_open"].tz_convert(self.tz).to_pydatetime(),
                close=row["market_close"].tz_convert(self.tz).to_pydatetime(),
            )
        # Fallback: Mon-Fri, 09:30-16:00 local, no holiday awareness.
        if date.weekday() >= 5:
            return None
        return Session(
            date=date,
            open=dt.datetime.combine(date, dt.time(9, 30), tzinfo=self.tz),
            close=dt.datetime.combine(date, dt.time(16, 0), tzinfo=self.tz),
        )

    def is_trading_day(self, date: dt.date | None = None) -> bool:
        return self.session_for(date) is not None

    def is_market_open(self, moment: dt.datetime | None = None) -> bool:
        moment = moment or self.now()
        session = self.session_for(moment.date())
        return session is not None and session.contains(moment)
