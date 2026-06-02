"""APScheduler wiring for the 'set and forget' daily lifecycle.

Registers three kinds of jobs (all in the configured market timezone):
* pre-market research once each weekday morning,
* a recurring trading cycle on ``poll_interval_seconds`` (a no-op outside RTH),
* market-close flatten + report.

Runs in a single long-lived process (systemd service). The scheduler itself is
resilient: individual job failures are logged by the Application stage methods.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from daytrader.app import Application
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _hhmm(value: str) -> tuple[int, int]:
    hh, mm = value.split(":")
    return int(hh), int(mm)


def build_scheduler(app: Application) -> BlockingScheduler:
    tz = ZoneInfo(app.settings.app.timezone)
    sched = BlockingScheduler(timezone=tz)
    schedule = app.settings.schedule

    pm_h, pm_m = _hhmm(schedule.premarket_research)
    sched.add_job(
        app.premarket_research, CronTrigger(day_of_week="mon-fri", hour=pm_h, minute=pm_m, timezone=tz),
        id="premarket_research", name="Pre-market research", misfire_grace_time=3600,
    )

    sched.add_job(
        app.trading_cycle,
        IntervalTrigger(seconds=app.settings.data.poll_interval_seconds, timezone=tz),
        id="trading_cycle", name="Trading cycle", max_instances=1, coalesce=True,
    )

    rc_h, rc_m = _hhmm(schedule.report_time)
    sched.add_job(
        app.market_close, CronTrigger(day_of_week="mon-fri", hour=rc_h, minute=rc_m, timezone=tz),
        id="market_close", name="Market close + report", misfire_grace_time=3600,
    )

    logger.info(
        "Scheduler configured: research @%s, cycle every %ss, report @%s (%s)",
        schedule.premarket_research, app.settings.data.poll_interval_seconds,
        schedule.report_time, app.settings.app.timezone,
    )
    return sched


def run(app: Application) -> None:
    app.on_start()
    sched = build_scheduler(app)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
