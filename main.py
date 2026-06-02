#!/usr/bin/env python3
"""daytrader-bot entry point.

Usage:
    python main.py                 # run the long-lived scheduled service
    python main.py --once research # run pre-market research once and exit
    python main.py --once trade    # run a single trading cycle and exit
    python main.py --once report   # generate today's report and exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from a checkout (src/ layout) without install.
SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daytrader.app import build_application  # noqa: E402
from daytrader import scheduler as scheduler_mod  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="daytrader-bot")
    parser.add_argument(
        "--once",
        choices=["research", "trade", "report"],
        help="Run a single stage once and exit (otherwise run the scheduler).",
    )
    args = parser.parse_args()

    app = build_application()

    if args.once == "research":
        app.premarket_research()
    elif args.once == "trade":
        app.on_start()
        app.trading_cycle()
    elif args.once == "report":
        app.market_close()
    else:
        scheduler_mod.run(app)


if __name__ == "__main__":
    main()
