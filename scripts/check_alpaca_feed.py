#!/usr/bin/env python3
"""Probe Alpaca market-data feed access (IEX vs SIP).

Usage:
    python scripts/check_alpaca_feed.py
    python scripts/check_alpaca_feed.py --symbol NVDA
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daytrader.config.settings import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check Alpaca IEX vs SIP feed entitlement")
    p.add_argument("--symbol", default="TSLA", help="symbol to probe (default: TSLA)")
    p.add_argument("--days", type=int, default=10, help="lookback days for daily bar probe")
    return p.parse_args()


def probe_feed(
    api_key: str,
    secret_key: str,
    symbol: str,
    feed: str,
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[str, str]:
    """Return (status, detail) where status is OK | EMPTY | DENIED | ERROR."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(api_key, secret_key)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start,
            end=end,
            limit=5,
            feed=feed,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return "EMPTY", "no bars returned"
        sub = df.xs(symbol, level=0) if isinstance(df.index, type(df.index)) and df.index.nlevels > 1 else df
        last = sub.iloc[-1]
        vol = int(last["volume"])
        close = float(last["close"])
        return "OK", f"bars={len(sub)} close={close:.2f} vol={vol:,}"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "403" in msg or "subscription" in msg.lower() or "does not permit" in msg.lower():
            return "DENIED", msg.split("\n")[0][:200]
        return "ERROR", f"{type(exc).__name__}: {msg.split(chr(10))[0][:200]}"


def main() -> int:
    args = parse_args()
    settings = get_settings()
    key = settings.secrets.alpaca_api_key
    secret = settings.secrets.alpaca_secret_key
    if not key or not secret:
        print("Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        return 2

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=args.days)
    symbol = args.symbol.upper()

    print(f"Probing Alpaca feeds for {symbol} ({args.days}d lookback)")
    results: dict[str, tuple[str, str]] = {}
    for feed in ("iex", "sip"):
        status, detail = probe_feed(key, secret, symbol, feed, start, end)
        results[feed] = (status, detail)
        print(f"  {feed.upper():4}  {status:6}  {detail}")

    iex_ok = results["iex"][0] == "OK"
    sip_ok = results["sip"][0] == "OK"

    print()
    if sip_ok:
        print("SIP: entitled — set data.feed: sip in config to match backtest volume.")
        if iex_ok:
            iex_vol = results["iex"][1].split("vol=")[-1]
            sip_vol = results["sip"][1].split("vol=")[-1]
            print(f"     Volume ratio (SIP/IEX last bar): compare {sip_vol} vs {iex_vol}")
    else:
        print("SIP: NOT entitled on this Alpaca account (403 / subscription required).")
        print("     Stay on feed: iex, or upgrade Alpaca market data (Algo Trader Plus).")
        if iex_ok:
            print("     IEX works — rescale research.filters.min_avg_daily_volume for IEX volume.")

    return 0 if iex_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
