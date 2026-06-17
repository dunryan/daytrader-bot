# daytrader-bot

A modular, production-grade **intraday day-trading bot** in Python. Runs as a
"set and forget" background service that handles the full daily lifecycle:
pre-market research → live data + technical analysis → strategy signals →
risk-managed execution → end-of-day PDF report emailed to you.

> **Simulation first.** With `app.simulation_mode: true` (the default) the bot
> never touches real capital. It uses Alpaca for *market data* but routes all
> orders through an isolated **VirtualBroker** with virtual balance, slippage,
> and fill tracking.

## Status

Built incrementally, module by module:

| Module | Area | Status |
|-------|------|--------|
| Foundation | config, logging, persistence, market calendar | ✅ |
| 1 | Market Research & Sentiment Engine | ✅ |
| 2 | Data Ingestion & Technical Analysis | ✅ |
| 3 | Strategy & Watchlist Router | ✅ |
| 4 | Risk Management & Execution | ✅ |
| 5 | Performance & State Database | ✅ |
| 6 | Reporting, Plotting & Emailer | ✅ |
| Orchestrator | main.py + APScheduler + systemd/cron | ✅ |
| 7 | Backtester (event-driven replay + metrics + labels) | ✅ |
| 8 | Signal pipeline (regime gate, meta-label filter, ranked allocation) | ✅ |

## Signal pipeline

Every trading cycle runs a layered decision pipeline. Strategies stay pure;
each surrounding layer can veto or shape, never negotiate:

```
watchlist → regime gate → strategies → meta-label filter → ranked allocator
          → risk sizing (1% fixed / capped fractional-Kelly)
          → execution (bracket + collared-limit entries in live mode)
          → management (trail after +1R using ATR-at-entry, 15:55 flatten,
                        peak-to-trough kill switch)
```

- **Regime gate** (`strategies.regime_filter`): classifies each symbol's
  session as `trend | balanced | quiet` and routes which strategies may run
  (breakout in trend, mean-reversion in balanced). Modes: `off`, `shadow`
  (log-only), `enforce`.
- **Meta-label filter** (`ml.meta_filter`): a trained classifier scores each
  signal's P(win); `shadow` scores and logs, `enforce` blocks below the
  threshold. Inert until a model exists at `ml.meta_filter.model_path`.
- **Ranked allocator**: actionable signals are sorted by
  `confidence × expected R:R × meta-probability` — capital goes to the best
  signal, not the first one.
- **Live execution**: entries are marketable **limit** orders with a price
  collar, submitted as bracket/OTO so a protective stop rests at the broker
  immediately; trailing replaces the stop leg. On restart, DB positions are
  reconciled against the broker's actual book.

## Backtesting & promotion gates

Nothing gets enabled in config because it sounds good. The promotion path:

1. **Backtest gate** — `python backtest.py --symbols AAPL,MSFT --start
   2026-01-02 --end 2026-05-30 --walk-forward`. Demand positive expectancy
   after costs, sane profit factor, and per-period stability.
2. **Train the meta filter** — `python backtest.py ... --export-labels
   data/backtest/labels.parquet`, then
   `python scripts/train_meta_model.py --labels data/backtest/labels.parquet`.
   Deploy in `shadow` mode and compare would-block vs would-pass cohorts.
3. **Paper gate** — run with `simulation_mode: true` for weeks; verify live
   fills/slippage/frequency track the backtest.
4. **Live gate** — small size, scaled only while live stats stay inside
   backtest confidence bands.

The replay engine evaluates completed bars only, fills at next-bar open with
spread-aware slippage, resolves intrabar stop/target conservatively
(stop-first), and mirrors the live kill switch and EOD flatten.

## Quickstart

```bash
cd daytrader-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes the `daytrader` package importable

cp .env.example .env        # fill in Alpaca + (optional) news keys
# Edit config/config.yaml to taste (risk limits, strategy toggles, schedule)

# Run the long-lived scheduled service (set & forget):
python main.py

# Or run a single stage once and exit:
python main.py --once research   # build today's watchlist
python main.py --once trade      # one scan/evaluate/execute/manage cycle
python main.py --once report     # generate + email today's report

# Run tests:
pytest -q
```

## Deployment

Recommended: the systemd service in `deploy/daytrader.service` (one warm
process that rehydrates state once and runs the in-process APScheduler).
A `deploy/crontab.example` is provided as an alternative.

```bash
sudo cp deploy/daytrader.service /etc/systemd/system/daytrader.service
# edit User/WorkingDirectory/paths inside the unit first
sudo systemctl daemon-reload && sudo systemctl enable --now daytrader.service
journalctl -u daytrader -f
```

## Configuration

- **`.env`** — secrets only (API keys, SMTP password). Git-ignored.
- **`config/config.yaml`** — all runtime config (risk, strategy toggles,
  schedule, filters). Safe to commit.

Both are loaded and validated at startup by `daytrader.config.settings`.
A bad config fails loudly at boot instead of mid-session.

## Layout

```
src/daytrader/
  config/        settings loader, market calendar
  research/      Module 1: screener + sentiment
  data/          Module 2: providers + indicators
  strategy/      Module 3: strategies + router
  execution/     Module 4: brokers + risk manager
  persistence/   Module 5: SQLAlchemy models + repositories
  reporting/     Module 6: charts + PDF + emailer
  utils/         logging, notifications
```

## Disclaimer

This is software for research and paper trading. Algorithmic trading carries
substantial risk of loss. Nothing here is financial advice. Run in simulation
mode and validate thoroughly before considering any real capital.
