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
