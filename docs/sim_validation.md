# Paper sim validation (Phase 0)

Single reference for comparing live VirtualBroker sim to backtest. Run weekly:

```bash
python scripts/sim_gate_status.py
python scripts/sim_gate_status.py --since 2026-06-09   # optional: post-deploy cutoff
```

---

## Backtest baseline (SIP, structural stack)

Config: core-10, gap-days, regime enforce, ORB only, 90min window, `min_breakout_rvol: 1.75`.

| Metric | Value |
|--------|-------|
| Window | 2025-01-02 → 2026-06-05 |
| Trades | 103 |
| PF (clean, 4bps / 0.05 atr) | 1.16 |
| PF (stress, 8bps / 0.10 atr) | 1.00 |
| Expectancy (clean) | ~+$27/trade |
| Pace | ~6 trades/month (103 ÷ 17 mo) |

**Phase 0 pass criteria**

| Gate | Threshold |
|------|-----------|
| Min closed sim trades | ≥ 30 |
| Sim PF vs backtest | within 0.15 of 1.16 → **≥ 1.01** |
| Screener | No systematic research/watchlist failures |
| Meta / PF gate | Shadow only — log, do not enforce |

---

## Frequency anchor (`--premarket-rvol`)

Command (default **TOD** mode — premarket vol vs avg premarket at 07:00):

```bash
python backtest.py \
  --symbols COIN,SMCI,PLTR,TSLA,AMD,MU,NVDA,AVGO,MSTR,META \
  --start 2025-01-02 --end 2026-06-05 \
  --gap-days-only --premarket-rvol \
  --regime enforce --vol-gate off \
  --strategies opening_range_breakout \
  --orb-entry-window 90 --orb-min-breakout-rvol 1.75 \
  --feed sip --cache-dir data/backtest_cache_sip
```

> **Live screener (fixed):** uses the same TOD-normalized premarket RVOL as backtest
> `tod` mode — premarket vol @ 07:00 vs trailing avg premarket vol, not vs full daily avg.

### Measured (SIP, 2025-01-02 → 2026-06-05)

| Metric | Gap-only + RVOL 1.75 | + Premarket RVOL (TOD) |
|--------|----------------------|-------------------------|
| Eligible symbol-days | 1,551 | **339** |
| Trades | 103 | **63** |
| PF (clean) | 1.16 | 0.93 |
| Expectancy | +$27 | −$15 |

**Live comparison:** watchlist size × gap-day frequency should sit between the
gap-only row (upper bound — backtest always trades the fixed universe) and the
premarket-TOD row (closer to screener-selected days). Target ~**4 trades/month**
(63 ÷ 17 mo) if premarket RVOL is doing real work in sim.

---

## Weekly checklist (Fridays)

1. `python scripts/sim_gate_status.py` on the sim host (or copy `trading.db` locally).
2. Confirm watchlist non-empty on gap days; check logs for research errors.
3. Do **not** tune parameters until Phase 0 passes.
4. When both gates pass, update `docs/action_plan.txt` Phase 0 checkboxes and review Phase 5 (#15 SIP, #16 Alpaca paper).
