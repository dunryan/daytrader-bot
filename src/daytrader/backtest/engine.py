"""Event-driven backtest engine.

Replays enriched intraday bars through the *same* Strategy and RiskManager
code paths used live, with execution realism rules a polling bot cannot beat:

* Signals are evaluated only on **completed** bars (the live loop's
  partial-bar repaint cannot occur here by construction).
* Entries fill at the **next bar's open**, never the signal bar's close.
* Slippage is spread-aware: half the assumed bid/ask spread plus a
  volatility-proportional impact term, applied against the trade.
* Intrabar stop/target resolution uses the bar's high/low, and when both are
  touched in one bar the **stop is assumed to fill first** (conservative).
* The daily-drawdown kill switch and end-of-session flatten mirror live
  behavior.

All indicators in :mod:`daytrader.data.indicators` are causal (recursive
EMAs/Wilder smoothing/cumulative VWAP), so frames can be enriched once and
sliced per bar without look-ahead.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from daytrader.config.settings import Settings
from daytrader.data.data_engine import MarketSnapshot
from daytrader.data.providers.base import Timeframe
from daytrader.execution.broker_base import Side
from daytrader.execution.risk_manager import RiskManager
from daytrader.ml.meta_label import SignalFilter
from daytrader.strategy.base import Direction, Signal, Strategy
from daytrader.strategy.pf_gate import evaluate_pf_gate
from daytrader.strategy.router import StrategyRouter
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  Result value objects
# ════════════════════════════════════════════════════════════
@dataclass
class SignalEvent:
    """An actionable signal observed during replay (filled or not)."""

    timestamp: dt.datetime
    trade_date: str
    symbol: str
    strategy: str
    direction: str
    confidence: float
    indicators: dict[str, Any]
    filled: bool = False
    label: int | None = None  # 1 = profitable round-trip, 0 = not, None = no trade
    exit_time: dt.datetime | None = None


@dataclass
class BtTrade:
    """A completed round-trip."""

    symbol: str
    strategy: str | None
    direction: str
    qty: int
    entry_time: dt.datetime
    entry_price: float
    exit_time: dt.datetime
    exit_price: float
    exit_reason: str
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    starting_cash: float
    trades: list[BtTrade] = field(default_factory=list)
    signals: list[SignalEvent] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def ending_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.starting_cash


# ════════════════════════════════════════════════════════════
#  Internal state
# ════════════════════════════════════════════════════════════
@dataclass
class _Position:
    symbol: str
    side: Side
    qty: int
    entry_price: float
    entry_time: dt.datetime
    stop: float
    take_profit: float | None
    strategy: str | None
    atr_at_entry: float
    initial_risk: float
    signal_idx: int
    peak: float = 0.0

    def __post_init__(self) -> None:
        if not self.peak:
            self.peak = self.entry_price


@dataclass
class _PendingEntry:
    """A signal accepted at bar t, awaiting a fill at bar t+1's open."""

    symbol: str
    side: Side
    strategy: str
    stop_hint: float | None
    target_hint: float | None
    atr: float
    signal_idx: int


def gap_eligible_days(
    daily: pd.DataFrame,
    min_gap_pct: float,
    max_gap_pct: float = 0.0,
    max_gap_norm: float = 0.0,
) -> set[pd.Timestamp]:
    """Sessions whose opening gap qualifies under the screener's threshold.

    Uses only the open vs the prior close — both known at 09:30, so this
    introduces no look-ahead. Deliberately does NOT condition on the day's
    full-session volume (that would be look-ahead; the live screener uses
    pre-market partials, which a daily-bar backtest cannot reproduce).

    ``max_gap_pct`` / ``max_gap_norm`` reject exhaustion gaps (gap-and-crap).
    """
    if daily is None or len(daily) < 2:
        return set()
    d = daily.sort_index()
    prev_close = d["close"].shift(1)
    gap_pct = (d["open"] - prev_close) / prev_close * 100.0
    atr = d["atr"] if "atr" in d.columns else pd.Series([float("nan")] * len(d), index=d.index)
    gap_norm = (d["open"] - prev_close).abs() / atr.replace(0, float("nan"))
    days = pd.Index(d.index).normalize()
    eligible: set[pd.Timestamp] = set()
    for day, gap, norm in zip(days, gap_pct, gap_norm):
        if pd.isna(gap) or abs(gap) < min_gap_pct:
            continue
        if max_gap_pct > 0 and abs(gap) > max_gap_pct:
            continue
        if max_gap_norm > 0 and pd.notna(norm) and norm > max_gap_norm:
            continue
        eligible.add(day)
    return eligible


class BacktestEngine:
    """Replays bars through strategies + risk management."""

    def __init__(
        self,
        settings: Settings,
        strategies: list[Strategy],
        starting_cash: float | None = None,
        spread_bps: float = 4.0,
        atr_impact_coeff: float = 0.05,
        warmup_bars: int = 25,
        signal_filter: SignalFilter | None = None,
    ) -> None:
        self.settings = settings
        self.risk = RiskManager(settings.risk)
        self.router = StrategyRouter(
            strategies,
            signal_repo=None,
            regime_config=settings.strategies.regime_filter,
        )
        self.signal_filter = signal_filter
        self.pf_gate = settings.strategies.pf_gate
        self.starting_cash = starting_cash or settings.risk.starting_equity
        self.spread_bps = spread_bps
        self.atr_impact_coeff = atr_impact_coeff
        self.warmup_bars = warmup_bars

    # ── fills ──────────────────────────────────────────────────
    def _slip_pct(self, atr: float, price: float) -> float:
        """Fractional cost per fill: half-spread + volatility impact."""
        half_spread = self.spread_bps / 2.0 / 10_000.0
        impact = self.atr_impact_coeff * (atr / price) if price > 0 and atr > 0 else 0.0
        return half_spread + impact

    def _fill(self, side: Side, ref_price: float, atr: float) -> float:
        slip = self._slip_pct(atr, ref_price)
        return ref_price * (1.0 + slip) if side is Side.BUY else ref_price * (1.0 - slip)

    # ── main loop ──────────────────────────────────────────────
    def run(
        self,
        intraday: dict[str, pd.DataFrame],
        daily: dict[str, pd.DataFrame],
        timeframe: Timeframe = Timeframe.MIN_5,
        eligible_days: dict[str, set[pd.Timestamp]] | None = None,
        blocked_days: set[pd.Timestamp] | None = None,
    ) -> BacktestResult:
        """Replay ``intraday`` (enriched, RTH-filtered) frames.

        ``daily`` frames provide regime/level context and are truncated to
        strictly *prior* sessions at every step (today's daily bar is future
        information until the close).

        ``eligible_days`` (symbol -> set of normalized session dates)
        restricts *new signal generation* to those sessions, mirroring the
        live screener's day selection. Open positions are still managed on
        every bar regardless.

        ``blocked_days`` suppresses new entries on those session dates (VIX
        gate, macro filters, etc.) while still managing open positions.
        """
        intraday = {s: df for s, df in intraday.items() if df is not None and not df.empty}
        if not intraday:
            return BacktestResult(starting_cash=self.starting_cash)

        # Per-symbol timestamp -> integer position, and session-final bars.
        ts_index: dict[str, dict[pd.Timestamp, int]] = {
            s: {ts: i for i, ts in enumerate(df.index)} for s, df in intraday.items()
        }
        session_last: dict[str, set[pd.Timestamp]] = {
            s: set(df.groupby(pd.Index(df.index).normalize()).tail(1).index)
            for s, df in intraday.items()
        }
        all_ts = sorted({ts for df in intraday.values() for ts in df.index})

        result = BacktestResult(starting_cash=self.starting_cash)
        cash = self.starting_cash
        positions: dict[str, _Position] = {}
        pending: list[_PendingEntry] = []
        last_close: dict[str, float] = {}

        current_day: pd.Timestamp | None = None
        day_peak_equity = self.starting_cash
        halted = False
        pf_blocked_today = False
        closed_pnls: list[float] = []

        def equity() -> float:
            eq = cash
            for pos in positions.values():
                price = last_close.get(pos.symbol, pos.entry_price)
                eq += pos.qty * price * (1 if pos.side is Side.BUY else -1)
            return eq

        def close_position(pos: _Position, price: float, ts: pd.Timestamp, reason: str) -> None:
            nonlocal cash
            exit_price = self._fill(pos.side.opposite, price, pos.atr_at_entry)
            if pos.side is Side.BUY:
                pnl = (exit_price - pos.entry_price) * pos.qty
                cash += exit_price * pos.qty
            else:
                pnl = (pos.entry_price - exit_price) * pos.qty
                cash -= exit_price * pos.qty
            pnl -= self.settings.risk.commission_per_trade
            pnl_pct = pnl / (pos.entry_price * pos.qty) * 100.0 if pos.entry_price else 0.0
            result.trades.append(BtTrade(
                symbol=pos.symbol, strategy=pos.strategy, direction=(
                    "LONG" if pos.side is Side.BUY else "SHORT"
                ),
                qty=pos.qty, entry_time=pos.entry_time, entry_price=pos.entry_price,
                exit_time=ts.to_pydatetime(), exit_price=exit_price, exit_reason=reason,
                pnl=pnl, pnl_pct=pnl_pct,
            ))
            result.signals[pos.signal_idx].label = 1 if pnl > 0 else 0
            result.signals[pos.signal_idx].exit_time = ts.to_pydatetime()
            closed_pnls.append(pnl)
            positions.pop(pos.symbol, None)

        for ts in all_ts:
            day = ts.normalize()
            if day != current_day:
                current_day = day
                day_peak_equity = equity()
                halted = False
                pending.clear()  # overnight signals never carry to the next open
                pf_blocked_today, _ = evaluate_pf_gate(closed_pnls, self.pf_gate)

            # ── 1. fill pending entries at this bar's open ─────
            still_pending: list[_PendingEntry] = []
            for pe in pending:
                i = ts_index[pe.symbol].get(ts)
                if i is None:
                    still_pending.append(pe)
                    continue
                if halted or pe.symbol in positions or not self.risk.can_open_new(len(positions)):
                    continue
                bar = intraday[pe.symbol].iloc[i]
                open_price = float(bar["open"])
                entry = self._fill(pe.side, open_price, pe.atr)
                stop = self.risk.resolve_stop(pe.side, entry, pe.atr, stop_hint=pe.stop_hint)
                if not self.risk._valid_stop(pe.side, entry, stop):
                    continue
                if self.settings.risk.take_profit.method == "trailing":
                    target = None
                else:
                    target = pe.target_hint or self.risk.take_profit(pe.side, entry, stop)
                sizing = self.risk.position_size(equity(), entry, stop, available_cash=cash)
                if sizing.qty <= 0:
                    continue
                if pe.side is Side.BUY:
                    cash -= entry * sizing.qty
                else:
                    cash += entry * sizing.qty
                positions[pe.symbol] = _Position(
                    symbol=pe.symbol, side=pe.side, qty=sizing.qty, entry_price=entry,
                    entry_time=ts.to_pydatetime(), stop=stop, take_profit=target,
                    strategy=pe.strategy, atr_at_entry=pe.atr,
                    initial_risk=abs(entry - stop), signal_idx=pe.signal_idx,
                )
                result.signals[pe.signal_idx].filled = True
            pending = still_pending

            # ── 2. manage open positions on this bar ───────────
            for symbol, pos in list(positions.items()):
                i = ts_index[symbol].get(ts)
                if i is None:
                    continue
                bar = intraday[symbol].iloc[i]
                o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
                last_close[symbol] = c

                exit_price: float | None = None
                reason = ""
                if pos.side is Side.BUY:
                    if o <= pos.stop:
                        exit_price, reason = o, "SL"      # gapped through
                    elif l <= pos.stop:
                        exit_price, reason = pos.stop, "SL"
                    elif pos.take_profit is not None and o >= pos.take_profit:
                        exit_price, reason = o, "TP"
                    elif pos.take_profit is not None and h >= pos.take_profit:
                        exit_price, reason = pos.take_profit, "TP"
                else:
                    if o >= pos.stop:
                        exit_price, reason = o, "SL"
                    elif h >= pos.stop:
                        exit_price, reason = pos.stop, "SL"
                    elif pos.take_profit is not None and o <= pos.take_profit:
                        exit_price, reason = o, "TP"
                    elif pos.take_profit is not None and l <= pos.take_profit:
                        exit_price, reason = pos.take_profit, "TP"

                if exit_price is not None:
                    close_position(pos, exit_price, ts, reason)
                    continue

                # Trail off the intrabar extreme; new stop applies next bar.
                pos.peak = max(pos.peak, h) if pos.side is Side.BUY else min(pos.peak, l)
                pos.stop = self.risk.update_trailing_stop(
                    pos.side, pos.stop, pos.peak, pos.atr_at_entry,
                    entry=pos.entry_price, initial_risk=pos.initial_risk,
                )

            # ── 3. kill switch (mirrors live behavior) ─────────
            eq_now = equity()
            day_peak_equity = max(day_peak_equity, eq_now)
            if not halted and self.risk.is_drawdown_breached(day_peak_equity, eq_now):
                for pos in list(positions.values()):
                    close_position(pos, last_close.get(pos.symbol, pos.entry_price), ts, "KILL_SWITCH")
                halted = True

            # ── 4. evaluate strategies on the completed bar ────
            if (
                not halted
                and not pf_blocked_today
                and (blocked_days is None or day not in blocked_days)
            ):
                for symbol, df in intraday.items():
                    if eligible_days is not None and day not in eligible_days.get(symbol, ()):
                        continue  # screener-style day selection
                    i = ts_index[symbol].get(ts)
                    if i is None or i < self.warmup_bars:
                        continue
                    if ts in session_last[symbol]:
                        continue  # no new entries on the session's final bar
                    snap = MarketSnapshot(symbol=symbol, as_of=ts.to_pydatetime())
                    snap.frames[timeframe] = df.iloc[: i + 1]
                    ddf = daily.get(symbol)
                    if ddf is not None and not ddf.empty:
                        snap.frames[Timeframe.DAY] = ddf[pd.Index(ddf.index).normalize() < day]
                    for sig in self.router.evaluate_snapshot(snap):
                        passed_meta = True
                        if self.signal_filter is not None:
                            passed_meta = self.signal_filter.passes(sig)
                        idx = self._record_signal(result, sig, ts)
                        if not passed_meta:
                            continue
                        if symbol not in positions:
                            pending.append(self._to_pending(sig, idx))

            # ── 5. end-of-session flatten ──────────────────────
            for symbol, pos in list(positions.items()):
                if ts in session_last[symbol] and ts_index[symbol].get(ts) is not None:
                    close_position(pos, last_close.get(symbol, pos.entry_price), ts, "EOD")

            result.equity_curve.loc[day] = equity()

        # Safety: flatten anything left at the very end of the replay.
        final_ts = all_ts[-1]
        for pos in list(positions.values()):
            close_position(pos, last_close.get(pos.symbol, pos.entry_price), final_ts, "EOD")
        if len(all_ts):
            result.equity_curve.loc[final_ts.normalize()] = equity()

        logger.info(
            "Backtest complete: %d trades, %d signals, equity %.2f -> %.2f",
            len(result.trades), len(result.signals),
            self.starting_cash, result.ending_equity,
        )
        return result

    # ── helpers ────────────────────────────────────────────────
    @staticmethod
    def _record_signal(result: BacktestResult, sig: Signal, ts: pd.Timestamp) -> int:
        result.signals.append(SignalEvent(
            timestamp=ts.to_pydatetime(),
            trade_date=str(ts.date()),
            symbol=sig.symbol,
            strategy=sig.strategy,
            direction=sig.direction.value,
            confidence=sig.confidence,
            indicators=dict(sig.indicators),
        ))
        return len(result.signals) - 1

    @staticmethod
    def _to_pending(sig: Signal, signal_idx: int) -> _PendingEntry:
        return _PendingEntry(
            symbol=sig.symbol,
            side=Side.BUY if sig.direction is Direction.BUY else Side.SELL,
            strategy=sig.strategy,
            stop_hint=sig.stop_hint,
            target_hint=sig.target_hint,
            atr=float(sig.indicators.get("atr") or 0.0),
            signal_idx=signal_idx,
        )
