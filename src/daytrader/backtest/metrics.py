"""Backtest performance metrics: expectancy, profit factor, Sharpe/Sortino,
max drawdown, per-strategy and per-period breakdowns.

Per-period stability matters more than the headline number: an edge that
lives entirely in one month is curve-fit, not edge.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from daytrader.backtest.engine import BtTrade

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(trades: list[BtTrade], equity_curve: pd.Series) -> dict[str, Any]:
    """Headline statistics for a set of round-trips and a daily equity curve."""
    out: dict[str, Any] = {
        "total_trades": len(trades),
        "expectancy": None, "win_rate": None, "profit_factor": None,
        "avg_win": None, "avg_loss": None, "payoff_ratio": None,
        "sharpe": None, "sortino": None, "max_drawdown_pct": None,
        "total_return_pct": None,
    }
    if trades:
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        out["expectancy"] = sum(pnls) / len(pnls)
        out["win_rate"] = len(wins) / len(pnls)
        out["avg_win"] = sum(wins) / len(wins) if wins else 0.0
        out["avg_loss"] = sum(losses) / len(losses) if losses else 0.0
        if losses:
            out["profit_factor"] = sum(wins) / abs(sum(losses)) if wins else 0.0
            if wins:
                out["payoff_ratio"] = out["avg_win"] / abs(out["avg_loss"])

    if len(equity_curve) >= 2:
        eq = equity_curve.sort_index()
        returns = eq.pct_change().dropna()
        if len(returns) >= 2 and returns.std() > 0:
            out["sharpe"] = float(
                returns.mean() / returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
            )
            downside = returns[returns < 0]
            if len(downside) >= 2 and downside.std() > 0:
                out["sortino"] = float(
                    returns.mean() / downside.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
                )
        running_peak = eq.cummax()
        drawdowns = (running_peak - eq) / running_peak * 100.0
        out["max_drawdown_pct"] = float(drawdowns.max())
        out["total_return_pct"] = float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0)

    return out


def metrics_by_strategy(trades: list[BtTrade]) -> dict[str, dict[str, Any]]:
    """Per-strategy breakdown (each strategy scored on its own trades)."""
    by_strategy: dict[str, list[BtTrade]] = {}
    for t in trades:
        by_strategy.setdefault(t.strategy or "unknown", []).append(t)
    return {
        name: compute_metrics(rows, pd.Series(dtype=float))
        for name, rows in sorted(by_strategy.items())
    }


def metrics_by_period(
    trades: list[BtTrade], equity_curve: pd.Series, freq: str = "M"
) -> dict[str, dict[str, Any]]:
    """Walk-forward-style stability report: metrics per calendar period.

    ``freq`` is a pandas *Period* frequency ("M" = month, "W" = week).
    """
    if not trades:
        return {}
    # Strip tz before Period conversion (Period is tz-naive by definition).
    exit_times = pd.to_datetime(
        [pd.Timestamp(t.exit_time) for t in trades], utc=True
    ).tz_localize(None)
    frame = pd.DataFrame({"exit_time": exit_times, "trade": trades})
    out: dict[str, dict[str, Any]] = {}
    eq = equity_curve.sort_index() if len(equity_curve) else pd.Series(dtype=float)
    eq_tz = getattr(eq.index, "tz", None) if len(eq) else None
    for period, group in frame.groupby(frame["exit_time"].dt.to_period(freq)):
        period_eq = pd.Series(dtype=float)
        if len(eq):
            start, end = period.start_time, period.end_time
            if eq_tz is not None:
                start, end = start.tz_localize(eq_tz), end.tz_localize(eq_tz)
            period_eq = eq[(eq.index >= start) & (eq.index <= end)]
        out[str(period)] = compute_metrics(list(group["trade"]), period_eq)
    return out


def format_report(
    overall: dict[str, Any],
    per_strategy: dict[str, dict[str, Any]],
    per_period: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Human-readable console report."""
    def fmt(value: Any, pct: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%" if pct else f"{value:.2f}"

    lines = ["", "=" * 64, "BACKTEST RESULTS", "=" * 64]
    lines.append(
        f"trades={overall['total_trades']}  win_rate={fmt(overall['win_rate'], pct=True)}  "
        f"PF={fmt(overall['profit_factor'])}  expectancy=${fmt(overall['expectancy'])}"
    )
    lines.append(
        f"payoff={fmt(overall['payoff_ratio'])}  sharpe={fmt(overall['sharpe'])}  "
        f"sortino={fmt(overall['sortino'])}  maxDD={fmt(overall['max_drawdown_pct'])}%  "
        f"return={fmt(overall['total_return_pct'])}%"
    )
    if per_strategy:
        lines.append("-" * 64)
        lines.append("Per strategy:")
        for name, m in per_strategy.items():
            lines.append(
                f"  {name:<26} trades={m['total_trades']:<4} "
                f"win={fmt(m['win_rate'], pct=True):<7} PF={fmt(m['profit_factor']):<6} "
                f"exp=${fmt(m['expectancy'])}"
            )
    if per_period:
        lines.append("-" * 64)
        lines.append("Per period (stability check — beware single-period edges):")
        for period, m in per_period.items():
            lines.append(
                f"  {period:<10} trades={m['total_trades']:<4} "
                f"win={fmt(m['win_rate'], pct=True):<7} PF={fmt(m['profit_factor']):<6} "
                f"exp=${fmt(m['expectancy'])}"
            )
    lines.append("=" * 64)
    return "\n".join(lines)
