"""Performance metrics computed from a BacktestResult."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestResult


@dataclass
class Metrics:
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    win_rate_pct: float
    profit_factor: float
    num_trades: int
    avg_trade_pnl: float
    monthly_returns_pct: pd.Series

    def summary(self) -> str:
        return (
            f"Total return:     {self.total_return_pct:+.2f}%\n"
            f"CAGR:             {self.cagr_pct:+.2f}%\n"
            f"Max drawdown:     {self.max_drawdown_pct:.2f}%\n"
            f"Sharpe ratio:     {self.sharpe:.2f}\n"
            f"Sortino ratio:    {self.sortino:.2f}\n"
            f"Calmar ratio:     {self.calmar:.2f}\n"
            f"Win rate:         {self.win_rate_pct:.1f}%\n"
            f"Profit factor:    {self.profit_factor:.2f}\n"
            f"Number of trades: {self.num_trades}\n"
            f"Avg trade P&L:    {self.avg_trade_pnl:+.2f}"
        )


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    median_delta = np.median(np.diff(index.values)).astype("timedelta64[s]").astype(float)
    if median_delta <= 0:
        return 252.0
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / median_delta


def compute_metrics(result: BacktestResult, risk_free_rate: float = 0.0) -> Metrics:
    equity = result.equity_curve
    initial = equity.iloc[0] if len(equity) else 0.0
    final = equity.iloc[-1] if len(equity) else 0.0

    total_return_pct = (final / initial - 1.0) * 100.0 if initial else 0.0

    periods_per_year = _periods_per_year(equity.index)
    n_periods = len(equity)
    years = n_periods / periods_per_year if periods_per_year else 0.0
    if years > 0 and initial > 0 and final > 0:
        cagr_pct = ((final / initial) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = total_return_pct

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, np.nan)
    max_drawdown_pct = abs(drawdown.min()) * 100.0 if len(drawdown) else 0.0

    period_returns = equity.pct_change().dropna()
    rf_per_period = risk_free_rate / periods_per_year if periods_per_year else 0.0
    excess = period_returns - rf_per_period

    if excess.std(ddof=0) > 0:
        sharpe = (excess.mean() / excess.std(ddof=0)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    downside = excess[excess < 0]
    if len(downside) and downside.std(ddof=0) > 0:
        sortino = (excess.mean() / downside.std(ddof=0)) * np.sqrt(periods_per_year)
    else:
        sortino = 0.0

    calmar = (cagr_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

    trades_df = result.trades_df()
    num_trades = len(trades_df)
    if num_trades:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate_pct = len(wins) / num_trades * 100.0
        gross_profit = wins["pnl"].sum()
        gross_loss = -losses["pnl"].sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        avg_trade_pnl = trades_df["pnl"].mean()
    else:
        win_rate_pct = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0

    monthly = equity.resample("ME").last().pct_change().dropna() * 100.0
    monthly.name = "monthly_return_pct"

    return Metrics(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        num_trades=num_trades,
        avg_trade_pnl=avg_trade_pnl,
        monthly_returns_pct=monthly,
    )
