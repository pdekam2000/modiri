"""Stateful risk controls for live trading: the same daily-loss limit and
drawdown kill-switch the backtest engine enforces, but persisted across
polling cycles instead of recomputed from a DataFrame."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class RiskLimits:
    risk_per_trade_pct: float
    max_concurrent_trades: int
    max_daily_loss_pct: float | None
    max_drawdown_pct: float | None


class RiskManager:
    def __init__(self, limits: RiskLimits, starting_equity: float):
        self.limits = limits
        self.equity_peak = starting_equity
        self._current_day: date | None = None
        self._day_start_equity = starting_equity
        self._halted_for_good = False

    def _roll_day_if_needed(self, today: date, equity: float) -> None:
        if today != self._current_day:
            self._current_day = today
            self._day_start_equity = equity

    def update_equity(self, equity: float, today: date) -> None:
        self._roll_day_if_needed(today, equity)
        self.equity_peak = max(self.equity_peak, equity)
        if self.limits.max_drawdown_pct is not None and self.equity_peak > 0:
            drawdown_pct = (self.equity_peak - equity) / self.equity_peak * 100.0
            if drawdown_pct >= self.limits.max_drawdown_pct:
                self._halted_for_good = True

    def can_open_new_trade(self, equity: float, open_trade_count: int) -> tuple[bool, str]:
        if self._halted_for_good:
            return False, "drawdown kill-switch triggered — trading halted until manual reset"
        if open_trade_count >= self.limits.max_concurrent_trades:
            return False, "max concurrent trades reached"
        if self.limits.max_daily_loss_pct is not None and self._day_start_equity > 0:
            day_loss_pct = (self._day_start_equity - equity) / self._day_start_equity * 100.0
            if day_loss_pct >= self.limits.max_daily_loss_pct:
                return False, "daily loss limit reached"
        return True, "ok"

    def reset_kill_switch(self) -> None:
        """Manual override after a human has reviewed why the drawdown
        limit tripped. Never call this automatically."""
        self._halted_for_good = False
        self.equity_peak = self._day_start_equity
