"""Bar-by-bar backtest engine with spread/commission costs, stop-loss /
take-profit, fixed-fractional position sizing, and simple risk controls
(daily loss limit, drawdown kill-switch).

This is deliberately conservative about realism: it charges the full spread
on entry, applies a round-turn commission, and checks stop-loss/take-profit
against each bar's high/low (not just the close), so results are a lot
closer to what a live account would see than a naive close-to-close model.
It is still a backtest, not a guarantee of future performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modiri_bot.risk.position_sizing import lots_for_fixed_risk
from modiri_bot.strategies.base import Strategy


@dataclass
class BacktestConfig:
    initial_balance: float = 1000.0
    pip_size: float = 0.0001
    pip_value_per_lot: float = 10.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float | None = None
    spread_pips: float = 1.2
    commission_per_lot: float = 7.0
    risk_per_trade_pct: float = 1.0
    stop_loss_pips: float = 40.0
    take_profit_pips: float = 80.0
    max_daily_loss_pct: float | None = 3.0
    max_drawdown_pct: float | None = 15.0


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    lots: float
    pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=["entry_time", "exit_time", "side", "entry_price",
                         "exit_price", "lots", "pnl", "exit_reason"]
            )
        return pd.DataFrame([t.__dict__ for t in self.trades])


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        cfg = self.config
        signals = strategy.generate_signals(df).to_numpy()

        idx = df.index
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        n = len(df)

        balance = cfg.initial_balance
        equity_curve = np.empty(n)
        trades: list[Trade] = []

        position = 0          # -1, 0, 1
        entry_price = 0.0
        lots = 0.0
        sl_price = 0.0
        tp_price = 0.0
        entry_time = None

        equity_peak = balance
        trading_halted_for_good = False
        current_day = None
        day_start_balance = balance
        day_trading_blocked = False

        spread_cost = cfg.spread_pips * cfg.pip_size

        def close_position(exit_price: float, exit_time, reason: str) -> None:
            nonlocal balance, position, lots
            direction = 1 if position == 1 else -1
            price_diff = (exit_price - entry_price) * direction
            gross_pnl = (price_diff / cfg.pip_size) * cfg.pip_value_per_lot * lots
            commission = cfg.commission_per_lot * lots
            pnl = gross_pnl - commission
            balance += pnl
            trades.append(
                Trade(entry_time, exit_time, "buy" if position == 1 else "sell",
                      entry_price, exit_price, lots, pnl, reason)
            )
            position = 0
            lots = 0.0

        for i in range(n):
            bar_time = idx[i]
            bar_day = bar_time.date()
            if bar_day != current_day:
                current_day = bar_day
                day_start_balance = balance
                day_trading_blocked = False

            # 1. Manage an existing position: check SL/TP against this bar's range.
            if position != 0:
                if position == 1:
                    hit_sl = l[i] <= sl_price
                    hit_tp = h[i] >= tp_price
                else:
                    hit_sl = h[i] >= sl_price
                    hit_tp = l[i] <= tp_price

                if hit_sl and hit_tp:
                    # Can't know intrabar order for sure; assume the worse outcome (SL) first.
                    close_position(sl_price, bar_time, "stop_loss")
                elif hit_sl:
                    close_position(sl_price, bar_time, "stop_loss")
                elif hit_tp:
                    close_position(tp_price, bar_time, "take_profit")
                elif signals[i] != position:
                    close_position(c[i], bar_time, "signal_flip")

            # 2. Daily loss limit check.
            if cfg.max_daily_loss_pct is not None and day_start_balance > 0:
                day_loss_pct = (day_start_balance - balance) / day_start_balance * 100.0
                if day_loss_pct >= cfg.max_daily_loss_pct:
                    day_trading_blocked = True

            # 3. Drawdown kill-switch (permanent for the rest of the run).
            equity_peak = max(equity_peak, balance)
            if cfg.max_drawdown_pct is not None and equity_peak > 0:
                drawdown_pct = (equity_peak - balance) / equity_peak * 100.0
                if drawdown_pct >= cfg.max_drawdown_pct:
                    trading_halted_for_good = True

            # 4. Open a new position if flat and the strategy wants exposure.
            if (
                position == 0
                and signals[i] != 0
                and not day_trading_blocked
                and not trading_halted_for_good
            ):
                side = signals[i]
                raw_entry = c[i] + spread_cost if side == 1 else c[i] - spread_cost
                lots = lots_for_fixed_risk(
                    equity=balance,
                    risk_per_trade_pct=cfg.risk_per_trade_pct,
                    stop_loss_pips=cfg.stop_loss_pips,
                    pip_value_per_lot=cfg.pip_value_per_lot,
                    min_lot=cfg.min_lot,
                    lot_step=cfg.lot_step,
                    max_lot=cfg.max_lot,
                )
                if lots > 0:
                    position = side
                    entry_price = raw_entry
                    entry_time = bar_time
                    if side == 1:
                        sl_price = entry_price - cfg.stop_loss_pips * cfg.pip_size
                        tp_price = entry_price + cfg.take_profit_pips * cfg.pip_size
                    else:
                        sl_price = entry_price + cfg.stop_loss_pips * cfg.pip_size
                        tp_price = entry_price - cfg.take_profit_pips * cfg.pip_size

            # Mark-to-market equity for the curve.
            if position != 0:
                direction = 1 if position == 1 else -1
                unrealized_pips = ((c[i] - entry_price) * direction) / cfg.pip_size
                equity_curve[i] = balance + unrealized_pips * cfg.pip_value_per_lot * lots
            else:
                equity_curve[i] = balance

        if position != 0:
            close_position(c[-1], idx[-1], "end_of_data")
            equity_curve[-1] = balance

        return BacktestResult(equity_curve=pd.Series(equity_curve, index=idx, name="equity"), trades=trades)
