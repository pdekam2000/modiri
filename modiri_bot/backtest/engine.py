"""Bar-by-bar backtest engine with spread/commission costs, stop-loss /
take-profit, fixed-fractional position sizing, and simple risk controls
(daily loss limit, drawdown kill-switch).

This is deliberately conservative about realism: it charges the full spread
on entry, applies a round-turn commission, and checks stop-loss/take-profit
against each bar's high/low (not just the close), so results are a lot
closer to what a live account would see than a naive close-to-close model.
It is still a backtest, not a guarantee of future performance.

Two optional risk-management upgrades on top of the fixed-pips baseline:
  - ATR-based stops: the stop distance (and therefore position size) adapts
    to current volatility instead of using the same pip count in a quiet
    market and a violent one.
  - Partial ("two-stage") take-profit: close part of the position at a
    closer first target and move the stop to breakeven, letting the rest
    run toward a further target -- instead of one all-or-nothing exit.
Both default to off so existing fixed-pips behaviour is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modiri_bot.risk.position_sizing import lots_for_fixed_risk
from modiri_bot.strategies.base import Strategy
from modiri_bot.strategies.indicators import atr as atr_indicator


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

    # ATR-adaptive stops: when enabled, the stop (and take-profit, and thus
    # position size) is computed from ATR at entry time instead of the
    # fixed stop_loss_pips/take_profit_pips above.
    use_atr_stops: bool = False
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0

    # Two-stage take-profit: close `tp1_close_fraction` of the position at
    # a first target `tp1_r_multiple` stop-distances away, optionally move
    # the stop to breakeven for the remainder, and let the rest run to the
    # normal (fixed-pips or ATR) take-profit level.
    use_partial_tp: bool = False
    tp1_r_multiple: float = 1.0
    tp1_close_fraction: float = 0.5
    move_sl_to_breakeven_after_tp1: bool = True


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

    def run(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        risk_multiplier: pd.Series | None = None,
        stop_multiplier: pd.Series | None = None,
    ) -> BacktestResult:
        """risk_multiplier/stop_multiplier: optional per-bar scale factors
        (aligned to df.index, evaluated at each new position's entry bar)
        applied to risk_per_trade_pct and stop_loss_pips respectively --
        e.g. to size counter-trend trades smaller or give them a tighter
        stop, without excluding them outright. Both default to 1.0
        (no change) when not provided."""
        cfg = self.config
        signals = strategy.generate_signals(df).to_numpy()

        idx = df.index
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        n = len(df)

        risk_mult = risk_multiplier.reindex(idx).fillna(1.0).to_numpy() if risk_multiplier is not None else None
        stop_mult = stop_multiplier.reindex(idx).fillna(1.0).to_numpy() if stop_multiplier is not None else None

        atr_values = (
            atr_indicator(df["high"], df["low"], df["close"], cfg.atr_period).to_numpy()
            if cfg.use_atr_stops else None
        )

        balance = cfg.initial_balance
        equity_curve = np.empty(n)
        trades: list[Trade] = []

        position = 0          # -1, 0, 1
        entry_price = 0.0
        lots = 0.0
        sl_price = 0.0
        tp_price = 0.0
        tp1_price = 0.0
        tp1_done = False
        entry_time = None

        equity_peak = balance
        trading_halted_for_good = False
        current_day = None
        day_start_balance = balance
        day_trading_blocked = False

        spread_cost = cfg.spread_pips * cfg.pip_size

        def close_position(exit_price: float, exit_time, reason: str, close_lots: float | None = None) -> None:
            nonlocal balance, position, lots
            direction = 1 if position == 1 else -1
            realized_lots = lots if close_lots is None else close_lots
            price_diff = (exit_price - entry_price) * direction
            gross_pnl = (price_diff / cfg.pip_size) * cfg.pip_value_per_lot * realized_lots
            commission = cfg.commission_per_lot * realized_lots
            pnl = gross_pnl - commission
            balance += pnl
            trades.append(
                Trade(entry_time, exit_time, "buy" if position == 1 else "sell",
                      entry_price, exit_price, realized_lots, pnl, reason)
            )
            lots -= realized_lots
            if close_lots is None or lots <= 1e-9:
                position = 0
                lots = 0.0

        for i in range(n):
            bar_time = idx[i]
            bar_day = bar_time.date()
            if bar_day != current_day:
                current_day = bar_day
                day_start_balance = balance
                day_trading_blocked = False

            # 1. Manage an existing position: partial TP first (if enabled),
            # then final SL/TP, against this bar's range.
            if position != 0:
                if cfg.use_partial_tp and not tp1_done:
                    hit_tp1 = (h[i] >= tp1_price) if position == 1 else (l[i] <= tp1_price)
                    if hit_tp1:
                        close_position(tp1_price, bar_time, "take_profit_1",
                                        close_lots=lots * cfg.tp1_close_fraction)
                        tp1_done = True
                        if cfg.move_sl_to_breakeven_after_tp1:
                            sl_price = entry_price

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

                if cfg.use_atr_stops and atr_values is not None and not np.isnan(atr_values[i]) and atr_values[i] > 0:
                    sl_pips = (atr_values[i] * cfg.atr_sl_mult) / cfg.pip_size
                    tp_pips = (atr_values[i] * cfg.atr_tp_mult) / cfg.pip_size
                else:
                    sl_pips = cfg.stop_loss_pips
                    tp_pips = cfg.take_profit_pips

                if stop_mult is not None:
                    sl_pips = sl_pips * stop_mult[i]
                effective_risk_pct = cfg.risk_per_trade_pct * (risk_mult[i] if risk_mult is not None else 1.0)

                lots = lots_for_fixed_risk(
                    equity=balance,
                    risk_per_trade_pct=effective_risk_pct,
                    stop_loss_pips=sl_pips,
                    pip_value_per_lot=cfg.pip_value_per_lot,
                    min_lot=cfg.min_lot,
                    lot_step=cfg.lot_step,
                    max_lot=cfg.max_lot,
                )
                if lots > 0:
                    position = side
                    entry_price = raw_entry
                    entry_time = bar_time
                    tp1_done = False
                    sl_distance = sl_pips * cfg.pip_size
                    tp_distance = tp_pips * cfg.pip_size
                    if side == 1:
                        sl_price = entry_price - sl_distance
                        tp_price = entry_price + tp_distance
                        tp1_price = entry_price + sl_distance * cfg.tp1_r_multiple
                    else:
                        sl_price = entry_price + sl_distance
                        tp_price = entry_price - tp_distance
                        tp1_price = entry_price - sl_distance * cfg.tp1_r_multiple

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
