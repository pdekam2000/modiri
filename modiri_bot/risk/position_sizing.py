"""Position sizing shared by the backtester and the live trader."""
from __future__ import annotations

import math


def lots_for_fixed_risk(
    equity: float,
    risk_per_trade_pct: float,
    stop_loss_pips: float,
    pip_value_per_lot: float,
    min_lot: float,
    lot_step: float,
    max_lot: float | None = None,
) -> float:
    """Size a position so that a full stop-loss hit loses exactly
    `risk_per_trade_pct` % of current equity (before rounding to the
    broker's lot step)."""
    if stop_loss_pips <= 0 or pip_value_per_lot <= 0:
        return min_lot

    risk_amount = equity * (risk_per_trade_pct / 100.0)
    raw_lots = risk_amount / (stop_loss_pips * pip_value_per_lot)

    steps = math.floor(raw_lots / lot_step)
    lots = max(steps, 0) * lot_step
    lots = max(lots, min_lot)
    if max_lot is not None:
        lots = min(lots, max_lot)
    return round(lots, 8)
