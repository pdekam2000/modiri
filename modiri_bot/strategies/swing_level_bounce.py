from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import swing_points


class SwingLevelBounceStrategy(Strategy):
    """Dynamic support/resistance from the market's own recent structure:
    tracks the most recent confirmed swing low (support) and swing high
    (resistance) -- via `swing_points`, so each is only known `swing_order`
    bars after it happens -- and trades a rejection bounce off either
    level, the same way PivotPointBounceStrategy trades static daily
    pivots. Position held for `hold_bars`."""

    name = "swing_level_bounce"

    def __init__(self, swing_order: int = 5, hold_bars: int = 5):
        super().__init__(swing_order=swing_order, hold_bars=hold_bars)
        self.swing_order = swing_order
        self.hold_bars = hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        swing_high, swing_low = swing_points(df["high"], df["low"], self.swing_order)
        sh_v, sl_v = swing_high.to_numpy(), swing_low.to_numpy()
        low, high, close = df["low"].to_numpy(), df["high"].to_numpy(), df["close"].to_numpy()
        n = len(close)

        out = np.zeros(n, dtype=int)
        position = 0
        hold_counter = 0
        current_support = np.nan
        current_resistance = np.nan

        for i in range(n):
            if not np.isnan(sl_v[i]):
                current_support = sl_v[i]
            if not np.isnan(sh_v[i]):
                current_resistance = sh_v[i]

            bullish = (not np.isnan(current_support) and low[i] <= current_support
                       and close[i] > current_support)
            bearish = (not np.isnan(current_resistance) and high[i] >= current_resistance
                       and close[i] < current_resistance)

            if bullish:
                position = 1
                hold_counter = self.hold_bars
            elif bearish:
                position = -1
                hold_counter = self.hold_bars
            elif hold_counter > 0:
                hold_counter -= 1
                if hold_counter == 0:
                    position = 0
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
