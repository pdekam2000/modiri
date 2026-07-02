from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import daily_pivot_points


class PivotPointBounceStrategy(Strategy):
    """Classic floor-trader pivots (from the prior day's H/L/C): long when
    price dips to the S1 support level and closes back above it (a
    rejection bounce), short on the mirror-image rejection at R1
    resistance. These are static, pre-computed levels widely watched by
    institutional desks -- a different mechanism from every rolling-
    indicator strategy in this library. Position held for `hold_bars`."""

    name = "pivot_points"

    def __init__(self, level: str = "1", hold_bars: int = 5):
        super().__init__(level=level, hold_bars=hold_bars)
        self.level = level  # "1" -> S1/R1, "2" -> S2/R2
        self.hold_bars = hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        levels = daily_pivot_points(df)
        support = levels[f"s{self.level}"].to_numpy()
        resistance = levels[f"r{self.level}"].to_numpy()
        low, high, close = df["low"].to_numpy(), df["high"].to_numpy(), df["close"].to_numpy()
        n = len(close)

        out = np.zeros(n, dtype=int)
        position = 0
        hold_counter = 0
        for i in range(n):
            bullish = not np.isnan(support[i]) and low[i] <= support[i] and close[i] > support[i]
            bearish = not np.isnan(resistance[i]) and high[i] >= resistance[i] and close[i] < resistance[i]
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
