from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import donchian_channel


class DonchianBreakoutStrategy(Strategy):
    """Turtle-style channel breakout: long on a new N-bar high, short on a
    new N-bar low, held until the opposite breakout fires."""

    name = "donchian_breakout"

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper, lower = donchian_channel(df["high"], df["low"], self.period)
        # Compare today's close against yesterday's channel to avoid look-ahead.
        prev_upper = upper.shift(1)
        prev_lower = lower.shift(1)
        close = df["close"]

        position = 0
        c = close.to_numpy()
        pu = prev_upper.to_numpy()
        pl = prev_lower.to_numpy()
        result = np.zeros(len(c), dtype=int)
        for i in range(len(c)):
            if pd.isna(pu[i]):
                result[i] = position
                continue
            if c[i] > pu[i]:
                position = 1
            elif c[i] < pl[i]:
                position = -1
            result[i] = position
        return pd.Series(result, index=df.index, dtype=int)
