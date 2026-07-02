from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import rsi


class RSIReversionStrategy(Strategy):
    """Mean-reversion: go long when oversold, short when overbought, flat
    once RSI crosses back through the midline."""

    name = "rsi_reversion"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        super().__init__(period=period, oversold=oversold, overbought=overbought)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], self.period)

        position = 0
        values = r.to_numpy()
        out = np.zeros(len(values), dtype=int)
        for i in range(len(values)):
            v = values[i]
            if position == 0:
                if v < self.oversold:
                    position = 1
                elif v > self.overbought:
                    position = -1
            elif position == 1 and v >= 50:
                position = 0
            elif position == -1 and v <= 50:
                position = 0
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
