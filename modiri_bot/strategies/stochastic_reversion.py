from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import stochastic


class StochasticReversionStrategy(Strategy):
    """Mean-reversion on the stochastic oscillator: long when %K dips below
    oversold, short when it rises above overbought, flat once %K crosses
    back through the midline."""

    name = "stochastic_reversion"

    def __init__(self, k_period: int = 14, d_period: int = 3, oversold: float = 20.0, overbought: float = 80.0):
        super().__init__(k_period=k_period, d_period=d_period, oversold=oversold, overbought=overbought)
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        percent_k, _ = stochastic(df["high"], df["low"], df["close"], self.k_period, self.d_period)
        values = percent_k.to_numpy()
        out = np.zeros(len(values), dtype=int)

        position = 0
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
