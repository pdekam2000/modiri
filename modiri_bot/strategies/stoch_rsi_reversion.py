from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import stoch_rsi


class StochRSIReversionStrategy(Strategy):
    """Mean-reversion on Stochastic RSI (the stochastic formula applied to
    RSI values rather than price) -- reacts faster and swings more
    sharply than plain RSI or plain Stochastic. Long on oversold, short on
    overbought, flat once it crosses back through the midline."""

    name = "stoch_rsi_reversion"

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14, oversold: float = 20.0, overbought: float = 80.0):
        super().__init__(rsi_period=rsi_period, stoch_period=stoch_period, oversold=oversold, overbought=overbought)
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        values = stoch_rsi(df["close"], self.rsi_period, self.stoch_period).to_numpy()
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
