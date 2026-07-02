from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import money_flow_index


class MFIReversionStrategy(Strategy):
    """Volume-weighted mean-reversion (Money Flow Index is RSI weighted by
    traded volume): long when MFI is oversold, short when overbought,
    flat once it crosses back through the midline."""

    name = "mfi_reversion"

    def __init__(self, period: int = 14, oversold: float = 20.0, overbought: float = 80.0):
        super().__init__(period=period, oversold=oversold, overbought=overbought)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        values = money_flow_index(df["high"], df["low"], df["close"], df["volume"], self.period).to_numpy()
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
