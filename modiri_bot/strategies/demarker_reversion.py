from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import demarker


class DeMarkerReversionStrategy(Strategy):
    """Mean-reversion on the DeMarker indicator (range 0-1, compares each
    bar's new highs/lows to recent ones): long below oversold, short above
    overbought, flat at 0.5."""

    name = "demarker_reversion"

    def __init__(self, period: int = 14, oversold: float = 0.3, overbought: float = 0.7):
        super().__init__(period=period, oversold=oversold, overbought=overbought)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        values = demarker(df["high"], df["low"], self.period).to_numpy()
        out = np.zeros(len(values), dtype=int)
        position = 0
        for i in range(len(values)):
            v = values[i]
            if position == 0:
                if v < self.oversold:
                    position = 1
                elif v > self.overbought:
                    position = -1
            elif position == 1 and v >= 0.5:
                position = 0
            elif position == -1 and v <= 0.5:
                position = 0
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
