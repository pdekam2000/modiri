from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import cci


class CCIReversionStrategy(Strategy):
    """Mean-reversion on the Commodity Channel Index: long when CCI drops
    below -threshold, short when it rises above +threshold, flat once it
    crosses back through zero."""

    name = "cci_reversion"

    def __init__(self, period: int = 20, threshold: float = 100.0):
        super().__init__(period=period, threshold=threshold)
        self.period = period
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        values = cci(df["high"], df["low"], df["close"], self.period).to_numpy()
        out = np.zeros(len(values), dtype=int)

        position = 0
        for i in range(len(values)):
            v = values[i]
            if np.isnan(v):
                out[i] = position
                continue
            if position == 0:
                if v < -self.threshold:
                    position = 1
                elif v > self.threshold:
                    position = -1
            elif position == 1 and v >= 0:
                position = 0
            elif position == -1 and v <= 0:
                position = 0
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
