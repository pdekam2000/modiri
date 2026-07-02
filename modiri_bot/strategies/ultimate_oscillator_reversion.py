from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import ultimate_oscillator


class UltimateOscillatorReversionStrategy(Strategy):
    """Mean-reversion on the Ultimate Oscillator, which blends 3
    timeframes (short/medium/long) of buying pressure into one reading --
    designed to reduce the false divergences a single-period oscillator
    gives. Long oversold, short overbought, flat at the midline."""

    name = "ultimate_oscillator_reversion"

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28,
                 oversold: float = 30.0, overbought: float = 70.0):
        super().__init__(period1=period1, period2=period2, period3=period3,
                          oversold=oversold, overbought=overbought)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        values = ultimate_oscillator(df["high"], df["low"], df["close"],
                                      self.period1, self.period2, self.period3).to_numpy()
        out = np.zeros(len(values), dtype=int)
        position = 0
        for i in range(len(values)):
            v = values[i]
            if np.isnan(v):
                out[i] = position
                continue
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
