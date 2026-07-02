from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import trix


class TrixMomentumStrategy(Strategy):
    """Momentum trend-following on TRIX (rate of change of a triple-
    smoothed EMA): long while TRIX is positive and rising, short while
    negative and falling -- a heavily filtered momentum signal that
    ignores a lot of short-term noise by construction."""

    name = "trix_momentum"

    def __init__(self, period: int = 15, signal_period: int = 9):
        super().__init__(period=period, signal_period=signal_period)
        self.period = period
        self.signal_period = signal_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trix_line = trix(df["close"], self.period)
        signal_line = trix_line.ewm(span=self.signal_period, adjust=False, min_periods=self.signal_period).mean()
        out = pd.Series(0, index=df.index, dtype=int)
        out[trix_line > signal_line] = 1
        out[trix_line < signal_line] = -1
        return out
