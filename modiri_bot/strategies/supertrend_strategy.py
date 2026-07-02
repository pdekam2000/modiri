from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import supertrend


class SuperTrendStrategy(Strategy):
    """ATR-based stop-and-reverse trend following (SuperTrend indicator):
    long while price holds above the trailing band, short while below."""

    name = "supertrend"

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__(period=period, multiplier=multiplier)
        self.period = period
        self.multiplier = multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        _line, trend = supertrend(df["high"], df["low"], df["close"], self.period, self.multiplier)
        return trend.astype(int)
