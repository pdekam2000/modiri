from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import aroon


class AroonTrendStrategy(Strategy):
    """Trend direction from time-since-extreme: long when Aroon-Up is
    strong and above Aroon-Down (a recent new high, uptrend), short for
    the mirror image."""

    name = "aroon_trend"

    def __init__(self, period: int = 25, threshold: float = 70.0):
        super().__init__(period=period, threshold=threshold)
        self.period = period
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        aroon_up, aroon_down = aroon(df["high"], df["low"], self.period)
        out = pd.Series(0, index=df.index, dtype=int)
        out[(aroon_up > self.threshold) & (aroon_up > aroon_down)] = 1
        out[(aroon_down > self.threshold) & (aroon_down > aroon_up)] = -1
        return out
