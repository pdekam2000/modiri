from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import chaikin_money_flow


class CMFReversionStrategy(Strategy):
    """Chaikin Money Flow mean-reversion: long when money flow is
    unusually negative (selling pressure likely exhausted), short when
    unusually positive."""

    name = "cmf_reversion"

    def __init__(self, period: int = 20, threshold: float = 0.15):
        super().__init__(period=period, threshold=threshold)
        self.period = period
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        cmf = chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"], self.period)
        out = pd.Series(0, index=df.index, dtype=int)
        out[cmf < -self.threshold] = 1
        out[cmf > self.threshold] = -1
        return out
