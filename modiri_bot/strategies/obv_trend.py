from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import ema, obv


class OBVTrendStrategy(Strategy):
    """Volume-flow trend confirmation: long while On-Balance Volume is
    above its own moving average (net buying pressure building), short
    while below."""

    name = "obv_trend"

    def __init__(self, obv_ma_period: int = 20):
        super().__init__(obv_ma_period=obv_ma_period)
        self.obv_ma_period = obv_ma_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        obv_line = obv(df["close"], df["volume"])
        obv_ma = ema(obv_line, self.obv_ma_period)
        out = pd.Series(0, index=df.index, dtype=int)
        out[obv_line > obv_ma] = 1
        out[obv_line < obv_ma] = -1
        return out
