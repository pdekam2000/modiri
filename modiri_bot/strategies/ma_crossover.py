from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import ema, sma


class MACrossoverStrategy(Strategy):
    """Classic trend-following crossover: long while fast MA > slow MA."""

    name = "ma_crossover"

    def __init__(self, fast_period: int = 10, slow_period: int = 50, use_ema: bool = True):
        super().__init__(fast_period=fast_period, slow_period=slow_period, use_ema=use_ema)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ma_fn = ema if self.use_ema else sma
        fast = ma_fn(df["close"], self.fast_period)
        slow = ma_fn(df["close"], self.slow_period)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[fast > slow] = 1
        signal[fast < slow] = -1
        return signal
