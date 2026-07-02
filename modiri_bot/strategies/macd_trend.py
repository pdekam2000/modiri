from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import macd


class MACDTrendStrategy(Strategy):
    """Trend-following: long while MACD line is above its signal line."""

    name = "macd_trend"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        macd_line, signal_line, _ = macd(df["close"], self.fast, self.slow, self.signal)
        out = pd.Series(0, index=df.index, dtype=int)
        out[macd_line > signal_line] = 1
        out[macd_line < signal_line] = -1
        return out
