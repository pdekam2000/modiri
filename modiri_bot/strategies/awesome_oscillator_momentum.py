from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import awesome_oscillator


class AwesomeOscillatorMomentumStrategy(Strategy):
    """Momentum trend-following: long while the Awesome Oscillator
    (SMA5 - SMA34 of the median price) is positive, short while negative."""

    name = "awesome_oscillator_momentum"

    def __init__(self, fast: int = 5, slow: int = 34):
        super().__init__(fast=fast, slow=slow)
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ao = awesome_oscillator(df["high"], df["low"], self.fast, self.slow)
        out = pd.Series(0, index=df.index, dtype=int)
        out[ao > 0] = 1
        out[ao < 0] = -1
        return out
