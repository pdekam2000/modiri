from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import parabolic_sar


class ParabolicSARTrendStrategy(Strategy):
    """Stop-and-reverse trend following: long while price is above the
    SAR dots, short while below."""

    name = "parabolic_sar_trend"

    def __init__(self, af_step: float = 0.02, af_max: float = 0.2):
        super().__init__(af_step=af_step, af_max=af_max)
        self.af_step = af_step
        self.af_max = af_max

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sar = parabolic_sar(df["high"], df["low"], self.af_step, self.af_max)
        out = pd.Series(0, index=df.index, dtype=int)
        out[df["close"] > sar] = 1
        out[df["close"] < sar] = -1
        return out
