from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import vortex


class VortexTrendStrategy(Strategy):
    """Trend direction from the Vortex Indicator: long when VI+ > VI-,
    short when VI- > VI+."""

    name = "vortex_trend"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        vi_plus, vi_minus = vortex(df["high"], df["low"], df["close"], self.period)
        out = pd.Series(0, index=df.index, dtype=int)
        out[vi_plus > vi_minus] = 1
        out[vi_minus > vi_plus] = -1
        return out
