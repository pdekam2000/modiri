from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import adx


class ADXTrendStrategy(Strategy):
    """Directional trend-following, gated by trend strength: long when
    +DI > -DI and ADX confirms a real trend is underway, short when
    -DI > +DI with the same confirmation, flat when ADX says it's a
    range (no reliable trend to follow)."""

    name = "adx_trend"

    def __init__(self, period: int = 14, adx_threshold: float = 25.0):
        super().__init__(period=period, adx_threshold=adx_threshold)
        self.period = period
        self.adx_threshold = adx_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        adx_line, plus_di, minus_di = adx(df["high"], df["low"], df["close"], self.period)
        out = pd.Series(0, index=df.index, dtype=int)
        trending = adx_line > self.adx_threshold
        out[trending & (plus_di > minus_di)] = 1
        out[trending & (minus_di > plus_di)] = -1
        return out
