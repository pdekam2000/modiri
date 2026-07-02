from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import atr, ema


class ATRChannelBreakoutStrategy(Strategy):
    """Keltner-style volatility breakout: an EMA midline +/- a multiple of
    ATR forms the channel; long on a close above the upper band, short on
    a close below the lower band, flat again once price is back near the
    midline."""

    name = "atr_channel_breakout"

    def __init__(self, ema_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0):
        super().__init__(ema_period=ema_period, atr_period=atr_period, atr_mult=atr_mult)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mid = ema(df["close"], self.ema_period)
        band = atr(df["high"], df["low"], df["close"], self.atr_period) * self.atr_mult
        upper = mid + band
        lower = mid - band
        close = df["close"].to_numpy()
        u, m, l = upper.to_numpy(), mid.to_numpy(), lower.to_numpy()

        out = [0] * len(close)
        position = 0
        for i in range(len(close)):
            if pd.isna(u[i]):
                out[i] = 0
                continue
            if position == 0:
                if close[i] > u[i]:
                    position = 1
                elif close[i] < l[i]:
                    position = -1
            elif position == 1 and close[i] < m[i]:
                position = 0
            elif position == -1 and close[i] > m[i]:
                position = 0
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
