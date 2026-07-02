from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import bollinger_bands


class BollingerBreakoutStrategy(Strategy):
    """Volatility breakout: long once price closes above the upper band,
    short once it closes below the lower band, flat again once price is
    back inside the bands (near the middle)."""

    name = "bollinger_breakout"

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(period=period, num_std=num_std)
        self.period = period
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper, mid, lower = bollinger_bands(df["close"], self.period, self.num_std)
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
