from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import rolling_vwap


class VWAPReversionStrategy(Strategy):
    """Mean-reversion around a rolling volume-weighted average price:
    long when price stretches more than k standard deviations below VWAP,
    short when it stretches above, flat once price reverts back to VWAP."""

    name = "vwap_reversion"

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(period=period, num_std=num_std)
        self.period = period
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        vwap, dev = rolling_vwap(df["high"], df["low"], df["close"], df["volume"], self.period)
        upper = vwap + self.num_std * dev
        lower = vwap - self.num_std * dev
        close = df["close"].to_numpy()
        v, u, l = vwap.to_numpy(), upper.to_numpy(), lower.to_numpy()

        out = np.zeros(len(close), dtype=int)
        position = 0
        for i in range(len(close)):
            if np.isnan(u[i]):
                out[i] = 0
                continue
            if position == 0:
                if close[i] < l[i]:
                    position = 1
                elif close[i] > u[i]:
                    position = -1
            elif position == 1 and close[i] >= v[i]:
                position = 0
            elif position == -1 and close[i] <= v[i]:
                position = 0
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
