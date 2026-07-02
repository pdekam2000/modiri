from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import ema, rsi


class TrendPullbackStrategy(Strategy):
    """Buy corrective dips within an uptrend, sell corrective bounces
    within a downtrend: an EMA defines the prevailing trend direction,
    and RSI dipping into oversold/overbought territory *while that trend
    holds* marks a retracement entry back in the trend's direction. Flat
    once RSI reverts to the midline or the trend itself flips."""

    name = "trend_pullback"

    def __init__(self, trend_period: int = 50, rsi_period: int = 14,
                 dip_oversold: float = 40.0, dip_overbought: float = 60.0):
        super().__init__(trend_period=trend_period, rsi_period=rsi_period,
                          dip_oversold=dip_oversold, dip_overbought=dip_overbought)
        self.trend_period = trend_period
        self.rsi_period = rsi_period
        self.dip_oversold = dip_oversold
        self.dip_overbought = dip_overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trend = ema(df["close"], self.trend_period)
        r = rsi(df["close"], self.rsi_period)
        close = df["close"].to_numpy()
        trend_v = trend.to_numpy()
        rsi_v = r.to_numpy()
        n = len(close)
        out = np.zeros(n, dtype=int)

        position = 0
        for i in range(n):
            if np.isnan(trend_v[i]):
                out[i] = 0
                continue
            uptrend = close[i] > trend_v[i]
            downtrend = close[i] < trend_v[i]

            if position == 1 and not uptrend:
                position = 0
            elif position == -1 and not downtrend:
                position = 0

            if position == 0:
                if uptrend and rsi_v[i] < self.dip_oversold:
                    position = 1
                elif downtrend and rsi_v[i] > self.dip_overbought:
                    position = -1
            elif position == 1 and rsi_v[i] >= 50:
                position = 0
            elif position == -1 and rsi_v[i] <= 50:
                position = 0

            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
