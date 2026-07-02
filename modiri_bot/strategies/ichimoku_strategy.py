from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import ichimoku


class IchimokuStrategy(Strategy):
    """Trend-following: long when Tenkan > Kijun and price is above the
    cloud, short when Tenkan < Kijun and price is below the cloud."""

    name = "ichimoku"

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
        super().__init__(tenkan_period=tenkan_period, kijun_period=kijun_period, senkou_b_period=senkou_b_period)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        tenkan, kijun, senkou_a, senkou_b = ichimoku(
            df["high"], df["low"], df["close"],
            self.tenkan_period, self.kijun_period, self.senkou_b_period,
        )
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        close = df["close"]

        out = pd.Series(0, index=df.index, dtype=int)
        out[(tenkan > kijun) & (close > cloud_top)] = 1
        out[(tenkan < kijun) & (close < cloud_bottom)] = -1
        return out
