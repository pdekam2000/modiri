from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import ema, sma


class MTFTrendFilterStrategy(Strategy):
    """Multi-timeframe confirmation: an MA-crossover signal on the base
    timeframe is only taken when it agrees with the trend on a higher
    timeframe (e.g. take H4 longs only when the daily trend is also up).
    The higher-timeframe trend is shifted by one HTF bar before being
    aligned back down, so only fully-closed higher-timeframe bars ever
    inform a decision -- no look-ahead into a still-forming bar."""

    name = "mtf_trend_filter"

    def __init__(self, fast_period: int = 10, slow_period: int = 30, use_ema: bool = True,
                 htf_rule: str = "1D", htf_trend_period: int = 20):
        super().__init__(fast_period=fast_period, slow_period=slow_period, use_ema=use_ema,
                          htf_rule=htf_rule, htf_trend_period=htf_trend_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema
        self.htf_rule = htf_rule
        self.htf_trend_period = htf_trend_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ma_fn = ema if self.use_ema else sma
        fast = ma_fn(df["close"], self.fast_period)
        slow = ma_fn(df["close"], self.slow_period)
        base = pd.Series(0, index=df.index, dtype=int)
        base[fast > slow] = 1
        base[fast < slow] = -1

        htf_close = df["close"].resample(self.htf_rule).last().dropna()
        htf_trend = ema(htf_close, self.htf_trend_period)
        htf_uptrend = (htf_close > htf_trend).astype(float)
        htf_uptrend[htf_trend.isna()] = np.nan
        htf_uptrend = htf_uptrend.shift(1)
        htf_aligned = htf_uptrend.reindex(df.index, method="ffill")

        out = pd.Series(0, index=df.index, dtype=int)
        out[(base == 1) & (htf_aligned == 1.0)] = 1
        out[(base == -1) & (htf_aligned == 0.0)] = -1
        return out
