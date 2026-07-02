from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import heikin_ashi


class HeikinAshiTrendStrategy(Strategy):
    """Trend-following on smoothed Heikin-Ashi candles instead of raw OHLC:
    long once `confirm_bars` consecutive HA candles close green (HA_close >
    HA_open), short once that many close red. Heikin-Ashi averages each
    bar with the synthetic previous one, so it reacts slower to single-bar
    noise than a raw-price trend filter."""

    name = "heikin_ashi_trend"

    def __init__(self, confirm_bars: int = 2):
        super().__init__(confirm_bars=confirm_bars)
        self.confirm_bars = confirm_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ha_open, _ha_high, _ha_low, ha_close = heikin_ashi(df["open"], df["high"], df["low"], df["close"])
        green = (ha_close > ha_open).to_numpy()
        red = (ha_close < ha_open).to_numpy()
        n = len(green)

        out = np.zeros(n, dtype=int)
        position = 0
        streak = 0
        streak_dir = 0
        for i in range(n):
            direction = 1 if green[i] else (-1 if red[i] else 0)
            if direction == streak_dir and direction != 0:
                streak += 1
            else:
                streak = 1
                streak_dir = direction
            if streak >= self.confirm_bars and direction != 0:
                position = direction
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
