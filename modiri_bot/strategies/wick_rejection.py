from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import candle_shape, ema


class WickRejectionStrategy(Strategy):
    """Long-wick rejection: a candle whose lower wick is an unusually large
    fraction of its total range shows the market rejected lower prices
    within that bar (bullish); the mirror image on the upper wick is
    bearish. Only counted against the prevailing EMA trend (a rejection
    candle in the direction of the trend is just normal continuation, not
    a reversal signal). Position is held for `hold_bars` bars."""

    name = "wick_rejection"

    def __init__(self, trend_period: int = 50, wick_ratio_min: float = 0.6, hold_bars: int = 5):
        super().__init__(trend_period=trend_period, wick_ratio_min=wick_ratio_min, hold_bars=hold_bars)
        self.trend_period = trend_period
        self.wick_ratio_min = wick_ratio_min
        self.hold_bars = hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        _body, upper_wick, lower_wick, range_ = candle_shape(o, h, l, c)
        trend = ema(c, self.trend_period)

        uw_v, lw_v, range_v = upper_wick.to_numpy(), lower_wick.to_numpy(), range_.to_numpy()
        c_v, trend_v = c.to_numpy(), trend.to_numpy()
        n = len(c_v)

        out = np.zeros(n, dtype=int)
        hold_counter = 0
        position = 0

        for i in range(n):
            bullish_signal = False
            bearish_signal = False

            if not np.isnan(range_v[i]) and range_v[i] > 0 and not np.isnan(trend_v[i]):
                if (lw_v[i] / range_v[i]) >= self.wick_ratio_min and c_v[i] < trend_v[i]:
                    bullish_signal = True
                if (uw_v[i] / range_v[i]) >= self.wick_ratio_min and c_v[i] > trend_v[i]:
                    bearish_signal = True

            if bullish_signal:
                position = 1
                hold_counter = self.hold_bars
            elif bearish_signal:
                position = -1
                hold_counter = self.hold_bars
            elif hold_counter > 0:
                hold_counter -= 1
                if hold_counter == 0:
                    position = 0
            out[i] = position

        return pd.Series(out, index=df.index, dtype=int)
