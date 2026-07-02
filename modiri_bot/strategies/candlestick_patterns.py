from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import candle_shape, ema


class CandlestickPatternStrategy(Strategy):
    """Classic single/two-candle reversal patterns, read from raw OHLC shape
    rather than a rolling formula: bullish/bearish engulfing (no trend
    context needed), and hammer/shooting star (only counted as a reversal
    signal when they occur against the prevailing EMA trend, since a hammer
    in an uptrend is just noise). A triggered pattern is held for
    `hold_bars` bars, refreshed by another same-direction pattern, and
    overridden immediately by an opposite one."""

    name = "candlestick_patterns"

    def __init__(self, trend_period: int = 50, body_ratio_max: float = 0.35,
                 wick_ratio_min: float = 2.0, hold_bars: int = 5):
        super().__init__(trend_period=trend_period, body_ratio_max=body_ratio_max,
                          wick_ratio_min=wick_ratio_min, hold_bars=hold_bars)
        self.trend_period = trend_period
        self.body_ratio_max = body_ratio_max
        self.wick_ratio_min = wick_ratio_min
        self.hold_bars = hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body, upper_wick, lower_wick, range_ = candle_shape(o, h, l, c)
        trend = ema(c, self.trend_period)

        o_v, c_v = o.to_numpy(), c.to_numpy()
        body_v, uw_v, lw_v, range_v = body.to_numpy(), upper_wick.to_numpy(), lower_wick.to_numpy(), range_.to_numpy()
        trend_v = trend.to_numpy()
        n = len(c_v)

        out = np.zeros(n, dtype=int)
        hold_counter = 0
        position = 0

        for i in range(1, n):
            bullish_signal = False
            bearish_signal = False

            prev_bearish = c_v[i - 1] < o_v[i - 1]
            prev_bullish = c_v[i - 1] > o_v[i - 1]
            curr_bullish = c_v[i] > o_v[i]
            curr_bearish = c_v[i] < o_v[i]

            if prev_bearish and curr_bullish and o_v[i] <= c_v[i - 1] and c_v[i] >= o_v[i - 1]:
                bullish_signal = True
            if prev_bullish and curr_bearish and o_v[i] >= c_v[i - 1] and c_v[i] <= o_v[i - 1]:
                bearish_signal = True

            if not np.isnan(range_v[i]) and range_v[i] > 0 and not np.isnan(trend_v[i]):
                small_body = body_v[i] / range_v[i] <= self.body_ratio_max
                is_hammer = (
                    small_body
                    and lw_v[i] >= self.wick_ratio_min * max(body_v[i], 1e-12)
                    and uw_v[i] <= body_v[i] + 1e-12
                )
                is_shooting_star = (
                    small_body
                    and uw_v[i] >= self.wick_ratio_min * max(body_v[i], 1e-12)
                    and lw_v[i] <= body_v[i] + 1e-12
                )
                if is_hammer and c_v[i] < trend_v[i]:
                    bullish_signal = True
                if is_shooting_star and c_v[i] > trend_v[i]:
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
