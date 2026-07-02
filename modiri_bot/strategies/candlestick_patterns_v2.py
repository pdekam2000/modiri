from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import candle_shape


class CandlestickPatternsV2Strategy(Strategy):
    """A second batch of classic reversal patterns not covered by
    CandlestickPatternStrategy: Doji (indecision, only actionable as a
    reversal at a trend extreme), Morning/Evening Star (3-candle), and
    Harami (2-candle inside-body contraction). Held for `hold_bars`."""

    name = "candlestick_patterns_v2"

    def __init__(self, doji_body_ratio: float = 0.1, star_body_ratio: float = 0.3, hold_bars: int = 5):
        super().__init__(doji_body_ratio=doji_body_ratio, star_body_ratio=star_body_ratio, hold_bars=hold_bars)
        self.doji_body_ratio = doji_body_ratio
        self.star_body_ratio = star_body_ratio
        self.hold_bars = hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body, _uw, _lw, range_ = candle_shape(o, h, l, c)
        o_v, c_v, body_v, range_v = o.to_numpy(), c.to_numpy(), body.to_numpy(), range_.to_numpy()
        n = len(c_v)

        pending: dict[int, int] = {}

        for i in range(2, n):
            if np.isnan(range_v[i]) or range_v[i] == 0:
                continue

            # Doji at a local extreme: indecision after a run in one
            # direction often precedes a reversal.
            is_doji = body_v[i] / range_v[i] <= self.doji_body_ratio
            if is_doji:
                prior_trend_up = c_v[i - 1] > c_v[i - 3] if i >= 3 else False
                prior_trend_down = c_v[i - 1] < c_v[i - 3] if i >= 3 else False
                if prior_trend_down:
                    pending[i] = 1
                elif prior_trend_up:
                    pending[i] = -1

            # Harami: candle i's body fully inside candle i-1's body, and
            # the two candles are opposite colors (contraction/indecision).
            prev_body_hi = max(o_v[i - 1], c_v[i - 1])
            prev_body_lo = min(o_v[i - 1], c_v[i - 1])
            cur_body_hi = max(o_v[i], c_v[i])
            cur_body_lo = min(o_v[i], c_v[i])
            prev_bearish = c_v[i - 1] < o_v[i - 1]
            prev_bullish = c_v[i - 1] > o_v[i - 1]
            inside = cur_body_hi <= prev_body_hi and cur_body_lo >= prev_body_lo
            if inside and prev_bearish and c_v[i] >= o_v[i]:
                pending[i] = 1
            elif inside and prev_bullish and c_v[i] <= o_v[i]:
                pending[i] = -1

            # Morning/Evening Star: big candle, small-body middle candle
            # (a gap-like pause), then a big candle closing back through
            # most of the first candle's body.
            if i >= 2:
                first_bearish = c_v[i - 2] < o_v[i - 2]
                first_bullish = c_v[i - 2] > o_v[i - 2]
                mid_small = range_v[i - 1] > 0 and body_v[i - 1] / range_v[i - 1] <= self.star_body_ratio
                first_body_mid = (o_v[i - 2] + c_v[i - 2]) / 2

                if first_bearish and mid_small and c_v[i] > o_v[i] and c_v[i] > first_body_mid:
                    pending[i] = 1  # morning star
                elif first_bullish and mid_small and c_v[i] < o_v[i] and c_v[i] < first_body_mid:
                    pending[i] = -1  # evening star

        out = np.zeros(n, dtype=int)
        position = 0
        hold_counter = 0
        for i in range(n):
            if i in pending:
                position = pending[i]
                hold_counter = self.hold_bars
            elif hold_counter > 0:
                hold_counter -= 1
                if hold_counter == 0:
                    position = 0
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
