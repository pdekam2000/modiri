from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import rsi


class RSIDivergenceStrategy(Strategy):
    """Bullish divergence: price makes a lower swing low while RSI makes a
    higher low at that same point -- momentum is fading even as price falls,
    a classic reversal warning. Bearish divergence is the mirror image on
    swing highs. A swing is only confirmed `swing_order` bars after it
    occurs (no look-ahead); the divergence check compares it against the
    previous confirmed swing of the same type. Position held for
    `hold_bars` bars once triggered."""

    name = "rsi_divergence"

    def __init__(self, rsi_period: int = 14, swing_order: int = 3, hold_bars: int = 8):
        super().__init__(rsi_period=rsi_period, swing_order=swing_order, hold_bars=hold_bars)
        self.rsi_period = rsi_period
        self.swing_order = swing_order
        self.hold_bars = hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], self.rsi_period)
        h, l = df["high"].to_numpy(), df["low"].to_numpy()
        rsi_v = r.to_numpy()
        n = len(h)
        order = self.swing_order

        # First pass: find swing points and record divergence trigger
        # events, keyed by the bar where the swing becomes confirmed
        # (order bars after it occurs -- no look-ahead).
        pending: dict[int, int] = {}
        last_low_price = None
        last_low_rsi = None
        last_high_price = None
        last_high_rsi = None

        for i in range(order, n - order):
            window_l = l[i - order: i + order + 1]
            if l[i] == window_l.min():
                if last_low_price is not None and l[i] < last_low_price and rsi_v[i] > last_low_rsi:
                    pending[i + order] = 1
                last_low_price = l[i]
                last_low_rsi = rsi_v[i]

            window_h = h[i - order: i + order + 1]
            if h[i] == window_h.max():
                if last_high_price is not None and h[i] > last_high_price and rsi_v[i] < last_high_rsi:
                    pending[i + order] = -1
                last_high_price = h[i]
                last_high_rsi = rsi_v[i]

        # Second pass: turn trigger events into a held position timeline.
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
