from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import fair_value_gaps


class FairValueGapStrategy(Strategy):
    """Basic Smart Money Concepts imbalance trade: a 3-bar Fair Value Gap
    (a price range the market jumped over) is treated as a magnet/support-
    resistance zone. Long when price retraces back down into an unfilled
    bullish gap and holds above its floor; short on the mirror-image
    bearish gap. Position held for `hold_bars`."""

    name = "fair_value_gap"

    def __init__(self, hold_bars: int = 5):
        super().__init__(hold_bars=hold_bars)
        self.hold_bars = hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        bull_gap, bear_gap = fair_value_gaps(df["high"], df["low"])
        h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
        bull_v, bear_v = bull_gap.to_numpy(), bear_gap.to_numpy()
        n = len(c)

        pending: dict[int, int] = {}
        active_bull = None  # (gap_low, gap_high)
        active_bear = None

        for i in range(2, n):
            if not np.isnan(bull_v[i]):
                active_bull = (bull_v[i], l[i])
            if not np.isnan(bear_v[i]):
                active_bear = (h[i], bear_v[i])

            if active_bull is not None:
                gap_low, gap_high = active_bull
                if l[i] <= gap_high and c[i] > gap_low:
                    pending[i] = 1
                    active_bull = None
                elif c[i] < gap_low:
                    active_bull = None  # gap fully broken through, invalidated

            if active_bear is not None:
                gap_low, gap_high = active_bear
                if h[i] >= gap_low and c[i] < gap_high:
                    pending[i] = -1
                    active_bear = None
                elif c[i] > gap_high:
                    active_bear = None

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
