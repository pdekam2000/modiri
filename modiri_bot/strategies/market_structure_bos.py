from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import swing_points


class MarketStructureBOSStrategy(Strategy):
    """Basic Smart Money Concepts market structure: tracks the sequence of
    confirmed swing highs/lows. A Break of Structure (BOS) -- price
    closing beyond the most recent swing in the direction of the current
    trend -- confirms and re-signals continuation. A Change of Character
    (CHOCH) -- price closing beyond the most recent swing *against* the
    current trend -- flips the trend and signals the reversal."""

    name = "market_structure_bos"

    def __init__(self, swing_order: int = 5):
        super().__init__(swing_order=swing_order)
        self.swing_order = swing_order

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        swing_high, swing_low = swing_points(df["high"], df["low"], self.swing_order)
        sh_v, sl_v = swing_high.to_numpy(), swing_low.to_numpy()
        c = df["close"].to_numpy()
        n = len(c)

        out = np.zeros(n, dtype=int)
        trend = 0
        last_swing_high = np.nan
        last_swing_low = np.nan

        for i in range(n):
            if not np.isnan(sh_v[i]):
                last_swing_high = sh_v[i]
            if not np.isnan(sl_v[i]):
                last_swing_low = sl_v[i]

            if not np.isnan(last_swing_high) and c[i] > last_swing_high:
                trend = 1  # BOS if already up, CHOCH (reversal into up) if was down
            elif not np.isnan(last_swing_low) and c[i] < last_swing_low:
                trend = -1

            out[i] = trend
        return pd.Series(out, index=df.index, dtype=int)
