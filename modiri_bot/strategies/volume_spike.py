from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class VolumeSpikeStrategy(Strategy):
    """A crude proxy for large-player activity: when a bar's traded volume
    is an unusual multiple of its recent average, treat that as a sign a
    big participant moved the market, and follow the direction of that
    bar. Volume is compared only against *prior* bars (shifted rolling
    mean) so there's no look-ahead. Position is held for `hold_bars`."""

    name = "volume_spike"

    def __init__(self, volume_period: int = 20, spike_mult: float = 2.5, hold_bars: int = 5):
        super().__init__(volume_period=volume_period, spike_mult=spike_mult, hold_bars=hold_bars)
        self.volume_period = volume_period
        self.spike_mult = spike_mult
        self.hold_bars = hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        avg_volume = df["volume"].rolling(self.volume_period, min_periods=self.volume_period).mean().shift(1)
        is_spike = df["volume"] > (avg_volume * self.spike_mult)
        bullish = df["close"] > df["open"]
        bearish = df["close"] < df["open"]

        spike_v = is_spike.to_numpy()
        bull_v = bullish.to_numpy()
        bear_v = bearish.to_numpy()
        n = len(spike_v)
        out = np.zeros(n, dtype=int)

        position = 0
        hold_counter = 0
        for i in range(n):
            if spike_v[i] and bull_v[i]:
                position = 1
                hold_counter = self.hold_bars
            elif spike_v[i] and bear_v[i]:
                position = -1
                hold_counter = self.hold_bars
            elif hold_counter > 0:
                hold_counter -= 1
                if hold_counter == 0:
                    position = 0
            out[i] = position
        return pd.Series(out, index=df.index, dtype=int)
