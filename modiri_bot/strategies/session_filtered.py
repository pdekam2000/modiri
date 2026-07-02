from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import ema, sma


class SessionFilteredStrategy(Strategy):
    """An MA-crossover signal, only taken during a specific trading session
    window (e.g. the London/New York open or their overlap), when large
    players are most active and liquidity/volatility is highest. Hour is
    read directly from the bar timestamps (broker server time, whatever
    that is for your data) -- adjust session_start_hour/session_end_hour
    to match your broker if you know its UTC offset."""

    name = "session_filtered"

    def __init__(self, fast_period: int = 10, slow_period: int = 30, use_ema: bool = True,
                 session_start_hour: int = 7, session_end_hour: int = 16):
        super().__init__(fast_period=fast_period, slow_period=slow_period, use_ema=use_ema,
                          session_start_hour=session_start_hour, session_end_hour=session_end_hour)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema
        self.session_start_hour = session_start_hour
        self.session_end_hour = session_end_hour

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ma_fn = ema if self.use_ema else sma
        fast = ma_fn(df["close"], self.fast_period)
        slow = ma_fn(df["close"], self.slow_period)
        base = pd.Series(0, index=df.index, dtype=int)
        base[fast > slow] = 1
        base[fast < slow] = -1

        hour = df.index.hour
        if self.session_start_hour <= self.session_end_hour:
            in_session = (hour >= self.session_start_hour) & (hour < self.session_end_hour)
        else:  # session wraps past midnight
            in_session = (hour >= self.session_start_hour) | (hour < self.session_end_hour)

        out = base.copy()
        out[~in_session] = 0
        return out
