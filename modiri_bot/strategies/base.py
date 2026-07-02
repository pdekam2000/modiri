"""Common interface every strategy implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """A strategy turns an OHLCV DataFrame into a target-position series.

    `generate_signals` must return a pd.Series aligned to `df.index` with
    values in {-1, 0, 1}: -1 short, 0 flat, 1 long. Signals are the *target*
    position for the bar; the backtest engine handles entries/exits, sizing,
    and stop-loss/take-profit.
    """

    name: str = "strategy"

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ...

    def with_params(self, **params: Any) -> "Strategy":
        merged = {**self.params, **params}
        return type(self)(**merged)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"
