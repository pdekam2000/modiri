"""Synthetic OHLCV generator.

This produces a random-walk price series purely so the test suite and the
demo script have *something* to run against without a network connection.
It has no relationship to real market behaviour and must never be used to
judge whether a strategy is actually profitable — only real historical
broker data (see csv_loader.py / fetch_mt5_history.py) can do that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_bars: int = 5000,
    start_price: float = 1.1000,
    annual_vol: float = 0.08,
    bars_per_year: int = 24 * 252,
    seed: int = 42,
    freq: str = "h",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year
    sigma = annual_vol * np.sqrt(dt)
    returns = rng.normal(loc=0.0, scale=sigma, size=n_bars)
    close = start_price * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.normal(0, sigma / 2, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, sigma / 2, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.integers(50, 500, n_bars).astype(float)

    index = pd.date_range("2024-01-01", periods=n_bars, freq=freq)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
