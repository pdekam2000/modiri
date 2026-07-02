"""Load historical OHLCV bars for backtesting from a CSV file.

Real backtests should use CSV files exported from MT5's History Center
(File > Open Data Folder, or Tools > History Center > Export), or fetched
with scripts/fetch_mt5_history.py while connected to a real broker feed.
Expected columns (case-insensitive): time/date, open, high, low, close,
volume (optional).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def load_ohlcv_csv(path: str | Path, time_column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if time_column is None:
        time_column = "time" if "time" in df.columns else "date"
    if time_column not in df.columns:
        raise ValueError(f"Could not find a time/date column in {path}: {list(df.columns)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {missing}")

    df[time_column] = pd.to_datetime(df[time_column])
    df = df.set_index(time_column).sort_index()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df[REQUIRED_COLUMNS + ["volume"]].astype(float)
