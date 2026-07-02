"""Load historical OHLCV bars for backtesting from a CSV file.

Real backtests should use CSV files exported from MT5's History Center
(File > Open Data Folder, or Tools > History Center > Export), or fetched
with scripts/fetch_mt5_history.py while connected to a real broker feed.

Handles two shapes:
  - The tidy shape scripts/fetch_mt5_history.py writes: comma-separated,
    a single "time" or "date" column, columns open/high/low/close/volume.
  - MT5 History Center's native export: tab-separated, "<DATE>" and
    "<TIME>" as separate columns, "<TICKVOL>" instead of volume.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def _clean_column(name: str) -> str:
    return name.strip().strip("<>").strip().lower()


def load_ohlcv_csv(path: str | Path, time_column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [_clean_column(c) for c in df.columns]

    if "volume" not in df.columns and "tickvol" in df.columns:
        df = df.rename(columns={"tickvol": "volume"})

    if time_column is None:
        if "date" in df.columns and "time" in df.columns:
            df["time"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                format="%Y.%m.%d %H:%M:%S", errors="coerce",
            )
            time_column = "time"
        elif "time" in df.columns:
            time_column = "time"
        elif "date" in df.columns:
            time_column = "date"
    if time_column is None or time_column not in df.columns:
        raise ValueError(f"Could not find a time/date column in {path}: {list(df.columns)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])
    df = df.set_index(time_column).sort_index()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df[REQUIRED_COLUMNS + ["volume"]].astype(float)
