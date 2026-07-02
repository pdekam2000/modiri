import pandas as pd
import pytest

from modiri_bot.data.csv_loader import load_ohlcv_csv


def test_loads_standard_mt5_export(tmp_path):
    csv_path = tmp_path / "eurusd.csv"
    csv_path.write_text(
        "time,open,high,low,close,volume\n"
        "2024-01-01 00:00:00,1.1000,1.1010,1.0995,1.1005,120\n"
        "2024-01-01 01:00:00,1.1005,1.1020,1.1000,1.1015,140\n"
    )
    df = load_ohlcv_csv(csv_path)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df["close"].iloc[0] == 1.1005


def test_fills_missing_volume_with_zero(tmp_path):
    csv_path = tmp_path / "no_volume.csv"
    csv_path.write_text(
        "date,open,high,low,close\n"
        "2024-01-01,1.1,1.2,1.0,1.15\n"
    )
    df = load_ohlcv_csv(csv_path)
    assert df["volume"].iloc[0] == 0.0


def test_raises_on_missing_required_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("time,open,close\n2024-01-01,1.1,1.15\n")
    with pytest.raises(ValueError):
        load_ohlcv_csv(csv_path)
