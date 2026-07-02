from __future__ import annotations

from datetime import timedelta

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    return timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
