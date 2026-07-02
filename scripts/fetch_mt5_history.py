#!/usr/bin/env python3
"""Export real historical bars from a running MT5 terminal to CSV, for use
with run_backtest.py / optimize_strategies.py. Windows only.

    python scripts/fetch_mt5_history.py --symbol EURUSD --timeframe H1 \\
        --from 2022-01-01 --to 2026-07-01 --out data/EURUSD_H1.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modiri_bot.data.mt5_client import MT5Client  # noqa: E402
from modiri_bot.utils.config import mt5_credentials  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True, choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    client = MT5Client(**mt5_credentials())
    client.connect()
    try:
        df = client.fetch_rates(
            args.symbol,
            args.timeframe,
            datetime.strptime(args.date_from, "%Y-%m-%d"),
            datetime.strptime(args.date_to, "%Y-%m-%d"),
        )
    finally:
        client.shutdown()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"Wrote {len(df)} real {args.symbol} {args.timeframe} bars to {args.out}")


if __name__ == "__main__":
    main()
