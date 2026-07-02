#!/usr/bin/env python3
"""Run the bot live against a real (ideally demo, at first) MT5 account.

Windows only — the MetaTrader5 package needs a running MT5 terminal.
Reads credentials from .env (see .env.example) and the winning strategy
from config/best_strategy.json (see scripts/optimize_strategies.py), unless
overridden on the command line.

    python scripts/run_live.py --symbol EURUSD --use ensemble --poll-seconds 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modiri_bot.data.mt5_client import MT5Client  # noqa: E402
from modiri_bot.live.live_trader import LiveTrader  # noqa: E402
from modiri_bot.risk.risk_manager import RiskLimits  # noqa: E402
from modiri_bot.strategies.registry import strategy_from_dict  # noqa: E402
from modiri_bot.utils.config import get_symbol_config, load_config, mt5_credentials  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--use", choices=["single", "ensemble"], default="ensemble",
                         help="Which winning config from best_strategy.json to trade")
    parser.add_argument("--best-strategy-file", type=str,
                         default=str(REPO_ROOT / "config" / "best_strategy.json"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config()
    sym_cfg = get_symbol_config(cfg, args.symbol)
    risk_cfg = cfg["risk"]

    best_path = Path(args.best_strategy_file)
    if not best_path.exists():
        raise SystemExit(
            f"{best_path} not found — run scripts/optimize_strategies.py first to "
            "produce a strategy config, or point --best-strategy-file elsewhere."
        )
    best = json.loads(best_path.read_text())[args.use]
    strategy = strategy_from_dict(best["config"])
    print(f"Trading strategy: {strategy!r}")
    print(f"Holdout backtest Sharpe was {best['holdout_sharpe']:.2f} "
          f"and holdout return was {best['holdout_total_return_pct']:+.2f}% — "
          "past backtest performance, not a live guarantee.")

    creds = mt5_credentials()
    client = MT5Client(**creds)
    client.connect()

    risk_limits = RiskLimits(
        risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
        max_concurrent_trades=risk_cfg["max_concurrent_trades"],
        max_daily_loss_pct=risk_cfg["max_daily_loss_pct"],
        max_drawdown_pct=risk_cfg["max_drawdown_pct"],
    )

    trader = LiveTrader(
        client=client,
        symbol_cfg=sym_cfg,
        strategy=strategy,
        risk_limits=risk_limits,
        stop_loss_pips=risk_cfg["default_stop_loss_pips"],
        take_profit_pips=risk_cfg["default_take_profit_pips"],
        magic=cfg["mt5"]["magic_number"],
        deviation=cfg["mt5"]["deviation_points"],
    )

    try:
        trader.run_forever(poll_seconds=args.poll_seconds)
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
