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
                         default=str(REPO_ROOT / "config" / "current_best_strategy.json"))
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
    metrics = best.get("production_holdout_metrics")
    if metrics:
        print(f"Holdout backtest (with production risk overlays): return "
              f"{metrics['total_return_pct']:+.2f}%, Sharpe {metrics['sharpe']:.2f}, "
              f"max drawdown {metrics['max_drawdown_pct']:.2f}%, "
              f"{metrics['num_trades']} trades over ~9 months — "
              "past backtest performance, not a live guarantee.")
    else:
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
        max_hold_bars=risk_cfg.get("max_hold_bars"),
        use_volatility_filter=risk_cfg.get("use_volatility_filter", False),
        volatility_percentile_threshold=risk_cfg.get("volatility_percentile_threshold", 95.0),
        volatility_size_mult=risk_cfg.get("volatility_size_mult", 0.5),
        use_trailing_stop=risk_cfg.get("use_trailing_stop", False),
        trailing_start_r_multiple=risk_cfg.get("trailing_start_r_multiple", 0.4),
        trailing_distance_r_multiple=risk_cfg.get("trailing_distance_r_multiple", 0.4),
    )

    try:
        trader.run_forever(poll_seconds=args.poll_seconds)
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
