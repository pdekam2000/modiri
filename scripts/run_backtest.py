#!/usr/bin/env python3
"""Run a single backtest and print a performance report.

Examples:
    # Quick sanity check against synthetic data (no real edge, just wiring).
    python scripts/run_backtest.py --demo --strategy ma_crossover

    # Real backtest against historical bars exported from MT5.
    python scripts/run_backtest.py --data data/EURUSD_H1.csv --symbol EURUSD \\
        --strategy donchian_breakout --params '{"period": 20}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modiri_bot.backtest.engine import BacktestConfig, BacktestEngine  # noqa: E402
from modiri_bot.backtest.metrics import compute_metrics  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.data.synthetic import generate_synthetic_ohlcv  # noqa: E402
from modiri_bot.strategies.registry import STRATEGY_CLASSES  # noqa: E402
from modiri_bot.utils.config import get_symbol_config, load_config  # noqa: E402


def build_config(cfg: dict, symbol_name: str) -> BacktestConfig:
    sym = get_symbol_config(cfg, symbol_name)
    risk = cfg["risk"]
    return BacktestConfig(
        initial_balance=cfg["account"]["initial_balance"],
        pip_size=sym.pip_size,
        pip_value_per_lot=sym.pip_value_per_lot,
        min_lot=sym.min_lot,
        lot_step=sym.lot_step,
        spread_pips=sym.spread_pips,
        commission_per_lot=sym.commission_per_lot,
        risk_per_trade_pct=risk["risk_per_trade_pct"],
        stop_loss_pips=risk["default_stop_loss_pips"],
        take_profit_pips=risk["default_take_profit_pips"],
        max_daily_loss_pct=risk["max_daily_loss_pct"],
        max_drawdown_pct=risk["max_drawdown_pct"],
        max_hold_bars=risk.get("max_hold_bars"),
        use_volatility_filter=risk.get("use_volatility_filter", False),
        volatility_percentile_threshold=risk.get("volatility_percentile_threshold", 95.0),
        volatility_size_mult=risk.get("volatility_size_mult", 0.5),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, help="Path to an OHLCV CSV file")
    parser.add_argument("--demo", action="store_true", help="Use synthetic random-walk data instead of --data")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol name from config.yaml")
    parser.add_argument("--strategy", type=str, default="ma_crossover", choices=list(STRATEGY_CLASSES))
    parser.add_argument("--params", type=str, default="{}", help="JSON dict of strategy parameters")
    parser.add_argument("--plot", type=str, default=None, help="Save an equity-curve PNG to this path")
    args = parser.parse_args()

    if not args.data and not args.demo:
        parser.error("pass --data <csv> for a real backtest, or --demo for a synthetic sanity check")

    cfg = load_config()
    bt_config = build_config(cfg, args.symbol)

    if args.demo:
        print("[!] Using SYNTHETIC random-walk data — this only checks that the code "
              "runs, it says nothing about real profitability.")
        df = generate_synthetic_ohlcv()
    else:
        df = load_ohlcv_csv(args.data)

    strategy_cls = STRATEGY_CLASSES[args.strategy]
    strategy = strategy_cls(**json.loads(args.params))

    engine = BacktestEngine(bt_config)
    result = engine.run(df, strategy)
    metrics = compute_metrics(result)

    print(f"Strategy: {strategy!r}")
    print(f"Bars: {len(df)}  ({df.index[0]} -> {df.index[-1]})")
    print()
    print(metrics.summary())
    print()
    print("Monthly returns (%):")
    print(metrics.monthly_returns_pct.round(2).to_string())

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        result.equity_curve.plot(ax=ax)
        ax.set_title(f"Equity curve — {strategy}")
        ax.set_ylabel("Account equity")
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot)
        print(f"\nSaved equity curve to {args.plot}")


if __name__ == "__main__":
    main()
