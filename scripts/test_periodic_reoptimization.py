#!/usr/bin/env python3
"""Simulates a more realistic deployment: instead of picking one fixed
ensemble composition and trading it for the whole 9-month holdout, split
the holdout into periods and re-run the (fast, small-grid) walk-forward
optimizer on an expanding window before each period, trading forward with
whatever it picks. Compares the combined result to keeping the static
champion the whole time.

    python scripts/test_periodic_reoptimization.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modiri_bot.backtest.engine import BacktestEngine  # noqa: E402
from modiri_bot.backtest.metrics import compute_metrics  # noqa: E402
from modiri_bot.backtest.optimizer import optimize_strategies, split_holdout  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.strategies.registry import strategy_from_dict  # noqa: E402
from modiri_bot.utils.config import load_config  # noqa: E402
from run_backtest import build_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    df = load_ohlcv_csv(REPO_ROOT / "data" / "EURUSD_H4_202401020000_202607020800.csv")
    train_val_df, holdout_df = split_holdout(df, 0.7)

    best = json.loads((REPO_ROOT / "config" / "current_best_strategy.json").read_text())
    static_strat = strategy_from_dict(best["ensemble"]["config"])

    bt_config = build_config(cfg, "EURUSD")
    engine = BacktestEngine(bt_config)

    n_periods = 3
    period_len = len(holdout_df) // n_periods
    period_bounds = [i * period_len for i in range(n_periods)] + [len(holdout_df)]

    print(f"Static champion over the full {len(holdout_df)}-bar holdout (for reference):")
    m_static = compute_metrics(engine.run(holdout_df, static_strat))
    print(f"  return {m_static.total_return_pct:+.2f}%  Sharpe {m_static.sharpe:+.2f}  trades {m_static.num_trades}")

    print(f"\nPeriodic re-optimization ({n_periods} periods, expanding window, re-fit before each):")
    combined_equity = []
    combined_trades = 0
    running_balance = bt_config.initial_balance
    for k in range(n_periods):
        period_start, period_end = period_bounds[k], period_bounds[k + 1]
        period_df = holdout_df.iloc[period_start:period_end]
        expanding_train = pd.concat([train_val_df, holdout_df.iloc[:period_start]]) if period_start > 0 else train_val_df

        report = optimize_strategies(
            expanding_train, bt_config, train_fraction=0.85, n_folds=3, objective="sharpe"
        )
        picked = report.best_ensemble.strategy
        print(f"\n  Period {k + 1} ({period_df.index[0]} -> {period_df.index[-1]}, {len(period_df)} bars):")
        print(f"    Picked: {picked!r}")

        period_cfg = build_config(cfg, "EURUSD")
        period_cfg.initial_balance = running_balance
        period_engine = BacktestEngine(period_cfg)
        result = period_engine.run(period_df, picked)
        m = compute_metrics(result)
        print(f"    Return this period: {m.total_return_pct:+.2f}%  Sharpe {m.sharpe:+.2f}  trades {m.num_trades}")

        running_balance = result.equity_curve.iloc[-1] if len(result.equity_curve) else running_balance
        combined_equity.append(result.equity_curve)
        combined_trades += m.num_trades

    full_equity = pd.concat(combined_equity)
    total_return_pct = (full_equity.iloc[-1] / bt_config.initial_balance - 1.0) * 100.0
    print(f"\nCombined periodic re-optimization result:")
    print(f"  Total return: {total_return_pct:+.2f}%   Total trades: {combined_trades}")
    print(f"  Final balance: {full_equity.iloc[-1]:.2f} (started {bt_config.initial_balance:.2f})")

    print(f"\nStatic champion (unchanged the whole time): {m_static.total_return_pct:+.2f}% return, "
          f"{m_static.num_trades} trades")
    verdict = "periodic re-optimization WON" if total_return_pct > m_static.total_return_pct else "static champion WON"
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
