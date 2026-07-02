#!/usr/bin/env python3
"""Final round of tests: trailing stop, weekend gap avoidance, and broker
cost sensitivity, all against the production baseline (champion ensemble
+ time stop + volatility filter, as wired into config.yaml).

    python scripts/test_final_ideas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from modiri_bot.backtest.engine import BacktestEngine  # noqa: E402
from modiri_bot.backtest.metrics import compute_metrics  # noqa: E402
from modiri_bot.backtest.optimizer import split_holdout  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.strategies.registry import strategy_from_dict  # noqa: E402
from modiri_bot.utils.config import load_config  # noqa: E402
from run_backtest import build_config  # noqa: E402


def report(name, strat, bt_config, holdout_df, holdout_a, holdout_b):
    engine = BacktestEngine(bt_config)
    r_full = engine.run(holdout_df, strat)
    m_full = compute_metrics(r_full)
    m_a = compute_metrics(engine.run(holdout_a, strat))
    m_b = compute_metrics(engine.run(holdout_b, strat))
    trades = r_full.trades_df()
    wins = (trades["pnl"] > 0).sum() if len(trades) else 0
    win_rate = wins / len(trades) * 100 if len(trades) else 0.0
    gross_profit = trades[trades["pnl"] > 0]["pnl"].sum() if len(trades) else 0.0
    gross_loss = -trades[trades["pnl"] <= 0]["pnl"].sum() if len(trades) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    consistent = m_a.total_return_pct > 0 and m_b.total_return_pct > 0
    print(f"\n--- {name} ---")
    print(f"Net return: {m_full.total_return_pct:+.2f}%   Max DD: {m_full.max_drawdown_pct:.2f}%   "
          f"Sharpe: {m_full.sharpe:+.2f}   Sortino: {m_full.sortino:+.2f}")
    print(f"Win rate: {win_rate:.1f}%   Profit factor: {profit_factor:.2f}   Trades: {m_full.num_trades}")
    print(f"  1st half: {m_a.total_return_pct:+.2f}% (Sharpe {m_a.sharpe:+.2f})   "
          f"2nd half: {m_b.total_return_pct:+.2f}% (Sharpe {m_b.sharpe:+.2f})   "
          f"consistent: {'YES' if consistent else 'NO'}")
    return m_full, consistent


def main() -> None:
    cfg = load_config()
    df = load_ohlcv_csv(REPO_ROOT / "data" / "EURUSD_H4_202401020000_202607020800.csv")
    _train_val_df, holdout_df = split_holdout(df, 0.7)
    mid = len(holdout_df) // 2
    holdout_a, holdout_b = holdout_df.iloc[:mid], holdout_df.iloc[mid:]

    best = json.loads((REPO_ROOT / "config" / "current_best_strategy.json").read_text())
    strat = strategy_from_dict(best["ensemble"]["config"])

    report("0. Production baseline (time stop + volatility filter)",
           strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b)

    # 1. Trailing stop, a few start/distance combinations.
    for start_r, dist_r in [(0.5, 0.5), (1.0, 1.0), (1.0, 0.5), (1.5, 1.0)]:
        cfg_ts = build_config(cfg, "EURUSD")
        cfg_ts.use_trailing_stop = True
        cfg_ts.trailing_start_r_multiple = start_r
        cfg_ts.trailing_distance_r_multiple = dist_r
        report(f"1. Trailing stop (start={start_r}R, distance={dist_r}R)",
               strat, cfg_ts, holdout_df, holdout_a, holdout_b)

    # 2. Weekend gap avoidance.
    cfg_wk = build_config(cfg, "EURUSD")
    cfg_wk.close_before_weekend = True
    cfg_wk.friday_close_hour = 20
    report("2. Weekend gap avoidance (close by Friday 20:00)",
           strat, cfg_wk, holdout_df, holdout_a, holdout_b)

    # 3. Broker cost sensitivity: ECN-tight vs. current vs. wide/retail.
    cost_profiles = {
        "ECN tight (0.2 pip spread, $3.5/lot)": (0.2, 3.5),
        "Current assumption (1.2 pip, $7/lot)": (1.2, 7.0),
        "Wide/retail (2.5 pip, $10/lot)": (2.5, 10.0),
    }
    for name, (spread, commission) in cost_profiles.items():
        cfg_cost = build_config(cfg, "EURUSD")
        cfg_cost.spread_pips = spread
        cfg_cost.commission_per_lot = commission
        report(f"3. Broker cost sensitivity -- {name}", strat, cfg_cost, holdout_df, holdout_a, holdout_b)


if __name__ == "__main__":
    main()
