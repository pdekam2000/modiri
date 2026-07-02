#!/usr/bin/env python3
"""Compare the current best validated ensemble against a few risk-management
variants: ATR-adaptive stops, two-stage take-profit, and looser entry
thresholds meant to raise trade frequency. Each variant is checked against
the same split-holdout consistency test as the main search, so "looks
better on the combined number" isn't enough -- it has to hold up on both
halves independently.

    python scripts/compare_risk_variants.py
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
    m_full = compute_metrics(engine.run(holdout_df, strat))
    m_a = compute_metrics(engine.run(holdout_a, strat))
    m_b = compute_metrics(engine.run(holdout_b, strat))
    print(f"\n--- {name} ---")
    print(f"Full 9mo:   return {m_full.total_return_pct:+.2f}%  Sharpe {m_full.sharpe:+.2f}  "
          f"maxDD {m_full.max_drawdown_pct:.1f}%  trades {m_full.num_trades}  winrate {m_full.win_rate_pct:.0f}%")
    print(f"1st half:   return {m_a.total_return_pct:+.2f}%  Sharpe {m_a.sharpe:+.2f}  trades {m_a.num_trades}")
    print(f"2nd half:   return {m_b.total_return_pct:+.2f}%  Sharpe {m_b.sharpe:+.2f}  trades {m_b.num_trades}")
    consistent = m_a.total_return_pct > 0 and m_b.total_return_pct > 0
    print(f"Consistent both halves: {'YES' if consistent else 'NO'}")
    return m_full, consistent


def main() -> None:
    cfg = load_config()
    df = load_ohlcv_csv(REPO_ROOT / "data" / "EURUSD_H4_202401020000_202607020800.csv")
    _train_val_df, holdout_df = split_holdout(df, 0.7)
    mid = len(holdout_df) // 2
    holdout_a, holdout_b = holdout_df.iloc[:mid], holdout_df.iloc[mid:]

    best = json.loads((REPO_ROOT / "config" / "current_best_strategy.json").read_text())
    base_strat = strategy_from_dict(best["ensemble"]["config"])

    report("A. Baseline (fixed 40/80 pips, single TP)", base_strat,
           build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b)

    cfg_b = build_config(cfg, "EURUSD")
    cfg_b.use_atr_stops = True
    cfg_b.atr_period = 14
    cfg_b.atr_sl_mult = 1.5
    cfg_b.atr_tp_mult = 3.0
    report("B. ATR-adaptive stops (1.5x/3.0x ATR)", base_strat, cfg_b, holdout_df, holdout_a, holdout_b)

    cfg_c = build_config(cfg, "EURUSD")
    cfg_c.use_partial_tp = True
    cfg_c.tp1_r_multiple = 1.0
    cfg_c.tp1_close_fraction = 0.5
    cfg_c.move_sl_to_breakeven_after_tp1 = True
    report("C. Two-stage TP (50% at 1R, breakeven, rest to 2R)", base_strat, cfg_c, holdout_df, holdout_a, holdout_b)

    loose_config = dict(best["ensemble"]["config"])
    loose_config["threshold"] = 0.10
    loose_strat = strategy_from_dict(loose_config)
    report("D. Loosened threshold 0.33 -> 0.10 (more trades)", loose_strat,
           build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b)

    loose_config2 = dict(best["ensemble"]["config"])
    loose_config2["threshold"] = 0.0
    loose_strat2 = strategy_from_dict(loose_config2)
    report("E. Loosened threshold 0.33 -> 0.0 (most trades)", loose_strat2,
           build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b)

    cfg_f = build_config(cfg, "EURUSD")
    cfg_f.risk_per_trade_pct = 2.0
    report("F. Loosened threshold 0.10 + 2% risk per trade", loose_strat, cfg_f, holdout_df, holdout_a, holdout_b)


if __name__ == "__main__":
    main()
