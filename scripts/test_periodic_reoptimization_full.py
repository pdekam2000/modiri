#!/usr/bin/env python3
"""The fair version of the periodic-re-optimization test: instead of the
fast/cheap small-grid optimizer (which was rejected because it kept
picking weak, overtraded compositions), re-run the FULL 3M-combination
mass search on an expanding window before each holdout period, then trade
forward with whatever it validates as the best candidate for that period.
Compares the combined result to the static champion over the same span.

This is expensive (~3 full mass searches, ~8-10 min each). Run in the
background.

    python scripts/test_periodic_reoptimization_full.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from modiri_bot.backtest.engine import BacktestEngine  # noqa: E402
from modiri_bot.backtest.mass_search import (  # noqa: E402
    approx_score_single,
    build_signals_matrix,
    build_variant_universe,
    candidate_to_ensemble,
    next_bar_returns,
    random_search_ensembles,
)
from modiri_bot.backtest.metrics import compute_metrics  # noqa: E402
from modiri_bot.backtest.optimizer import split_holdout, walk_forward_splits  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.strategies.registry import strategy_from_dict  # noqa: E402
from modiri_bot.utils.config import load_config  # noqa: E402
from run_backtest import build_config  # noqa: E402


def find_best_ensemble(train_df, bt_config, n_samples=1_500_000, top_variants=150, n_folds=4):
    variants = build_variant_universe()
    inner_train_val, inner_holdout = split_holdout(train_df, 0.85)
    folds = walk_forward_splits(inner_train_val, n_folds)

    cost_pct = (bt_config.spread_pips + bt_config.commission_per_lot / bt_config.pip_value_per_lot) \
        * bt_config.pip_size / train_df["close"].mean()

    fold_signals = []
    fold_returns = []
    for _tr, test_df in folds:
        fold_signals.append(build_signals_matrix(variants, test_df))
        fold_returns.append(next_bar_returns(test_df))

    import numpy as np
    avg_scores = np.zeros(len(variants))
    for sig, ret in zip(fold_signals, fold_returns):
        avg_scores += approx_score_single(sig, ret, cost_pct)
    avg_scores /= len(folds)
    ranked = np.argsort(-avg_scores)
    top_idx = ranked[:top_variants]
    top_vars = [variants[i] for i in top_idx]
    fold_top_signals = [sig[:, top_idx] for sig in fold_signals]

    candidates = random_search_ensembles(
        fold_top_signals, fold_returns, n_top=len(top_vars),
        n_samples=n_samples, cost_pct=cost_pct, keep_top=50,
    )

    engine = BacktestEngine(bt_config)
    validated = []
    for c in candidates[:15]:
        ens = candidate_to_ensemble(c, top_vars)
        m = compute_metrics(engine.run(inner_holdout, ens))
        validated.append((ens, m))
    validated.sort(key=lambda pair: pair[1].sharpe, reverse=True)
    return validated[0][0]


def main() -> None:
    cfg = load_config()
    df = load_ohlcv_csv(REPO_ROOT / "data" / "EURUSD_H4_202401020000_202607020800.csv")
    train_val_df, holdout_df = split_holdout(df, 0.7)

    best = json.loads((REPO_ROOT / "config" / "current_best_strategy.json").read_text())
    static_strat = strategy_from_dict(best["ensemble"]["config"])

    bt_config = build_config(cfg, "EURUSD")
    engine = BacktestEngine(bt_config)

    m_static = compute_metrics(engine.run(holdout_df, static_strat))
    print(f"Static champion over the full holdout: return {m_static.total_return_pct:+.2f}%  "
          f"Sharpe {m_static.sharpe:+.2f}  trades {m_static.num_trades}")

    n_periods = 3
    period_len = len(holdout_df) // n_periods
    period_bounds = [i * period_len for i in range(n_periods)] + [len(holdout_df)]

    combined_equity = []
    running_balance = bt_config.initial_balance
    for k in range(n_periods):
        t0 = time.time()
        period_start, period_end = period_bounds[k], period_bounds[k + 1]
        period_df = holdout_df.iloc[period_start:period_end]
        expanding_train = pd.concat([train_val_df, holdout_df.iloc[:period_start]]) if period_start > 0 else train_val_df

        picked = find_best_ensemble(expanding_train, bt_config)
        print(f"\nPeriod {k + 1} ({period_df.index[0]} -> {period_df.index[-1]}): picked {picked!r} "
              f"[search took {time.time() - t0:.0f}s]")

        period_cfg = build_config(cfg, "EURUSD")
        period_cfg.initial_balance = running_balance
        period_engine = BacktestEngine(period_cfg)
        result = period_engine.run(period_df, picked)
        m = compute_metrics(result)
        print(f"  Return this period: {m.total_return_pct:+.2f}%  Sharpe {m.sharpe:+.2f}  trades {m.num_trades}")

        running_balance = result.equity_curve.iloc[-1] if len(result.equity_curve) else running_balance
        combined_equity.append(result.equity_curve)

    full_equity = pd.concat(combined_equity)
    total_return_pct = (full_equity.iloc[-1] / bt_config.initial_balance - 1.0) * 100.0
    print(f"\nCombined full-search periodic re-optimization: {total_return_pct:+.2f}% "
          f"(final balance {full_equity.iloc[-1]:.2f})")
    print(f"Static champion (unchanged): {m_static.total_return_pct:+.2f}%")
    verdict = "periodic re-optimization WON" if total_return_pct > m_static.total_return_pct else "static champion WON"
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
