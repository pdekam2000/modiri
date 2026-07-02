#!/usr/bin/env python3
"""Large-scale search: build ~600-900 parameterized strategy variants, rank
them with a fast vectorized approximation, then test millions of random
ensemble-weight combinations of the best variants the same way. Only the
top handful of survivors get validated with the real bar-by-bar engine
against a holdout slice the whole search never touched -- that holdout
number, not the search score, is the one to trust.

    python scripts/mass_search.py --data data/EURUSD_H1_last4months.csv \\
        --symbol EURUSD --n-samples 2000000 --top-variants 150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from optimize_strategies import strategy_to_dict  # noqa: E402
from run_backtest import build_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--n-samples", type=int, default=2_000_000)
    parser.add_argument("--top-variants", type=int, default=150)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validate-top", type=int, default=8)
    parser.add_argument("--stop-loss-pips", type=float, default=None,
                         help="Override config.yaml's default (useful for timeframes other than H1)")
    parser.add_argument("--take-profit-pips", type=float, default=None)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "config" / "best_strategy_mass_search.json"))
    args = parser.parse_args()

    cfg = load_config()
    bt_config = build_config(cfg, args.symbol)
    if args.stop_loss_pips is not None:
        bt_config.stop_loss_pips = args.stop_loss_pips
    if args.take_profit_pips is not None:
        bt_config.take_profit_pips = args.take_profit_pips
    df = load_ohlcv_csv(args.data)

    train_val_df, holdout_df = split_holdout(df, args.train_fraction)
    folds = walk_forward_splits(train_val_df, args.folds)
    print(f"{len(df)} bars total | train/CV: {len(train_val_df)} in {len(folds)} folds | "
          f"untouched holdout: {len(holdout_df)} bars ({holdout_df.index[0]} -> {holdout_df.index[-1]})")

    cost_pct = (bt_config.spread_pips + bt_config.commission_per_lot / bt_config.pip_value_per_lot) \
        * bt_config.pip_size / df["close"].mean()

    t0 = time.time()
    variants = build_variant_universe()
    print(f"Built {len(variants)} single-strategy variants "
          f"(fast MA cross / RSI / MACD / Bollinger / Donchian, many parameterizations)")

    fold_full_signals = []
    fold_returns = []
    for _train_df, test_df in folds:
        sig = build_signals_matrix(variants, test_df)
        fold_full_signals.append(sig)
        fold_returns.append(next_bar_returns(test_df))
    print(f"Signals generated for all variants across {len(folds)} folds in {time.time() - t0:.1f}s")

    avg_single_scores = np.zeros(len(variants))
    for sig, ret in zip(fold_full_signals, fold_returns):
        avg_single_scores += approx_score_single(sig, ret, cost_pct)
    avg_single_scores /= len(folds)

    ranked = np.argsort(-avg_single_scores)
    top_idx = ranked[: args.top_variants]
    top_variants = [variants[i] for i in top_idx]
    print(f"\nTop 10 individual variants by approx walk-forward score:")
    for i in ranked[:10]:
        print(f"  {avg_single_scores[i]:+.4f}  {variants[i]!r}")

    fold_top_signals = [sig[:, top_idx] for sig in fold_full_signals]

    print(f"\nSearching {args.n_samples:,} random ensemble-weight combinations "
          f"over the top {args.top_variants} variants...")
    t0 = time.time()
    candidates = random_search_ensembles(
        fold_top_signals, fold_returns, n_top=len(top_variants),
        n_samples=args.n_samples, cost_pct=cost_pct, keep_top=200,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({args.n_samples / max(elapsed, 1e-9):,.0f} combos/sec)")

    print(f"\nTop 10 ensemble combinations by approx walk-forward score:")
    for c in candidates[:10]:
        ens = candidate_to_ensemble(c, top_variants)
        print(f"  {c.score:+.4f}  {ens!r}")

    # Validate the real single best variant, plus the top N ensemble
    # candidates, with the actual engine on the untouched holdout slice.
    engine = BacktestEngine(bt_config)
    best_single_variant = variants[ranked[0]]
    single_metrics = compute_metrics(engine.run(holdout_df, best_single_variant))

    print(f"\n=== HONEST holdout validation (real engine, real costs, never seen during search) ===")
    print(f"\nBest individual variant: {best_single_variant!r}")
    print(single_metrics.summary())

    validated = []
    for c in candidates[: args.validate_top]:
        ens = candidate_to_ensemble(c, top_variants)
        m = compute_metrics(engine.run(holdout_df, ens))
        validated.append((ens, m))

    validated.sort(key=lambda pair: pair[1].sharpe, reverse=True)
    best_ensemble, best_ensemble_metrics = validated[0]
    print(f"\nBest ensemble on real holdout (out of top {args.validate_top} search candidates): {best_ensemble!r}")
    print(best_ensemble_metrics.summary())

    print(f"\nAll {args.validate_top} validated ensemble candidates, ranked by real holdout Sharpe:")
    for ens, m in validated:
        print(f"  Sharpe {m.sharpe:+.2f}  return {m.total_return_pct:+.2f}%  "
              f"maxDD {m.max_drawdown_pct:.1f}%  trades {m.num_trades}  {ens!r}")

    if single_metrics.sharpe <= 0 and best_ensemble_metrics.sharpe <= 0:
        print(
            "\n[!] Even after searching millions of combinations, nothing survived "
            "validation on the untouched holdout. That's the honest result on this "
            "dataset -- it means no robust edge was found, not that the search failed."
        )

    # Split-holdout consistency check: a handful of trades over one holdout
    # window can look good by luck. Require the winner to also hold up on
    # each half of the holdout separately, not just combined.
    mid = len(holdout_df) // 2
    holdout_a, holdout_b = holdout_df.iloc[:mid], holdout_df.iloc[mid:]
    metrics_a = compute_metrics(engine.run(holdout_a, best_ensemble))
    metrics_b = compute_metrics(engine.run(holdout_b, best_ensemble))
    print(f"\n=== Split-holdout consistency check for the winning ensemble ===")
    print(f"First half  ({holdout_a.index[0]} -> {holdout_a.index[-1]}, {len(holdout_a)} bars): "
          f"return {metrics_a.total_return_pct:+.2f}%  Sharpe {metrics_a.sharpe:+.2f}  trades {metrics_a.num_trades}")
    print(f"Second half ({holdout_b.index[0]} -> {holdout_b.index[-1]}, {len(holdout_b)} bars): "
          f"return {metrics_b.total_return_pct:+.2f}%  Sharpe {metrics_b.sharpe:+.2f}  trades {metrics_b.num_trades}")
    if metrics_a.total_return_pct > 0 and metrics_b.total_return_pct > 0:
        print("Both halves positive -- more consistent with a real, if modest, effect than with luck.")
    else:
        print("[!] Not positive on both halves -- treat the combined holdout number with real skepticism; "
              "it may be carried by one lucky stretch rather than a repeatable edge.")

    output = {
        "n_variants_generated": len(variants),
        "n_ensemble_combos_tested": args.n_samples,
        "best_single": {
            "config": strategy_to_dict(best_single_variant),
            "holdout_sharpe": single_metrics.sharpe,
            "holdout_total_return_pct": single_metrics.total_return_pct,
        },
        "best_ensemble": {
            "config": strategy_to_dict(best_ensemble),
            "holdout_sharpe": best_ensemble_metrics.sharpe,
            "holdout_total_return_pct": best_ensemble_metrics.total_return_pct,
            "holdout_first_half_return_pct": metrics_a.total_return_pct,
            "holdout_second_half_return_pct": metrics_b.total_return_pct,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote result to {args.output}")


if __name__ == "__main__":
    main()
