#!/usr/bin/env python3
"""Walk-forward search across all strategies + ensemble weights, then report
an honest out-of-sample result on a holdout slice the search never saw.

Writes the winning single-strategy and ensemble configs to
config/best_strategy.json so scripts/run_live.py can pick one up.

Examples:
    python scripts/optimize_strategies.py --demo
    python scripts/optimize_strategies.py --data data/EURUSD_H1.csv --symbol EURUSD
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modiri_bot.backtest.optimizer import optimize_strategies  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.data.synthetic import generate_synthetic_ohlcv  # noqa: E402
from modiri_bot.strategies.ensemble import EnsembleStrategy  # noqa: E402
from modiri_bot.utils.config import load_config  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_backtest import build_config  # noqa: E402


def strategy_to_dict(strategy) -> dict:
    if isinstance(strategy, EnsembleStrategy):
        return {
            "type": "ensemble",
            "threshold": strategy.threshold,
            "members": [
                {"name": s.name, "params": s.params, "weight": w}
                for s, w in zip(strategy.strategies, strategy.weights)
            ],
        }
    return {"type": "single", "name": strategy.name, "params": strategy.params}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--objective", type=str, default=None,
                         choices=["sharpe", "sortino", "calmar", "total_return_pct"])
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "config" / "best_strategy.json"))
    args = parser.parse_args()

    if not args.data and not args.demo:
        parser.error("pass --data <csv> for a real search, or --demo for a synthetic sanity check")

    cfg = load_config()
    bt_config = build_config(cfg, args.symbol)
    objective = args.objective or cfg["backtest"]["objective"]
    folds = args.folds or cfg["backtest"]["walk_forward_folds"]
    train_fraction = args.train_fraction or cfg["backtest"]["train_test_split"]

    if args.demo:
        print("[!] Using SYNTHETIC random-walk data — expect the holdout result to "
              "show a loss (net of spread/commission), since a random walk has no "
              "real edge to find. This is here to prove the search doesn't fool "
              "itself, not to produce a tradeable strategy.")
        df = generate_synthetic_ohlcv(n_bars=6000)
    else:
        df = load_ohlcv_csv(args.data)

    print(f"Optimizing on {len(df)} bars, objective={objective}, folds={folds}, "
          f"train_fraction={train_fraction}")
    report = optimize_strategies(
        df, bt_config, train_fraction=train_fraction, n_folds=folds, objective=objective
    )

    print("\n--- Per-strategy walk-forward CV score (avg out-of-sample %s across folds) ---" % objective)
    for s in report.per_strategy_scores:
        print(f"  {s.strategy!r}: {s.oos_score:.3f}")

    print(f"\nBest single strategy: {report.best_single.strategy!r} "
          f"(CV score {report.best_single.oos_score:.3f})")
    print("Holdout performance (never seen during search):")
    print(report.holdout_single_metrics.summary())

    print(f"\nBest ensemble: {report.best_ensemble.strategy!r} "
          f"(CV score {report.best_ensemble.oos_score:.3f})")
    print("Holdout performance (never seen during search):")
    print(report.holdout_ensemble_metrics.summary())

    if report.holdout_single_metrics.sharpe <= 0 and report.holdout_ensemble_metrics.sharpe <= 0:
        print(
            "\n[!] Neither candidate held up on the untouched holdout data. That's "
            "a real result, not a bug — publish it, don't paper over it by "
            "re-running the search until something looks good."
        )

    output = {
        "objective": objective,
        "single": {
            "config": strategy_to_dict(report.best_single.strategy),
            "cv_score": report.best_single.oos_score,
            "holdout_sharpe": report.holdout_single_metrics.sharpe,
            "holdout_total_return_pct": report.holdout_single_metrics.total_return_pct,
        },
        "ensemble": {
            "config": strategy_to_dict(report.best_ensemble.strategy),
            "cv_score": report.best_ensemble.oos_score,
            "holdout_sharpe": report.holdout_ensemble_metrics.sharpe,
            "holdout_total_return_pct": report.holdout_ensemble_metrics.total_return_pct,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote best configs to {args.output}")


if __name__ == "__main__":
    main()
