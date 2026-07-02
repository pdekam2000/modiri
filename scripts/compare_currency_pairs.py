#!/usr/bin/env python3
"""Compares the trading system across 6 real currency pairs (H4 data):
EURUSD (existing champion, config/current_best_strategy.json) plus 5 newly
supplied pairs -- AUDUSD, GBPUSD, USDCAD, USDCHF, USDJPY.

For each NEW pair, runs the same full mass-combination search used to find
the EURUSD champion (build_variant_universe -> walk-forward approximate
scoring -> random_search_ensembles -> validate top candidates with the real
bar-by-bar engine), on that pair's own 70% train/validation slice, then
reports true out-of-sample performance on its untouched 30% holdout -- same
methodology, same discipline (split-holdout consistency check) used
throughout this project.

Reported WITHOUT the 3 EURUSD-tuned risk overlays (time stop / volatility
filter / trailing stop) by default, since those parameters were tuned
specifically for EURUSD H4 and re-using them elsewhere unvalidated would
conflate "is this pair's price action profitable" with "did we get lucky
re-using someone else's overlay settings." Every pair's OWN best ensemble
composition is still found by its own dedicated search.

    python scripts/compare_currency_pairs.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from modiri_bot.backtest.engine import BacktestConfig, BacktestEngine  # noqa: E402
from modiri_bot.backtest.metrics import compute_metrics  # noqa: E402
from modiri_bot.backtest.optimizer import split_holdout  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.strategies.registry import strategy_from_dict  # noqa: E402
from test_periodic_reoptimization_full import find_best_ensemble  # noqa: E402

# name -> (csv filename, pip_size). All non-JPY pairs here are 5-digit
# brokers (pip = 10x point); USDJPY is a 3-digit broker (pip = 0.01).
PAIRS = {
    "AUDUSD": ("AUDUSD_H4_202108020000_202607020800.csv", 0.0001),
    "GBPUSD": ("GBPUSD_H4_202108020000_202607020800.csv", 0.0001),
    "USDCAD": ("USDCAD_H4_202501020000_202607020800.csv", 0.0001),
    "USDCHF": ("USDCHF_H4_202108020000_202607020800.csv", 0.0001),
    "USDJPY": ("USDJPY_H4_202108020000_202607020800.csv", 0.01),
}

# Same cost assumptions for every pair so the comparison isn't confounded by
# guessed per-broker spread/commission differences -- a simplification worth
# re-checking against your actual broker's real costs per symbol.
COMMON_KWARGS = dict(
    initial_balance=1000.0,
    pip_value_per_lot=10.0,
    min_lot=0.01,
    lot_step=0.01,
    spread_pips=1.2,
    commission_per_lot=7.0,
    risk_per_trade_pct=1.0,
    stop_loss_pips=40.0,
    take_profit_pips=80.0,
    max_daily_loss_pct=3.0,
    max_drawdown_pct=15.0,
)


def report_pair(name: str, m_full, m_a, m_b, num_bars: int, date_range: str) -> dict:
    consistent = m_a.total_return_pct > 0 and m_b.total_return_pct > 0
    print(f"\n--- {name} ({num_bars} H4 bars, {date_range}) ---")
    print(f"Holdout return: {m_full.total_return_pct:+.2f}%   Max DD: {m_full.max_drawdown_pct:.2f}%   "
          f"Sharpe: {m_full.sharpe:+.2f}   Sortino: {m_full.sortino:+.2f}")
    print(f"Win rate: {m_full.win_rate_pct:.1f}%   Profit factor: {m_full.profit_factor:.2f}   "
          f"Trades: {m_full.num_trades}")
    print(f"  1st half: {m_a.total_return_pct:+.2f}% (Sharpe {m_a.sharpe:+.2f})   "
          f"2nd half: {m_b.total_return_pct:+.2f}% (Sharpe {m_b.sharpe:+.2f})   "
          f"consistent: {'YES' if consistent else 'NO'}")
    return {
        "return_pct": m_full.total_return_pct, "max_dd_pct": m_full.max_drawdown_pct,
        "sharpe": m_full.sharpe, "sortino": m_full.sortino, "win_rate_pct": m_full.win_rate_pct,
        "profit_factor": m_full.profit_factor, "num_trades": m_full.num_trades,
        "consistent": consistent,
    }


def main() -> None:
    results = {}

    # EURUSD: reuse the already-validated champion, no need to re-search.
    best = json.loads((REPO_ROOT / "config" / "current_best_strategy.json").read_text())
    eur_metrics = best["ensemble"]["production_holdout_metrics"]
    print("--- EURUSD (existing champion, WITH its 3 production risk overlays) ---")
    print(f"Holdout return: {eur_metrics['total_return_pct']:+.2f}%   "
          f"Max DD: {eur_metrics['max_drawdown_pct']:.2f}%   Sharpe: {eur_metrics['sharpe']:+.2f}   "
          f"Win rate: {eur_metrics['win_rate_pct']:.1f}%   PF: {eur_metrics['profit_factor']:.2f}   "
          f"Trades: {eur_metrics['num_trades']}")
    results["EURUSD"] = {
        "return_pct": eur_metrics["total_return_pct"], "max_dd_pct": eur_metrics["max_drawdown_pct"],
        "sharpe": eur_metrics["sharpe"], "sortino": eur_metrics["sortino"],
        "win_rate_pct": eur_metrics["win_rate_pct"], "profit_factor": eur_metrics["profit_factor"],
        "num_trades": eur_metrics["num_trades"], "consistent": True,
        "note": "with production overlays (time stop/vol filter/trailing stop)",
    }

    for pair_name, (filename, pip_size) in PAIRS.items():
        t0 = time.time()
        df = load_ohlcv_csv(REPO_ROOT / "data" / filename)
        train_val_df, holdout_df = split_holdout(df, 0.7)
        mid = len(holdout_df) // 2
        holdout_a, holdout_b = holdout_df.iloc[:mid], holdout_df.iloc[mid:]

        bt_config = BacktestConfig(pip_size=pip_size, **COMMON_KWARGS)

        print(f"\n[{pair_name}] searching on {len(train_val_df)} train bars "
              f"({df.index[0]} -> {train_val_df.index[-1]})...")
        best_ensemble = find_best_ensemble(train_val_df, bt_config)
        print(f"[{pair_name}] picked: {best_ensemble!r}  [search took {time.time() - t0:.0f}s]")

        engine = BacktestEngine(bt_config)
        m_full = compute_metrics(engine.run(holdout_df, best_ensemble))
        m_a = compute_metrics(engine.run(holdout_a, best_ensemble))
        m_b = compute_metrics(engine.run(holdout_b, best_ensemble))

        date_range = f"{df.index[0].date()} -> {df.index[-1].date()}"
        results[pair_name] = report_pair(pair_name, m_full, m_a, m_b, len(df), date_range)
        results[pair_name]["strategy"] = repr(best_ensemble)
        results[pair_name]["note"] = "own best ensemble, no EURUSD-tuned overlays"

    print("\n" + "=" * 100)
    print(f"{'Pair':<10}{'Return%':>10}{'Sharpe':>10}{'MaxDD%':>10}{'WinRate%':>10}{'PF':>8}{'Trades':>9}{'Consistent':>12}")
    print("-" * 100)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["sharpe"], reverse=True)
    for name, r in ranked:
        print(f"{name:<10}{r['return_pct']:>10.2f}{r['sharpe']:>10.2f}{r['max_dd_pct']:>10.2f}"
              f"{r['win_rate_pct']:>10.1f}{r['profit_factor']:>8.2f}{r['num_trades']:>9}"
              f"{'YES' if r['consistent'] else 'NO':>12}")
    print("=" * 100)

    credible = [(n, r) for n, r in ranked if r["consistent"]]
    if credible:
        winner = credible[0]
        print(f"\nBest pair by Sharpe among split-holdout-consistent results: {winner[0]} "
              f"(Sharpe {winner[1]['sharpe']:+.2f}, return {winner[1]['return_pct']:+.2f}%)")
    else:
        print("\nNo pair passed the split-holdout consistency check -- no credible winner.")

    out_path = REPO_ROOT / "config" / "currency_pair_comparison.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
