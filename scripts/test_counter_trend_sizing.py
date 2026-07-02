#!/usr/bin/env python3
"""Test reduced-risk variants for counter-trend trades (against the
200-period EMA) instead of excluding them outright, since excluding them
was shown to gut the strategy (most of its winners are counter-trend too).

Variants:
  1. Counter-trend trades at 50% position size.
  2. Counter-trend trades at 25% position size.
  3. Counter-trend trades with a 20% tighter stop loss.
  4. Counter-trend trades with both 50% size and the tighter stop.

Reports holdout-only Net return, Max drawdown, Sharpe, and number of
trades for each variant against the current baseline, plus the same
split-holdout consistency check used throughout this project, since a
combined holdout number alone isn't enough to trust a result.

    python scripts/test_counter_trend_sizing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from modiri_bot.backtest.engine import BacktestEngine  # noqa: E402
from modiri_bot.backtest.metrics import compute_metrics  # noqa: E402
from modiri_bot.backtest.optimizer import split_holdout  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.strategies.indicators import ema  # noqa: E402
from modiri_bot.strategies.registry import strategy_from_dict  # noqa: E402
from modiri_bot.utils.config import load_config  # noqa: E402
from run_backtest import build_config  # noqa: E402


def counter_trend_mask(df: pd.DataFrame, strategy, trend_period: int = 200) -> pd.Series:
    """True on bars where the strategy's signal opposes the major trend."""
    signal = strategy.generate_signals(df)
    trend = ema(df["close"], trend_period)
    uptrend = df["close"] > trend
    is_counter = ((signal == 1) & (~uptrend)) | ((signal == -1) & uptrend)
    return is_counter


def report(name, strat, bt_config, holdout_df, holdout_a, holdout_b, risk_mult=None, stop_mult=None):
    engine = BacktestEngine(bt_config)
    m_full = compute_metrics(engine.run(holdout_df, strat, risk_multiplier=risk_mult, stop_multiplier=stop_mult))
    m_a = compute_metrics(engine.run(holdout_a, strat, risk_multiplier=risk_mult, stop_multiplier=stop_mult))
    m_b = compute_metrics(engine.run(holdout_b, strat, risk_multiplier=risk_mult, stop_multiplier=stop_mult))
    print(f"\n--- {name} ---")
    print(f"Net return:   {m_full.total_return_pct:+.2f}%")
    print(f"Max drawdown: {m_full.max_drawdown_pct:.2f}%")
    print(f"Sharpe:       {m_full.sharpe:+.2f}")
    print(f"Trades:       {m_full.num_trades}")
    print(f"  1st half: return {m_a.total_return_pct:+.2f}%  Sharpe {m_a.sharpe:+.2f}  trades {m_a.num_trades}")
    print(f"  2nd half: return {m_b.total_return_pct:+.2f}%  Sharpe {m_b.sharpe:+.2f}  trades {m_b.num_trades}")
    consistent = m_a.total_return_pct > 0 and m_b.total_return_pct > 0
    print(f"  Consistent both halves: {'YES' if consistent else 'NO'}")
    return m_full, consistent


def main() -> None:
    cfg = load_config()
    df = load_ohlcv_csv(REPO_ROOT / "data" / "EURUSD_H4_202401020000_202607020800.csv")
    _train_val_df, holdout_df = split_holdout(df, 0.7)
    mid = len(holdout_df) // 2
    holdout_a, holdout_b = holdout_df.iloc[:mid], holdout_df.iloc[mid:]

    best = json.loads((REPO_ROOT / "config" / "current_best_strategy.json").read_text())
    strat = strategy_from_dict(best["ensemble"]["config"])

    is_counter = counter_trend_mask(df, strat, trend_period=200)

    report("0. Baseline (no size/stop adjustment)", strat, build_config(cfg, "EURUSD"),
           holdout_df, holdout_a, holdout_b)

    risk_50 = is_counter.map({True: 0.5, False: 1.0})
    report("1. Counter-trend trades at 50% position size", strat, build_config(cfg, "EURUSD"),
           holdout_df, holdout_a, holdout_b, risk_mult=risk_50)

    risk_25 = is_counter.map({True: 0.25, False: 1.0})
    report("2. Counter-trend trades at 25% position size", strat, build_config(cfg, "EURUSD"),
           holdout_df, holdout_a, holdout_b, risk_mult=risk_25)

    stop_80 = is_counter.map({True: 0.8, False: 1.0})
    report("3. Counter-trend trades with 20% tighter stop loss", strat, build_config(cfg, "EURUSD"),
           holdout_df, holdout_a, holdout_b, stop_mult=stop_80)

    report("4. Counter-trend trades: 50% size AND 20% tighter stop", strat, build_config(cfg, "EURUSD"),
           holdout_df, holdout_a, holdout_b, risk_mult=risk_50, stop_mult=stop_80)


if __name__ == "__main__":
    main()
