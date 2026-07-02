#!/usr/bin/env python3
"""Two more risk-filter categories requested but not yet tested on the
champion ensemble: a volatility-percentile regime filter (distinct from
the earlier ADX-based regime filter, which failed) and a genuine
multi-timeframe confirmation overlay (require the daily trend to agree
with the H4 signal, not just include MTFTrendFilterStrategy as one
candidate among thousands in the mass search).

    python scripts/test_volatility_and_mtf_filters.py
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
from modiri_bot.backtest.optimizer import split_holdout  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.strategies.indicators import atr, ema  # noqa: E402
from modiri_bot.strategies.registry import strategy_from_dict  # noqa: E402
from modiri_bot.utils.config import load_config  # noqa: E402
from run_backtest import build_config  # noqa: E402


def report(name, strat, bt_config, holdout_df, holdout_a, holdout_b, **run_kwargs):
    engine = BacktestEngine(bt_config)
    r_full = engine.run(holdout_df, strat, **run_kwargs)
    m_full = compute_metrics(r_full)
    m_a = compute_metrics(engine.run(holdout_a, strat, **run_kwargs))
    m_b = compute_metrics(engine.run(holdout_b, strat, **run_kwargs))
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

    report("0. Baseline (with validated 15-bar time stop, config.yaml default)",
           strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b)

    # 1. Volatility-percentile regime filter: reduce size when current ATR
    # is in the extreme upper tail of its own recent history (unstable/
    # expanding volatility -- different information than ADX trend strength).
    atr_line = atr(df["high"], df["low"], df["close"], 14)
    atr_percentile = atr_line.rolling(252, min_periods=100).apply(
        lambda x: (x[-1] > x).mean() * 100, raw=True
    )
    for pct_threshold in [85, 90, 95]:
        vol_mult = pd.Series(1.0, index=df.index)
        vol_mult[atr_percentile > pct_threshold] = 0.5
        report(f"1. Volatility-percentile filter (ATR>{pct_threshold}th pct -> 0.5x)",
               strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=vol_mult)

    # 2. Genuine multi-timeframe confirmation: only take the H4 ensemble's
    # signal when the daily trend (EMA20 on D1) agrees with its direction;
    # halve size (don't fully block, per the earlier lesson about gutting
    # trade flow) when it disagrees.
    daily_close = df["close"].resample("1D").last().dropna()
    daily_trend = ema(daily_close, 20)
    daily_uptrend = (daily_close > daily_trend).astype(float)
    daily_uptrend[daily_trend.isna()] = np.nan
    daily_uptrend = daily_uptrend.shift(1)  # no look-ahead into a still-forming daily bar
    daily_aligned = daily_uptrend.reindex(df.index, method="ffill")

    signal = strat.generate_signals(df)
    mtf_agrees = ((signal == 1) & (daily_aligned == 1.0)) | ((signal == -1) & (daily_aligned == 0.0))
    mtf_mult_half = pd.Series(1.0, index=df.index)
    mtf_mult_half[~mtf_agrees & (signal != 0)] = 0.5
    report("2. MTF confirmation (D1 EMA20 trend) -- disagreement -> 0.5x",
           strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=mtf_mult_half)

    mtf_mult_block = pd.Series(1.0, index=df.index)
    mtf_mult_block[~mtf_agrees & (signal != 0)] = 0.0
    report("2b. MTF confirmation (D1 EMA20 trend) -- disagreement -> blocked entirely",
           strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=mtf_mult_block)


if __name__ == "__main__":
    main()
