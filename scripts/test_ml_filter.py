#!/usr/bin/env python3
"""Tests a simple ML meta-label filter: a heavily-regularized logistic
regression trained on in-sample trade features (ADX, ATR percentile, hour,
day of week, trend alignment) to predict whether a signal will win,
applied purely out-of-sample on the holdout.

IMPORTANT CAVEAT stated up front: there are only ~60-70 in-sample trades
to train on. That is a very small sample for any ML model -- results here
should be read as "does this look promising enough to pursue with more
data", not as a validated result on the same footing as the rest of this
project's findings.

    python scripts/test_ml_filter.py
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
from sklearn.linear_model import LogisticRegression  # noqa: E402

from modiri_bot.backtest.engine import BacktestEngine  # noqa: E402
from modiri_bot.backtest.metrics import compute_metrics  # noqa: E402
from modiri_bot.backtest.optimizer import split_holdout  # noqa: E402
from modiri_bot.data.csv_loader import load_ohlcv_csv  # noqa: E402
from modiri_bot.strategies.indicators import adx, atr, ema  # noqa: E402
from modiri_bot.strategies.registry import strategy_from_dict  # noqa: E402
from modiri_bot.utils.config import load_config  # noqa: E402
from run_backtest import build_config  # noqa: E402


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    adx_line, _plus, _minus = adx(df["high"], df["low"], df["close"], 14)
    atr_line = atr(df["high"], df["low"], df["close"], 14)
    atr_pct_rank = atr_line.rolling(100, min_periods=50).apply(lambda x: (x[-1] > x).mean() * 100, raw=True)
    trend_200 = ema(df["close"], 200)
    return pd.DataFrame({
        "adx": adx_line,
        "atr_pct_rank": atr_pct_rank,
        "hour": df.index.hour,
        "dow": df.index.dayofweek,
        "above_trend": (df["close"] > trend_200).astype(float),
    }, index=df.index)


def trades_with_features(trades_df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = trades_df.merge(features, left_on="entry_time", right_index=True, how="left")
    merged["is_win"] = (merged["pnl"] > 0).astype(int)
    return merged.dropna(subset=["adx", "atr_pct_rank"])


def main() -> None:
    print("[!] CAVEAT: only ~60-70 in-sample trades available to train on -- treat this "
          "as exploratory, not validated on the same footing as the rest of the project.\n")

    cfg = load_config()
    df = load_ohlcv_csv(REPO_ROOT / "data" / "EURUSD_H4_202401020000_202607020800.csv")
    train_val_df, holdout_df = split_holdout(df, 0.7)

    best = json.loads((REPO_ROOT / "config" / "current_best_strategy.json").read_text())
    strat = strategy_from_dict(best["ensemble"]["config"])
    bt_config = build_config(cfg, "EURUSD")
    engine = BacktestEngine(bt_config)

    train_result = engine.run(train_val_df, strat)
    train_features = build_features(train_val_df)
    train_trades = trades_with_features(train_result.trades_df(), train_features)
    print(f"In-sample trades available to train on: {len(train_trades)} "
          f"({train_trades['is_win'].sum()} wins, {(1 - train_trades['is_win']).sum()} losses)")

    feature_cols = ["adx", "atr_pct_rank", "hour", "dow", "above_trend"]
    X_train = train_trades[feature_cols].to_numpy()
    y_train = train_trades["is_win"].to_numpy()

    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    print(f"In-sample training accuracy: {train_acc:.1%} (baseline win rate: {y_train.mean():.1%})")
    print(f"Feature coefficients: {dict(zip(feature_cols, model.coef_[0].round(3)))}")

    holdout_result = engine.run(holdout_df, strat)
    holdout_features = build_features(holdout_df)
    holdout_trades = trades_with_features(holdout_result.trades_df(), holdout_features)
    print(f"\nHoldout trades available to test on: {len(holdout_trades)}")

    X_holdout = holdout_trades[feature_cols].to_numpy()
    predicted_win_prob = model.predict_proba(X_holdout)[:, 1]
    holdout_trades = holdout_trades.copy()
    holdout_trades["predicted_win_prob"] = predicted_win_prob

    m_baseline = compute_metrics(holdout_result)
    print(f"\n--- Baseline (no ML filter) ---")
    print(f"Return {m_baseline.total_return_pct:+.2f}%  Sharpe {m_baseline.sharpe:+.2f}  "
          f"trades {m_baseline.num_trades}")

    for threshold in [0.4, 0.5, 0.6]:
        kept = holdout_trades[holdout_trades["predicted_win_prob"] >= threshold]
        if len(kept) == 0:
            print(f"\n--- ML filter (keep predicted win-prob >= {threshold}) ---")
            print("No trades pass this threshold.")
            continue
        actual_win_rate = kept["is_win"].mean() * 100
        total_pnl = kept["pnl"].sum()
        print(f"\n--- ML filter (keep predicted win-prob >= {threshold}) ---")
        print(f"Trades kept: {len(kept)}/{len(holdout_trades)}   "
              f"Actual win rate of kept trades: {actual_win_rate:.1f}%   Total P&L: {total_pnl:+.2f}")


if __name__ == "__main__":
    main()
