#!/usr/bin/env python3
"""Re-tests the previously-rejected filters (ADX regime, MTF confirmation,
ATR-distance-from-mean) against the CURRENT production baseline, which now
includes the trailing stop -- a methodology gap, since those filters were
originally rejected against an older, weaker baseline (no trailing stop).
Also tests a trailing-stop + partial-TP combination and Kelly-style
dynamic position sizing, neither tried before.

    python scripts/test_rejected_filters_on_new_baseline.py
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
from modiri_bot.strategies.indicators import adx, atr, ema  # noqa: E402
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

    report("0. Current production baseline (time stop + vol filter + trailing stop)",
           strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b)

    # 1. ADX regime filter, re-tested on the new baseline.
    adx_line, _p, _m = adx(df["high"], df["low"], df["close"], 14)
    regime_mult = pd.Series(1.0, index=df.index)
    regime_mult[adx_line > 35] = 0.5
    report("1. ADX regime filter (ADX>35 -> 0.5x) on new baseline",
           strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=regime_mult)

    # 2. MTF (D1 EMA20) confirmation, re-tested on the new baseline.
    daily_close = df["close"].resample("1D").last().dropna()
    daily_trend = ema(daily_close, 20)
    daily_uptrend = (daily_close > daily_trend).astype(float)
    daily_uptrend[daily_trend.isna()] = np.nan
    daily_uptrend = daily_uptrend.shift(1)
    daily_aligned = daily_uptrend.reindex(df.index, method="ffill")
    signal = strat.generate_signals(df)
    mtf_agrees = ((signal == 1) & (daily_aligned == 1.0)) | ((signal == -1) & (daily_aligned == 0.0))
    mtf_mult = pd.Series(1.0, index=df.index)
    mtf_mult[~mtf_agrees & (signal != 0)] = 0.5
    report("2. MTF confirmation (D1 EMA20) on new baseline",
           strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=mtf_mult)

    # 3. ATR-distance-from-mean filter, re-tested on the new baseline.
    atr_line = atr(df["high"], df["low"], df["close"], 14)
    dist_from_mean_atr = (df["close"] - ema(df["close"], 50)).abs() / atr_line.replace(0, np.nan)
    distance_mult = pd.Series(1.0, index=df.index)
    distance_mult[dist_from_mean_atr > 3.0] = 0.25
    report("3. ATR-distance-from-mean filter (>3 ATR -> 0.25x) on new baseline",
           strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=distance_mult)

    # 4. Trailing stop + partial (two-stage) take-profit combined.
    cfg_combo = build_config(cfg, "EURUSD")
    cfg_combo.use_partial_tp = True
    cfg_combo.tp1_r_multiple = 1.0
    cfg_combo.tp1_close_fraction = 0.5
    cfg_combo.move_sl_to_breakeven_after_tp1 = True
    report("4. Trailing stop + two-stage take-profit combined",
           strat, cfg_combo, holdout_df, holdout_a, holdout_b)

    # 5. Kelly-style dynamic sizing: scale risk_per_trade_pct with a
    # trailing estimate of the strategy's own edge (win rate / avg win-loss
    # ratio) instead of a fixed 1%, using a half-Kelly fraction for safety.
    # Estimated from the in-sample period only (no look-ahead into holdout).
    train_val_df, _ = split_holdout(df, 0.7)
    engine_est = BacktestEngine(build_config(cfg, "EURUSD"))
    train_trades = engine_est.run(train_val_df, strat).trades_df()
    win_rate_est = (train_trades["pnl"] > 0).mean()
    avg_win = train_trades[train_trades["pnl"] > 0]["pnl"].mean()
    avg_loss = -train_trades[train_trades["pnl"] <= 0]["pnl"].mean()
    b = avg_win / avg_loss if avg_loss > 0 else 1.0
    kelly_fraction = win_rate_est - (1 - win_rate_est) / b
    half_kelly_risk_pct = max(min(kelly_fraction / 2 * 100, 3.0), 0.25)  # clamp to a sane range
    print(f"\nEstimated in-sample edge: win_rate={win_rate_est:.2f}, b={b:.2f}, "
          f"full Kelly={kelly_fraction * 100:.2f}%, half-Kelly risk/trade={half_kelly_risk_pct:.2f}%")
    cfg_kelly = build_config(cfg, "EURUSD")
    cfg_kelly.risk_per_trade_pct = half_kelly_risk_pct
    report(f"5. Half-Kelly fixed risk/trade ({half_kelly_risk_pct:.2f}% vs. default 1%)",
           strat, cfg_kelly, holdout_df, holdout_a, holdout_b)


if __name__ == "__main__":
    main()
