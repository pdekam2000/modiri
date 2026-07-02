#!/usr/bin/env python3
"""Systematically test a second round of risk-filter ideas, in the safer
order suggested: baseline, graded counter-trend sizing (by trend strength
rather than just direction), an ATR-distance-from-mean filter, a time
stop, an ADX/volatility regime filter applied to every trade, and finally
a combination of whichever individual ideas actually held up.

Every variant is judged on the untouched holdout split into two halves,
same as the rest of this project's methodology -- a combined number alone
proves nothing.

    python scripts/test_friend_filters.py
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
    m_full = compute_metrics(engine.run(holdout_df, strat, **run_kwargs))
    m_a = compute_metrics(engine.run(holdout_a, strat, **run_kwargs))
    m_b = compute_metrics(engine.run(holdout_b, strat, **run_kwargs))
    consistent = m_a.total_return_pct > 0 and m_b.total_return_pct > 0
    print(f"\n--- {name} ---")
    print(f"Return {m_full.total_return_pct:+.2f}%  Sharpe {m_full.sharpe:+.2f}  Calmar {m_full.calmar:+.2f}  "
          f"MaxDD {m_full.max_drawdown_pct:.2f}%  trades {m_full.num_trades}")
    print(f"  1st half: {m_a.total_return_pct:+.2f}% (Sharpe {m_a.sharpe:+.2f}, {m_a.num_trades} trades)   "
          f"2nd half: {m_b.total_return_pct:+.2f}% (Sharpe {m_b.sharpe:+.2f}, {m_b.num_trades} trades)   "
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

    signal = strat.generate_signals(df)
    trend_200 = ema(df["close"], 200)
    uptrend = df["close"] > trend_200
    is_counter = ((signal == 1) & (~uptrend)) | ((signal == -1) & uptrend)

    adx_line, _plus_di, _minus_di = adx(df["high"], df["low"], df["close"], 14)
    atr_line = atr(df["high"], df["low"], df["close"], 14)
    dist_from_mean_atr = (df["close"] - ema(df["close"], 50)).abs() / atr_line.replace(0, np.nan)

    results = {}

    # 0. Baseline
    m, c = report("0. Baseline", strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b)
    results["0_baseline"] = (m, c)

    # 1. Graded counter-trend sizing by ADX severity: mild trend (ADX<20)
    # -> 0.75x, strong established trend (ADX>=20) -> 0.5x, trend-aligned
    # trades untouched.
    severity = pd.Series(1.0, index=df.index)
    severity[is_counter & (adx_line < 20)] = 0.75
    severity[is_counter & (adx_line >= 20)] = 0.5
    m, c = report("1. Graded counter-trend sizing (ADX severity: 0.75x/0.5x)",
                   strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=severity)
    results["1_graded_sizing"] = (m, c)

    # 2. ATR-distance-from-mean filter: reduce size when price is more than
    # 3 ATR away from its 50-EMA (likely trend continuation, not reversion).
    distance_mult = pd.Series(1.0, index=df.index)
    distance_mult[dist_from_mean_atr > 3.0] = 0.25
    m, c = report("2. ATR-distance-from-mean filter (>3 ATR -> 0.25x)",
                   strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=distance_mult)
    results["2_distance_filter"] = (m, c)

    # 3. Time stop: force-exit after N bars if nothing else has closed the trade.
    for n_bars in (10, 15):
        cfg_ts = build_config(cfg, "EURUSD")
        cfg_ts.max_hold_bars = n_bars
        m, c = report(f"3. Time stop ({n_bars} bars)", strat, cfg_ts, holdout_df, holdout_a, holdout_b)
        results[f"3_time_stop_{n_bars}"] = (m, c)

    # 4. ADX/volatility regime filter applied to EVERY trade (not just
    # counter-trend ones): a strongly trending or unusually volatile market
    # is riskier for a mean-reversion strategy either direction.
    regime_mult = pd.Series(1.0, index=df.index)
    regime_mult[adx_line > 35] = 0.5
    m, c = report("4. ADX regime filter on all trades (ADX>35 -> 0.5x)",
                   strat, build_config(cfg, "EURUSD"), holdout_df, holdout_a, holdout_b, risk_multiplier=regime_mult)
    results["4_adx_regime"] = (m, c)

    # 5. Combine whichever individual ideas were both consistent AND an
    # improvement (Sharpe or Calmar) over baseline.
    baseline_m, _ = results["0_baseline"]
    winners = []
    for key in ["1_graded_sizing", "2_distance_filter", "3_time_stop_10", "3_time_stop_15", "4_adx_regime"]:
        m, c = results[key]
        if c and (m.sharpe >= baseline_m.sharpe or m.calmar >= baseline_m.calmar):
            winners.append(key)
    print(f"\nIndividual ideas that were both consistent and >= baseline on Sharpe or Calmar: {winners or 'none'}")

    if len(winners) >= 2:
        combo_mult = pd.Series(1.0, index=df.index)
        cfg_combo = build_config(cfg, "EURUSD")
        if "1_graded_sizing" in winners:
            combo_mult = combo_mult * severity
        if "2_distance_filter" in winners:
            combo_mult = combo_mult * distance_mult
        if "4_adx_regime" in winners:
            combo_mult = combo_mult * regime_mult
        if any("3_time_stop" in w for w in winners):
            n_bars = 10 if "3_time_stop_10" in winners else 15
            cfg_combo.max_hold_bars = n_bars
        report("5. Combination of winning ideas", strat, cfg_combo, holdout_df, holdout_a, holdout_b,
               risk_multiplier=combo_mult)
    else:
        print("Fewer than 2 individual winners -- skipping the combination step.")


if __name__ == "__main__":
    main()
