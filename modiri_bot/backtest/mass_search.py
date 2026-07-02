"""Large-scale strategy-combination search.

Generates a wide universe of parameterized strategy variants, then scores
individual variants AND millions of random ensemble-weight combinations
using a fast vectorized approximation (next-bar return * position, matrix
multiplication instead of a Python loop per combination) rather than the
full bar-by-bar engine. That approximation is what makes testing millions
of combinations computationally feasible in a few minutes instead of days.

The approximation is only used to *rank* candidates. Only the handful of
top survivors get validated with the real BacktestEngine (real spread/
commission/SL-TP/risk limits) against data the search never touched, and
only those numbers should be trusted -- the approximate score is a cheap
filter, not a performance estimate.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from modiri_bot.strategies.base import Strategy
from modiri_bot.strategies.ensemble import EnsembleStrategy
from modiri_bot.strategies.registry import STRATEGY_CLASSES

# Much finer-grained than modiri_bot/strategies/registry.py's PARAM_GRIDS --
# this is the whole point of the mass search. ~1000-1200 individual variants
# across 13 strategy families (trend, momentum/oscillator mean-reversion,
# volatility breakout, retracement-in-trend, and multi-timeframe filtered).
FINE_PARAM_GRIDS = {
    "ma_crossover": {
        "fast_period": [3, 5, 8, 10, 13, 16, 20, 25],
        "slow_period": [20, 30, 40, 50, 65, 80, 100, 130, 160, 200],
        "use_ema": [True, False],
    },
    "rsi_reversion": {
        "period": [5, 7, 9, 11, 14, 18, 21, 25, 30],
        "oversold": [10, 15, 20, 25, 30, 35],
        "overbought": [65, 70, 75, 80, 85, 90],
    },
    "macd_trend": {
        "fast": [5, 8, 10, 12, 15, 20],
        "slow": [20, 26, 30, 35, 40],
        "signal": [5, 9, 12, 15],
    },
    "bollinger_breakout": {
        "period": [10, 14, 18, 20, 25, 30, 40],
        "num_std": [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "donchian_breakout": {
        "period": [5, 8, 10, 15, 20, 25, 30, 40, 55, 70, 100],
    },
    "ichimoku": {
        "tenkan_period": [5, 7, 9, 12, 15],
        "kijun_period": [18, 22, 26, 30, 35],
        "senkou_b_period": [40, 52, 60, 70],
    },
    "stochastic_reversion": {
        "k_period": [7, 9, 14, 21],
        "oversold": [10, 15, 20, 25],
        "overbought": [75, 80, 85, 90],
    },
    "adx_trend": {
        "period": [7, 10, 14, 21],
        "adx_threshold": [15, 20, 25, 30, 35],
    },
    "atr_channel_breakout": {
        "ema_period": [10, 14, 20, 30, 40],
        "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "cci_reversion": {
        "period": [10, 14, 20, 30],
        "threshold": [80, 100, 120, 150, 200],
    },
    "williams_r_reversion": {
        "period": [7, 9, 14, 21],
        "oversold": [-90, -85, -80, -75, -70],
        "overbought": [-30, -25, -20, -15, -10],
    },
    "parabolic_sar_trend": {
        "af_step": [0.01, 0.02, 0.03, 0.04],
        "af_max": [0.1, 0.15, 0.2, 0.3, 0.4],
    },
    "trend_pullback": {
        "trend_period": [20, 30, 50, 80, 100],
        "rsi_period": [7, 9, 14, 21],
    },
    "mtf_trend_filter": {
        "fast_period": [5, 8, 10, 15],
        "slow_period": [20, 30, 40, 50],
        "htf_trend_period": [10, 15, 20, 30],
    },
}


def build_variant_universe() -> list[Strategy]:
    variants: list[Strategy] = []
    for name, grid in FINE_PARAM_GRIDS.items():
        cls = STRATEGY_CLASSES[name]
        keys = list(grid.keys())
        for values in itertools.product(*grid.values()):
            params = dict(zip(keys, values))
            if name == "ma_crossover" and params["fast_period"] >= params["slow_period"]:
                continue
            if name == "macd_trend" and params["fast"] >= params["slow"]:
                continue
            if name == "ichimoku" and params["tenkan_period"] >= params["kijun_period"]:
                continue
            if name == "mtf_trend_filter" and params["fast_period"] >= params["slow_period"]:
                continue
            variants.append(cls(**params))
    return variants


def build_signals_matrix(variants: list[Strategy], df: pd.DataFrame) -> np.ndarray:
    """Shape (n_bars, n_variants); column j is variant j's -1/0/1 signal."""
    cols = [v.generate_signals(df).to_numpy() for v in variants]
    return np.column_stack(cols).astype(np.float32)


def next_bar_returns(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].to_numpy()
    returns = np.zeros(len(close), dtype=np.float64)
    returns[:-1] = close[1:] / close[:-1] - 1.0
    return returns


def approx_score_single(signals: np.ndarray, returns: np.ndarray, cost_pct: float) -> np.ndarray:
    """Vectorized approximate Sharpe-like score for every variant (column) at once."""
    position = signals[:-1]
    ret = returns[:-1]
    turnover = np.abs(np.diff(signals, axis=0, prepend=signals[:1]))[:-1]
    pnl = position * ret[:, None] - cost_pct * turnover
    mean = pnl.mean(axis=0)
    std = pnl.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(std > 0, mean / std, 0.0)
    return score


def approx_score_ensembles(
    signals: np.ndarray,     # (n_bars, n_top)
    returns: np.ndarray,     # (n_bars,)
    weights: np.ndarray,     # (n_combos, n_top)
    thresholds: np.ndarray,  # (n_combos,)
    cost_pct: float,
) -> np.ndarray:
    """Score many ensemble weight combinations in one shot via matrix multiply."""
    combined = signals @ weights.T                       # (n_bars, n_combos)
    norm = np.abs(weights).sum(axis=1)
    norm[norm == 0] = 1.0
    combined = combined / norm[None, :]
    position = np.where(combined > thresholds[None, :], 1.0,
                         np.where(combined < -thresholds[None, :], -1.0, 0.0))
    pos = position[:-1]
    ret = returns[:-1]
    turnover = np.abs(np.diff(position, axis=0, prepend=position[:1]))[:-1]
    pnl = pos * ret[:, None] - cost_pct * turnover
    mean = pnl.mean(axis=0)
    std = pnl.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(std > 0, mean / std, 0.0)
    return score


def _sparse_weight_batch(rng: np.random.Generator, batch_n: int, n_top: int, k: int) -> np.ndarray:
    """batch_n rows, each with exactly k random nonzero weights among n_top columns."""
    idx = np.argsort(rng.random((batch_n, n_top)), axis=1)[:, :k]
    vals = rng.uniform(0.3, 2.0, size=(batch_n, k)).astype(np.float32)
    weights = np.zeros((batch_n, n_top), dtype=np.float32)
    np.put_along_axis(weights, idx, vals, axis=1)
    return weights


@dataclass
class EnsembleCandidate:
    weights_row: np.ndarray
    threshold: float
    score: float


def random_search_ensembles(
    fold_signals: list[np.ndarray],
    fold_returns: list[np.ndarray],
    n_top: int,
    n_samples: int,
    cost_pct: float,
    batch_size: int = 50_000,
    member_range: tuple[int, int] = (2, 5),
    keep_top: int = 200,
    seed: int = 0,
) -> list[EnsembleCandidate]:
    rng = np.random.default_rng(seed)
    kept: list[EnsembleCandidate] = []

    done = 0
    batch_idx = 0
    while done < n_samples:
        batch_n = min(batch_size, n_samples - done)
        k = member_range[0] + (batch_idx % (member_range[1] - member_range[0] + 1))
        weights = _sparse_weight_batch(rng, batch_n, n_top, k)
        thresholds = rng.uniform(0.0, 0.4, size=batch_n).astype(np.float32)

        scores = np.zeros(batch_n, dtype=np.float64)
        for signals, returns in zip(fold_signals, fold_returns):
            scores += approx_score_ensembles(signals, returns, weights, thresholds, cost_pct)
        scores /= len(fold_signals)

        top_n = min(keep_top, batch_n)
        top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
        for i in top_idx:
            kept.append(EnsembleCandidate(weights[i].copy(), float(thresholds[i]), float(scores[i])))

        kept.sort(key=lambda c: c.score, reverse=True)
        kept = kept[:keep_top]

        done += batch_n
        batch_idx += 1

    return kept


def candidate_to_ensemble(candidate: EnsembleCandidate, variants: list[Strategy]) -> EnsembleStrategy:
    idx = np.nonzero(candidate.weights_row)[0]
    members = [variants[i] for i in idx]
    weights = [float(candidate.weights_row[i]) for i in idx]
    return EnsembleStrategy(members, weights=weights, threshold=candidate.threshold)
