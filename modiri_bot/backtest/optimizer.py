"""Walk-forward strategy search: grid-search each strategy's parameters,
score them out-of-sample across several time folds, then search ensemble
weights the same way. A final untouched holdout segment is used only once,
at the very end, to report a realistic out-of-sample result.

This deliberately optimizes a risk-adjusted objective (Sharpe/Sortino/
Calmar), not raw return, because chasing raw return on a fixed dataset is
exactly how you find a strategy that looks great in the backtest and blows
up live.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from modiri_bot.strategies.base import Strategy
from modiri_bot.strategies.ensemble import EnsembleStrategy
from modiri_bot.strategies.registry import PARAM_GRIDS, STRATEGY_CLASSES

from .engine import BacktestConfig, BacktestEngine
from .metrics import Metrics, compute_metrics

OBJECTIVES = {"sharpe", "sortino", "calmar"}


def split_holdout(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * train_fraction)
    return df.iloc[:cut], df.iloc[cut:]


def walk_forward_splits(df: pd.DataFrame, n_folds: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window walk-forward: fold k trains on everything up to the
    end of chunk k and tests on chunk k+1."""
    n_chunks = n_folds + 1
    chunk_size = len(df) // n_chunks
    if chunk_size < 20:
        raise ValueError("Not enough bars for the requested number of walk-forward folds")

    bounds = [i * chunk_size for i in range(n_chunks)] + [len(df)]
    folds = []
    for k in range(n_folds):
        train_end = bounds[k + 1]
        test_end = bounds[k + 2]
        folds.append((df.iloc[:train_end], df.iloc[train_end:test_end]))
    return folds


def _score(metrics: Metrics, objective: str) -> float:
    return getattr(metrics, objective)


def _avg_oos_score(
    strategy: Strategy,
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    config: BacktestConfig,
    objective: str,
) -> float:
    engine = BacktestEngine(config)
    scores = []
    for _train_df, test_df in folds:
        result = engine.run(test_df, strategy)
        if len(result.trades) == 0:
            scores.append(0.0)
            continue
        scores.append(_score(compute_metrics(result), objective))
    return sum(scores) / len(scores) if scores else 0.0


@dataclass
class StrategyScore:
    strategy: Strategy
    oos_score: float


@dataclass
class OptimizationReport:
    per_strategy_scores: list[StrategyScore]
    best_single: StrategyScore
    best_ensemble: StrategyScore
    holdout_single_metrics: Metrics
    holdout_ensemble_metrics: Metrics
    objective: str


def optimize_strategies(
    df: pd.DataFrame,
    config: BacktestConfig,
    train_fraction: float = 0.7,
    n_folds: int = 4,
    objective: str = "sharpe",
    ensemble_top_k: int = 3,
) -> OptimizationReport:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")

    train_val_df, holdout_df = split_holdout(df, train_fraction)
    folds = walk_forward_splits(train_val_df, n_folds)

    per_strategy_scores: list[StrategyScore] = []
    for name, cls in STRATEGY_CLASSES.items():
        grid = PARAM_GRIDS[name]
        keys = list(grid.keys())
        best: StrategyScore | None = None
        for values in itertools.product(*grid.values()):
            params = dict(zip(keys, values))
            candidate = cls(**params)
            score = _avg_oos_score(candidate, folds, config, objective)
            if best is None or score > best.oos_score:
                best = StrategyScore(candidate, score)
        assert best is not None
        per_strategy_scores.append(best)

    per_strategy_scores.sort(key=lambda s: s.oos_score, reverse=True)
    best_single = per_strategy_scores[0]

    top = [s for s in per_strategy_scores[:ensemble_top_k]]
    ensemble_strategies = [s.strategy for s in top]

    weight_levels = [0.0, 0.5, 1.0, 1.5]
    threshold_levels = [0.0, 0.15, 0.3]
    best_ensemble: StrategyScore | None = None
    for weights in itertools.product(weight_levels, repeat=len(ensemble_strategies)):
        if sum(weights) == 0:
            continue
        for threshold in threshold_levels:
            candidate = EnsembleStrategy(ensemble_strategies, weights=list(weights), threshold=threshold)
            score = _avg_oos_score(candidate, folds, config, objective)
            if best_ensemble is None or score > best_ensemble.oos_score:
                best_ensemble = StrategyScore(candidate, score)
    assert best_ensemble is not None

    engine = BacktestEngine(config)
    holdout_single_metrics = compute_metrics(engine.run(holdout_df, best_single.strategy))
    holdout_ensemble_metrics = compute_metrics(engine.run(holdout_df, best_ensemble.strategy))

    return OptimizationReport(
        per_strategy_scores=per_strategy_scores,
        best_single=best_single,
        best_ensemble=best_ensemble,
        holdout_single_metrics=holdout_single_metrics,
        holdout_ensemble_metrics=holdout_ensemble_metrics,
        objective=objective,
    )
