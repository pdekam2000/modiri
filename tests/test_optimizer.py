import pandas as pd
import pytest

from modiri_bot.backtest.engine import BacktestConfig
from modiri_bot.backtest.optimizer import optimize_strategies, split_holdout, walk_forward_splits
from modiri_bot.data.synthetic import generate_synthetic_ohlcv
from modiri_bot.strategies.registry import STRATEGY_CLASSES


def test_split_holdout_respects_fraction():
    df = generate_synthetic_ohlcv(n_bars=1000)
    train, holdout = split_holdout(df, 0.7)
    assert len(train) == 700
    assert len(holdout) == 300
    assert train.index[-1] < holdout.index[0]


def test_walk_forward_splits_are_expanding_and_non_overlapping():
    df = generate_synthetic_ohlcv(n_bars=1000)
    folds = walk_forward_splits(df, n_folds=3)
    assert len(folds) == 3
    for train_df, test_df in folds:
        assert train_df.index[-1] < test_df.index[0]
    # Each fold's train window should be at least as long as the previous one.
    lengths = [len(train_df) for train_df, _ in folds]
    assert lengths == sorted(lengths)


def test_walk_forward_splits_raises_when_too_few_bars():
    df = generate_synthetic_ohlcv(n_bars=10)
    with pytest.raises(ValueError):
        walk_forward_splits(df, n_folds=5)


def test_optimize_strategies_end_to_end_smoke():
    df = generate_synthetic_ohlcv(n_bars=1500, seed=7)
    report = optimize_strategies(df, BacktestConfig(), n_folds=2, objective="sharpe")

    assert len(report.per_strategy_scores) == len(STRATEGY_CLASSES)
    assert report.best_single is not None
    assert report.best_ensemble is not None
    assert report.holdout_single_metrics is not None
    assert report.holdout_ensemble_metrics is not None
    # Scores should be sorted descending.
    scores = [s.oos_score for s in report.per_strategy_scores]
    assert scores == sorted(scores, reverse=True)
