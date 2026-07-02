import numpy as np
import pandas as pd
import pytest

from modiri_bot.backtest.engine import BacktestResult, Trade
from modiri_bot.backtest.metrics import compute_metrics


def make_result(equity_values, freq="h"):
    index = pd.date_range("2024-01-01", periods=len(equity_values), freq=freq)
    equity = pd.Series(equity_values, index=index, dtype=float)
    return BacktestResult(equity_curve=equity, trades=[])


def test_flat_equity_has_zero_return_and_zero_sharpe():
    result = make_result([1000.0] * 100)
    m = compute_metrics(result)
    assert m.total_return_pct == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown_pct == 0.0


def test_steadily_rising_equity_has_positive_return_and_sharpe():
    values = np.linspace(1000, 1200, 200)
    result = make_result(values)
    m = compute_metrics(result)
    assert m.total_return_pct == pytest.approx(20.0)
    assert m.sharpe > 0
    assert m.max_drawdown_pct == pytest.approx(0.0, abs=0.5)


def test_drawdown_is_measured_from_peak():
    values = [1000, 1100, 1050, 900, 950, 1000]
    result = make_result(values)
    m = compute_metrics(result)
    # Peak 1100 -> trough 900 => 18.18% drawdown
    assert m.max_drawdown_pct == pytest.approx(18.18, abs=0.1)


def test_profit_factor_and_win_rate_from_trades():
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    equity = pd.Series([1000.0, 1010.0], index=index)
    trades = [
        Trade(index[0], index[0], "buy", 1.1, 1.11, 0.1, 20.0, "take_profit"),
        Trade(index[1], index[1], "sell", 1.1, 1.105, 0.1, -10.0, "stop_loss"),
    ]
    result = BacktestResult(equity_curve=equity, trades=trades)
    m = compute_metrics(result)
    assert m.num_trades == 2
    assert m.win_rate_pct == 50.0
    assert m.profit_factor == pytest.approx(2.0)
