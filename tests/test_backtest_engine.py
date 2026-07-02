import numpy as np
import pandas as pd

from modiri_bot.backtest.engine import BacktestConfig, BacktestEngine
from modiri_bot.strategies.base import Strategy


class ConstantSignalStrategy(Strategy):
    name = "constant"

    def __init__(self, value: int = 0):
        super().__init__(value=value)
        self.value = value

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index, dtype=int)


def flat_df(n=50, price=1.1000):
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, price)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 100.0},
        index=index,
    )


def make_config(**overrides) -> BacktestConfig:
    base = dict(
        initial_balance=1000.0,
        pip_size=0.0001,
        pip_value_per_lot=10.0,
        min_lot=0.01,
        lot_step=0.01,
        spread_pips=1.0,
        commission_per_lot=5.0,
        risk_per_trade_pct=1.0,
        stop_loss_pips=20.0,
        take_profit_pips=40.0,
        max_daily_loss_pct=None,
        max_drawdown_pct=None,
    )
    base.update(overrides)
    return BacktestConfig(**base)


def test_no_signal_means_no_trades_and_flat_equity():
    df = flat_df()
    engine = BacktestEngine(make_config())
    result = engine.run(df, ConstantSignalStrategy(0))
    assert len(result.trades) == 0
    assert (result.equity_curve == 1000.0).all()


def test_take_profit_closes_a_long_trade_with_correct_pnl():
    n = 10
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    # Flat, then a big bar whose high clears the take-profit level.
    close = np.full(n, 1.1000)
    high = np.full(n, 1.1000)
    high[-1] = 1.1100  # +100 pips
    low = np.full(n, 1.0995)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 100.0},
        index=index,
    )

    cfg = make_config(stop_loss_pips=20.0, take_profit_pips=40.0, spread_pips=0.0, commission_per_lot=0.0)
    engine = BacktestEngine(cfg)
    result = engine.run(df, ConstantSignalStrategy(1))

    # The strategy still wants exposure after the TP hit, so the engine
    # re-enters on the same bar; that second trade is force-closed flat at
    # end_of_data. Only the first trade is the one under test here.
    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.pnl > 0


def test_stop_loss_closes_a_long_trade_with_negative_pnl():
    n = 10
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, 1.1000)
    high = np.full(n, 1.1005)
    low = np.full(n, 1.1000)
    low[-1] = 1.0950  # -50 pips, below the 20 pip stop
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 100.0},
        index=index,
    )

    cfg = make_config(stop_loss_pips=20.0, take_profit_pips=100.0, spread_pips=0.0, commission_per_lot=0.0)
    engine = BacktestEngine(cfg)
    result = engine.run(df, ConstantSignalStrategy(1))

    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.pnl < 0


def test_spread_and_commission_are_a_real_cost_on_a_flat_market():
    df = flat_df(n=5)
    cfg = make_config(spread_pips=2.0, commission_per_lot=7.0, stop_loss_pips=20, take_profit_pips=1000)
    engine = BacktestEngine(cfg)
    result = engine.run(df, ConstantSignalStrategy(1))
    # Position never hits SL/TP and is force-closed at end_of_data at the same
    # close price it entered near -> the only PnL is the cost of the spread + commission.
    assert len(result.trades) == 1
    assert result.trades[0].pnl < 0


def test_drawdown_kill_switch_stops_new_trades():
    n = 200
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    # Every bar's low takes out the stop loss for a fresh long entered that bar.
    close = np.full(n, 1.1000)
    high = np.full(n, 1.1000)
    low = np.full(n, 1.0900)  # -100 pips, well past a 20 pip stop
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 100.0},
        index=index,
    )
    cfg = make_config(stop_loss_pips=20.0, take_profit_pips=1000.0, max_drawdown_pct=15.0,
                       risk_per_trade_pct=5.0)
    engine = BacktestEngine(cfg)
    result = engine.run(df, ConstantSignalStrategy(1))

    # Every bar would open a new losing trade if nothing stopped it; the
    # kill-switch should cut that off well before the end of the run.
    assert len(result.trades) < n / 2
    equity = result.equity_curve
    assert (equity.iloc[-30:] == equity.iloc[-1]).all()
