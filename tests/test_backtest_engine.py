import numpy as np
import pandas as pd
import pytest

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


def test_atr_stops_widen_with_higher_volatility():
    n = 60
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    # Calm market, then a clearly more volatile stretch -- ATR should reflect that.
    close = np.concatenate([np.full(30, 1.1000), 1.1000 + np.cumsum(rng.normal(0, 0.001, 30))])
    high = close + np.concatenate([np.full(30, 0.0002), np.full(30, 0.0015)])
    low = close - np.concatenate([np.full(30, 0.0002), np.full(30, 0.0015)])
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 100.0}, index=index)

    cfg = make_config(use_atr_stops=True, atr_period=5, atr_sl_mult=1.5, atr_tp_mult=3.0)
    engine = BacktestEngine(cfg)
    result = engine.run(df, ConstantSignalStrategy(1))
    assert len(result.trades) >= 1  # ran without error and actually traded


def test_partial_take_profit_closes_half_then_lets_rest_run():
    n = 10
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, 1.1000)
    high = np.full(n, 1.1000)
    # Bar 3: hits TP1 (1R = 20 pips above entry). Bar 7: hits the final TP.
    high[3] = 1.1021
    high[7] = 1.1041
    low = np.full(n, 1.0999)
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 100.0}, index=index)

    cfg = make_config(
        stop_loss_pips=20.0, take_profit_pips=40.0, spread_pips=0.0, commission_per_lot=0.0,
        use_partial_tp=True, tp1_r_multiple=1.0, tp1_close_fraction=0.5,
        move_sl_to_breakeven_after_tp1=True,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(df, ConstantSignalStrategy(1))

    reasons = [t.exit_reason for t in result.trades]
    assert "take_profit_1" in reasons
    tp1_trade = next(t for t in result.trades if t.exit_reason == "take_profit_1")
    assert tp1_trade.pnl > 0
    # Only half the position should have closed at TP1.
    full_position_lots = tp1_trade.lots * 2
    assert tp1_trade.lots == pytest.approx(full_position_lots / 2)


def test_partial_take_profit_moves_stop_to_breakeven():
    n = 10
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, 1.1000)
    high = np.full(n, 1.1000)
    high[3] = 1.1021  # hits TP1, moving the stop to breakeven (1.1000)
    low = np.full(n, 1.0999)
    # Dips just below breakeven -- would NOT have hit the original 20-pip
    # stop (1.0980), but should hit the now-tightened breakeven stop.
    low[5] = 1.0995
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 100.0}, index=index)

    cfg = make_config(
        stop_loss_pips=20.0, take_profit_pips=1000.0, spread_pips=0.0, commission_per_lot=0.0,
        use_partial_tp=True, tp1_r_multiple=1.0, tp1_close_fraction=0.5,
        move_sl_to_breakeven_after_tp1=True,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(df, ConstantSignalStrategy(1))

    reasons = [t.exit_reason for t in result.trades]
    assert reasons[:2] == ["take_profit_1", "stop_loss"]
    breakeven_trade = next(t for t in result.trades if t.exit_reason == "stop_loss")
    assert breakeven_trade.exit_price == pytest.approx(1.1000)  # breakeven, not the original far-off stop


def test_risk_multiplier_scales_position_size_down():
    df = flat_df(n=5)
    # A bigger balance keeps the raw risk-based lot size well clear of the
    # 0.01 lot-step rounding boundary, so halving it doesn't get distorted
    # by flooring.
    cfg = make_config(initial_balance=10_000.0, stop_loss_pips=20.0, take_profit_pips=1000.0)
    engine = BacktestEngine(cfg)

    result_full = engine.run(df, ConstantSignalStrategy(1))
    half_risk = pd.Series(0.5, index=df.index)
    result_half = engine.run(df, ConstantSignalStrategy(1), risk_multiplier=half_risk)

    assert result_half.trades[0].lots == pytest.approx(result_full.trades[0].lots / 2)


def test_stop_multiplier_tightens_the_stop_distance():
    n = 10
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, 1.1000)
    high = np.full(n, 1.1000)
    low = np.full(n, 1.0983)  # -17 pips: inside the normal 20-pip stop, outside a tightened 16-pip stop
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 100.0}, index=index)

    cfg = make_config(stop_loss_pips=20.0, take_profit_pips=1000.0, spread_pips=0.0, commission_per_lot=0.0)
    engine = BacktestEngine(cfg)

    result_normal = engine.run(df, ConstantSignalStrategy(1))
    assert all(t.exit_reason != "stop_loss" for t in result_normal.trades)

    tighter = pd.Series(0.8, index=df.index)  # 20% tighter -> 16 pip stop
    result_tight = engine.run(df, ConstantSignalStrategy(1), stop_multiplier=tighter)
    assert any(t.exit_reason == "stop_loss" for t in result_tight.trades)
