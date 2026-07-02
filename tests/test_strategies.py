import numpy as np
import pandas as pd
import pytest

from modiri_bot.strategies.adx_trend import ADXTrendStrategy
from modiri_bot.strategies.aroon_trend import AroonTrendStrategy
from modiri_bot.strategies.atr_channel_breakout import ATRChannelBreakoutStrategy
from modiri_bot.strategies.awesome_oscillator_momentum import AwesomeOscillatorMomentumStrategy
from modiri_bot.strategies.bollinger_breakout import BollingerBreakoutStrategy
from modiri_bot.strategies.candlestick_patterns import CandlestickPatternStrategy
from modiri_bot.strategies.cci_reversion import CCIReversionStrategy
from modiri_bot.strategies.donchian_breakout import DonchianBreakoutStrategy
from modiri_bot.strategies.ensemble import EnsembleStrategy
from modiri_bot.strategies.ichimoku_strategy import IchimokuStrategy
from modiri_bot.strategies.ma_crossover import MACrossoverStrategy
from modiri_bot.strategies.macd_trend import MACDTrendStrategy
from modiri_bot.strategies.mfi_reversion import MFIReversionStrategy
from modiri_bot.strategies.mtf_trend_filter import MTFTrendFilterStrategy
from modiri_bot.strategies.parabolic_sar_trend import ParabolicSARTrendStrategy
from modiri_bot.strategies.rsi_reversion import RSIReversionStrategy
from modiri_bot.strategies.session_filtered import SessionFilteredStrategy
from modiri_bot.strategies.stochastic_reversion import StochasticReversionStrategy
from modiri_bot.strategies.supertrend_strategy import SuperTrendStrategy
from modiri_bot.strategies.trend_pullback import TrendPullbackStrategy
from modiri_bot.strategies.volume_spike import VolumeSpikeStrategy
from modiri_bot.strategies.vortex_trend import VortexTrendStrategy
from modiri_bot.strategies.vwap_reversion import VWAPReversionStrategy
from modiri_bot.strategies.wick_rejection import WickRejectionStrategy
from modiri_bot.strategies.williams_r_reversion import WilliamsRReversionStrategy

ALL_STRATEGIES = [
    MACrossoverStrategy(fast_period=5, slow_period=20),
    RSIReversionStrategy(period=14),
    MACDTrendStrategy(),
    BollingerBreakoutStrategy(period=20),
    DonchianBreakoutStrategy(period=20),
    IchimokuStrategy(),
    StochasticReversionStrategy(),
    ADXTrendStrategy(),
    ATRChannelBreakoutStrategy(),
    CCIReversionStrategy(),
    WilliamsRReversionStrategy(),
    ParabolicSARTrendStrategy(),
    TrendPullbackStrategy(),
    MTFTrendFilterStrategy(),
    SuperTrendStrategy(),
    AroonTrendStrategy(),
    MFIReversionStrategy(),
    AwesomeOscillatorMomentumStrategy(),
    VortexTrendStrategy(),
    VWAPReversionStrategy(),
    CandlestickPatternStrategy(),
    WickRejectionStrategy(),
    VolumeSpikeStrategy(),
    SessionFilteredStrategy(),
]


def make_trending_df(n=300, drift=0.0005, seed=1):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.0003, n)
    close = 1.10 * np.exp(np.cumsum(drift + noise))
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.0006,
            "low": close * 0.9994,
            "close": close,
            "volume": 100.0,
        },
        index=index,
    )


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
def test_signals_are_valid_and_aligned(strategy):
    df = make_trending_df()
    signals = strategy.generate_signals(df)
    assert isinstance(signals, pd.Series)
    assert list(signals.index) == list(df.index)
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert not signals.isna().any()


def test_ma_crossover_goes_long_in_strong_uptrend():
    df = make_trending_df(n=200, drift=0.002, seed=2)
    strategy = MACrossoverStrategy(fast_period=5, slow_period=20)
    signals = strategy.generate_signals(df)
    assert signals.iloc[-1] == 1


def test_donchian_breakout_flips_on_new_extremes():
    n = 100
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.concatenate([np.full(50, 1.10), np.linspace(1.10, 1.20, 50)])
    df = pd.DataFrame(
        {"open": close, "high": close + 0.0005, "low": close - 0.0005, "close": close, "volume": 100.0},
        index=index,
    )
    strategy = DonchianBreakoutStrategy(period=20)
    signals = strategy.generate_signals(df)
    assert signals.iloc[-1] == 1


def test_ensemble_majority_vote():
    df = make_trending_df(n=200, drift=0.002, seed=3)
    always_long = MACrossoverStrategy(fast_period=2, slow_period=3)
    ensemble = EnsembleStrategy([always_long, always_long], weights=[1.0, 1.0], threshold=0.5)
    signals = ensemble.generate_signals(df)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_ensemble_requires_matching_weight_length():
    with pytest.raises(ValueError):
        EnsembleStrategy([MACrossoverStrategy(), MACDTrendStrategy()], weights=[1.0])


def test_bullish_engulfing_triggers_a_long():
    n = 60
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, 1.1000)
    open_ = np.full(n, 1.1000)
    # A clean bearish candle at n-2, then a bullish candle at n-1 that
    # fully engulfs it.
    open_[-2], close[-2] = 1.1010, 1.0990
    open_[-1], close[-1] = 1.0985, 1.1015
    df = pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 0.0002,
         "low": np.minimum(open_, close) - 0.0002, "close": close, "volume": 100.0},
        index=index,
    )
    strategy = CandlestickPatternStrategy(trend_period=10, hold_bars=3)
    signals = strategy.generate_signals(df)
    assert signals.iloc[-1] == 1


def test_wick_rejection_needs_both_long_wick_and_counter_trend():
    n = 60
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    # A steady downtrend so close < EMA(trend), then one candle with a very
    # long lower wick (rejection of lower prices) near the end.
    close = 1.15 - np.arange(n) * 0.0003
    open_ = close.copy()
    high = close + 0.0002
    low = close - 0.0002
    low[-1] = close[-1] - 0.005  # long lower wick on the last candle
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 100.0}, index=index)

    strategy = WickRejectionStrategy(trend_period=10, wick_ratio_min=0.6, hold_bars=3)
    signals = strategy.generate_signals(df)
    assert signals.iloc[-1] == 1


def test_volume_spike_follows_direction_of_the_spike_bar():
    n = 40
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, 1.1000)
    open_ = np.full(n, 1.1000)
    volume = np.full(n, 100.0)
    # A clearly bullish candle with volume far above the rolling average.
    open_[-1], close[-1] = 1.0990, 1.1030
    volume[-1] = 1000.0
    df = pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 0.0002,
         "low": np.minimum(open_, close) - 0.0002, "close": close, "volume": volume},
        index=index,
    )
    strategy = VolumeSpikeStrategy(volume_period=10, spike_mult=2.0, hold_bars=3)
    signals = strategy.generate_signals(df)
    assert signals.iloc[-1] == 1


def test_session_filter_zeroes_signal_outside_the_window():
    n = 48
    index = pd.date_range("2024-01-01", periods=n, freq="h")  # covers all 24 hours twice
    close = 1.10 + np.arange(n) * 0.0005  # steady uptrend -> base signal would be long throughout
    df = pd.DataFrame(
        {"open": close, "high": close + 0.0002, "low": close - 0.0002, "close": close, "volume": 100.0},
        index=index,
    )
    strategy = SessionFilteredStrategy(fast_period=2, slow_period=5, session_start_hour=8, session_end_hour=16)
    signals = strategy.generate_signals(df)
    outside_hours = signals.index.hour[(signals.index.hour < 8) | (signals.index.hour >= 16)]
    assert len(outside_hours) > 0
    assert (signals[(signals.index.hour < 8) | (signals.index.hour >= 16)] == 0).all()
