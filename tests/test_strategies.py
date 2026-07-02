import numpy as np
import pandas as pd
import pytest

from modiri_bot.strategies.adx_trend import ADXTrendStrategy
from modiri_bot.strategies.aroon_trend import AroonTrendStrategy
from modiri_bot.strategies.atr_channel_breakout import ATRChannelBreakoutStrategy
from modiri_bot.strategies.awesome_oscillator_momentum import AwesomeOscillatorMomentumStrategy
from modiri_bot.strategies.bollinger_breakout import BollingerBreakoutStrategy
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
from modiri_bot.strategies.stochastic_reversion import StochasticReversionStrategy
from modiri_bot.strategies.supertrend_strategy import SuperTrendStrategy
from modiri_bot.strategies.trend_pullback import TrendPullbackStrategy
from modiri_bot.strategies.vortex_trend import VortexTrendStrategy
from modiri_bot.strategies.vwap_reversion import VWAPReversionStrategy
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
