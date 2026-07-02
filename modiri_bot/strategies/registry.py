"""Maps strategy names (as used in config/CLI) to classes and their
default parameter search grids for the optimizer."""
from __future__ import annotations

from .adx_trend import ADXTrendStrategy
from .aroon_trend import AroonTrendStrategy
from .atr_channel_breakout import ATRChannelBreakoutStrategy
from .awesome_oscillator_momentum import AwesomeOscillatorMomentumStrategy
from .bollinger_breakout import BollingerBreakoutStrategy
from .candlestick_patterns import CandlestickPatternStrategy
from .cci_reversion import CCIReversionStrategy
from .donchian_breakout import DonchianBreakoutStrategy
from .ichimoku_strategy import IchimokuStrategy
from .ma_crossover import MACrossoverStrategy
from .macd_trend import MACDTrendStrategy
from .mfi_reversion import MFIReversionStrategy
from .mtf_trend_filter import MTFTrendFilterStrategy
from .parabolic_sar_trend import ParabolicSARTrendStrategy
from .rsi_reversion import RSIReversionStrategy
from .stochastic_reversion import StochasticReversionStrategy
from .supertrend_strategy import SuperTrendStrategy
from .trend_pullback import TrendPullbackStrategy
from .vortex_trend import VortexTrendStrategy
from .vwap_reversion import VWAPReversionStrategy
from .wick_rejection import WickRejectionStrategy
from .williams_r_reversion import WilliamsRReversionStrategy

STRATEGY_CLASSES = {
    "ma_crossover": MACrossoverStrategy,
    "rsi_reversion": RSIReversionStrategy,
    "macd_trend": MACDTrendStrategy,
    "bollinger_breakout": BollingerBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "ichimoku": IchimokuStrategy,
    "stochastic_reversion": StochasticReversionStrategy,
    "adx_trend": ADXTrendStrategy,
    "atr_channel_breakout": ATRChannelBreakoutStrategy,
    "cci_reversion": CCIReversionStrategy,
    "williams_r_reversion": WilliamsRReversionStrategy,
    "parabolic_sar_trend": ParabolicSARTrendStrategy,
    "trend_pullback": TrendPullbackStrategy,
    "mtf_trend_filter": MTFTrendFilterStrategy,
    "supertrend": SuperTrendStrategy,
    "aroon_trend": AroonTrendStrategy,
    "mfi_reversion": MFIReversionStrategy,
    "awesome_oscillator_momentum": AwesomeOscillatorMomentumStrategy,
    "vortex_trend": VortexTrendStrategy,
    "vwap_reversion": VWAPReversionStrategy,
    "candlestick_patterns": CandlestickPatternStrategy,
    "wick_rejection": WickRejectionStrategy,
}

# Reasonably small grids -- kept tight so walk-forward optimization runs in
# seconds to minutes rather than hours. Widen them once you've got real data
# and time to spare. modiri_bot/backtest/mass_search.py has much finer
# grids for the large-scale combination search.
PARAM_GRIDS = {
    "ma_crossover": {
        "fast_period": [5, 10, 20],
        "slow_period": [30, 50, 100],
        "use_ema": [True, False],
    },
    "rsi_reversion": {
        "period": [7, 14, 21],
        "oversold": [20, 30],
        "overbought": [70, 80],
    },
    "macd_trend": {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal": [9],
    },
    "bollinger_breakout": {
        "period": [14, 20, 30],
        "num_std": [1.5, 2.0, 2.5],
    },
    "donchian_breakout": {
        "period": [10, 20, 55],
    },
    "ichimoku": {
        "tenkan_period": [7, 9, 12],
        "kijun_period": [22, 26, 30],
        "senkou_b_period": [44, 52, 60],
    },
    "stochastic_reversion": {
        "k_period": [9, 14, 21],
        "oversold": [15, 20, 25],
        "overbought": [75, 80, 85],
    },
    "adx_trend": {
        "period": [10, 14, 21],
        "adx_threshold": [20, 25, 30],
    },
    "atr_channel_breakout": {
        "ema_period": [14, 20, 30],
        "atr_mult": [1.5, 2.0, 2.5],
    },
    "cci_reversion": {
        "period": [14, 20, 30],
        "threshold": [80, 100, 150],
    },
    "williams_r_reversion": {
        "period": [9, 14, 21],
        "oversold": [-90, -80, -70],
        "overbought": [-30, -20, -10],
    },
    "parabolic_sar_trend": {
        "af_step": [0.01, 0.02, 0.03],
        "af_max": [0.15, 0.2, 0.3],
    },
    "trend_pullback": {
        "trend_period": [30, 50, 80],
        "rsi_period": [9, 14, 21],
    },
    "mtf_trend_filter": {
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 50],
        "htf_trend_period": [10, 20, 30],
    },
    "supertrend": {
        "period": [7, 10, 14],
        "multiplier": [2.0, 3.0, 4.0],
    },
    "aroon_trend": {
        "period": [14, 25, 40],
        "threshold": [60, 70, 80],
    },
    "mfi_reversion": {
        "period": [9, 14, 21],
        "oversold": [15, 20, 25],
        "overbought": [75, 80, 85],
    },
    "awesome_oscillator_momentum": {
        "fast": [5],
        "slow": [21, 34],
    },
    "vortex_trend": {
        "period": [10, 14, 21],
    },
    "vwap_reversion": {
        "period": [14, 20, 30],
        "num_std": [1.5, 2.0, 2.5],
    },
    "candlestick_patterns": {
        "trend_period": [30, 50, 80],
        "body_ratio_max": [0.3, 0.35, 0.4],
        "wick_ratio_min": [1.5, 2.0, 2.5],
    },
    "wick_rejection": {
        "trend_period": [30, 50, 80],
        "wick_ratio_min": [0.5, 0.6, 0.7],
    },
}


def strategy_from_dict(config: dict) -> "Strategy":
    """Reconstruct a Strategy (single or ensemble) from the JSON shape
    written by scripts/optimize_strategies.py."""
    from .ensemble import EnsembleStrategy

    if config["type"] == "single":
        return STRATEGY_CLASSES[config["name"]](**config["params"])

    members = [STRATEGY_CLASSES[m["name"]](**m["params"]) for m in config["members"]]
    weights = [m["weight"] for m in config["members"]]
    return EnsembleStrategy(members, weights=weights, threshold=config["threshold"])
