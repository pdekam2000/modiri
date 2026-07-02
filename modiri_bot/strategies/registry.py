"""Maps strategy names (as used in config/CLI) to classes and their
default parameter search grids for the optimizer."""
from __future__ import annotations

from .bollinger_breakout import BollingerBreakoutStrategy
from .donchian_breakout import DonchianBreakoutStrategy
from .ma_crossover import MACrossoverStrategy
from .macd_trend import MACDTrendStrategy
from .rsi_reversion import RSIReversionStrategy

STRATEGY_CLASSES = {
    "ma_crossover": MACrossoverStrategy,
    "rsi_reversion": RSIReversionStrategy,
    "macd_trend": MACDTrendStrategy,
    "bollinger_breakout": BollingerBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}

# Reasonably small grids -- kept tight so walk-forward optimization runs in
# seconds to minutes rather than hours. Widen them once you've got real data
# and time to spare.
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
