from __future__ import annotations

from typing import Sequence

import pandas as pd

from .base import Strategy


class EnsembleStrategy(Strategy):
    """Combines several strategies into one signal via weighted voting.

    Each sub-strategy votes -1/0/1; votes are multiplied by that strategy's
    weight and summed. The ensemble goes long/short only if the weighted
    vote clears `threshold`, otherwise it stays flat. This is what the
    optimizer tunes (weights + threshold) once individual strategies have
    been parameter-optimized.
    """

    name = "ensemble"

    def __init__(
        self,
        strategies: Sequence[Strategy],
        weights: Sequence[float] | None = None,
        threshold: float = 0.0,
    ):
        weights = list(weights) if weights is not None else [1.0] * len(strategies)
        if len(weights) != len(strategies):
            raise ValueError("weights must have the same length as strategies")
        super().__init__(weights=weights, threshold=threshold)
        self.strategies = list(strategies)
        self.weights = weights
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        total_weight = sum(abs(w) for w in self.weights) or 1.0
        combined = pd.Series(0.0, index=df.index)
        for strategy, weight in zip(self.strategies, self.weights):
            combined = combined.add(strategy.generate_signals(df) * weight, fill_value=0.0)
        combined = combined / total_weight

        out = pd.Series(0, index=df.index, dtype=int)
        out[combined > self.threshold] = 1
        out[combined < -self.threshold] = -1
        return out

    def __repr__(self) -> str:
        parts = ", ".join(f"{s.name}(w={w:.2f})" for s, w in zip(self.strategies, self.weights))
        return f"Ensemble[{parts}, threshold={self.threshold:.2f}]"
