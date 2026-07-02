"""Plain pandas/numpy technical indicators (no TA-Lib dependency needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def donchian_channel(high: pd.Series, low: pd.Series, period: int = 20):
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    return upper, lower


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    rng = (highest - lowest).replace(0, np.nan)
    percent_k = 100 * (close - lowest) / rng
    percent_k = percent_k.fillna(50)
    percent_d = percent_k.rolling(d_period, min_periods=d_period).mean()
    return percent_k, percent_d


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = true_range(high, low, close)
    atr_smoothed = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smoothed.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smoothed.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_line.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    ma = typical.rolling(period, min_periods=period).mean()
    mean_dev = typical.rolling(period, min_periods=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - ma) / (0.015 * mean_dev.replace(0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    rng = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - close) / rng


def parabolic_sar(high: pd.Series, low: pd.Series, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    h, l = high.to_numpy(), low.to_numpy()
    n = len(h)
    sar = np.zeros(n)
    if n == 0:
        return pd.Series(sar, index=high.index)

    uptrend = True
    af = af_step
    ep = h[0]
    sar[0] = l[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if uptrend:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < new_sar:
                uptrend = False
                new_sar = ep
                ep = l[i]
                af = af_step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > new_sar:
                uptrend = True
                new_sar = ep
                ep = h[i]
                af = af_step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)
        sar[i] = new_sar

    return pd.Series(sar, index=high.index)


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
    tenkan = (high.rolling(tenkan_period, min_periods=tenkan_period).max()
              + low.rolling(tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period, min_periods=kijun_period).max()
             + low.rolling(kijun_period, min_periods=kijun_period).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    senkou_b = ((high.rolling(senkou_b_period, min_periods=senkou_b_period).max()
                 + low.rolling(senkou_b_period, min_periods=senkou_b_period).min()) / 2).shift(kijun_period)
    return tenkan, kijun, senkou_a, senkou_b


def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
    return (series / series.shift(period) - 1.0) * 100.0
