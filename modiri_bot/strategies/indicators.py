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


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0):
    """Returns (supertrend_line, is_uptrend)."""
    mid = (high + low) / 2.0
    band = atr(high, low, close, period) * multiplier
    basic_upper = mid + band
    basic_lower = mid - band

    n = len(close)
    c = close.to_numpy()
    bu = basic_upper.to_numpy()
    bl = basic_lower.to_numpy()
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.zeros(n)
    st = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(bu[i]):
            continue
        if np.isnan(final_upper[i - 1]) if i > 0 else True:
            final_upper[i] = bu[i]
            final_lower[i] = bl[i]
            trend[i] = 1
        else:
            final_upper[i] = bu[i] if (bu[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
            final_lower[i] = bl[i] if (bl[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
            if trend[i - 1] == 1:
                trend[i] = -1 if c[i] < final_lower[i] else 1
            else:
                trend[i] = 1 if c[i] > final_upper[i] else -1
        st[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    return pd.Series(st, index=close.index), pd.Series(trend, index=close.index)


def aroon(high: pd.Series, low: pd.Series, period: int = 25):
    periods_since_high = high.rolling(period + 1).apply(lambda x: period - np.argmax(x.to_numpy()), raw=False)
    periods_since_low = low.rolling(period + 1).apply(lambda x: period - np.argmin(x.to_numpy()), raw=False)
    aroon_up = 100 * (period - periods_since_high) / period
    aroon_down = 100 * (period - periods_since_low) / period
    return aroon_up, aroon_down


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    direction = typical.diff()
    pos_flow = raw_flow.where(direction > 0, 0.0).rolling(period, min_periods=period).sum()
    neg_flow = raw_flow.where(direction < 0, 0.0).rolling(period, min_periods=period).sum()
    ratio = pos_flow / neg_flow.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


def awesome_oscillator(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
    median_price = (high + low) / 2.0
    return sma(median_price, fast) - sma(median_price, slow)


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr = true_range(high, low, close)
    tr_sum = tr.rolling(period, min_periods=period).sum().replace(0, np.nan)
    vi_plus = vm_plus.rolling(period, min_periods=period).sum() / tr_sum
    vi_minus = vm_minus.rolling(period, min_periods=period).sum() / tr_sum
    return vi_plus, vi_minus


def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20):
    typical = (high + low + close) / 3.0
    pv = typical * volume
    vwap = pv.rolling(period, min_periods=period).sum() / volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    dev = (typical - vwap).rolling(period, min_periods=period).std()
    return vwap, dev


# --- Candle shape (raw price action, not derived from a rolling formula) ---

def candle_shape(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series):
    """Returns (body, upper_wick, lower_wick, range_) as price-unit Series."""
    body = (close - open_).abs()
    range_ = (high - low).replace(0, np.nan)
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    return body, upper_wick, lower_wick, range_


def heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series):
    """Smoothed candles: each HA candle blends the current bar with the
    previous HA candle, filtering out a lot of single-bar noise."""
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = pd.Series(np.nan, index=open_.index)
    o = open_.to_numpy()
    c = close.to_numpy()
    hc = ha_close.to_numpy()
    ho = np.empty(len(o))
    ho[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(o)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
    ha_open = pd.Series(ho, index=open_.index)
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close


def swing_points(high: pd.Series, low: pd.Series, order: int = 3):
    """A bar is a confirmed swing high/low once `order` bars on both sides
    are known to be lower/higher. Returned series are NaN except at the
    (lagged, look-ahead-free once `order` bars have passed) confirmation
    point, where they carry the swing's price."""
    h, l = high.to_numpy(), low.to_numpy()
    n = len(h)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    for i in range(order, n - order):
        window_h = h[i - order: i + order + 1]
        window_l = l[i - order: i + order + 1]
        if h[i] == window_h.max():
            swing_high[i + order] = h[i]
        if l[i] == window_l.min():
            swing_low[i + order] = l[i]
    return pd.Series(swing_high, index=high.index), pd.Series(swing_low, index=low.index)


def daily_pivot_points(df: pd.DataFrame):
    """Classic floor-trader pivots from the prior day's H/L/C, aligned back
    onto the original (intraday) index without look-ahead."""
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    prev = daily.shift(1)
    pp = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    r1 = 2 * pp - prev["low"]
    s1 = 2 * pp - prev["high"]
    r2 = pp + (prev["high"] - prev["low"])
    s2 = pp - (prev["high"] - prev["low"])
    levels = pd.DataFrame({"pp": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2})
    return levels.reindex(df.index, method="ffill")
