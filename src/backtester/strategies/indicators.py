"""Vectorized technical indicators applied to pandas DataFrames."""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence).

    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = series.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns (upper, middle, lower).
    """
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator.

    Returns (%K, %D).
    """
    low_min = df["Low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["High"].rolling(window=k_period, min_periods=k_period).max()
    k = 100.0 * (df["Close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder's smoothing)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder's smoothing (consistent with rsi implementation)
    alpha = 1.0 / period
    atr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_smooth
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_smooth

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx_series


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    sign = np.sign(df["Close"].diff())
    return (sign * df["Volume"]).fillna(0).cumsum()


# ── Additional swing-trading indicators ────────────────────────────


def keltner(
    df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5, atr_period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels (EMA-based with ATR bands).

    Returns (upper, middle, lower).
    """
    middle = df["Close"].ewm(span=period, min_periods=period, adjust=False).mean()
    atr_val = atr(df, atr_period)
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    return upper, middle, lower


def donchian(
    df: pd.DataFrame, period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channels (highest high / lowest low over period).

    Returns (upper, middle, lower).
    """
    upper = df["High"].rolling(window=period, min_periods=period).max()
    lower = df["Low"].rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R oscillator. Range: -100 (oversold) to 0 (overbought)."""
    high_max = df["High"].rolling(window=period, min_periods=period).max()
    low_min = df["Low"].rolling(window=period, min_periods=period).min()
    denom = high_max - low_min
    return -100.0 * (high_max - df["Close"]) / denom.replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (volume-weighted RSI)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw_mf = tp * df["Volume"]
    delta = tp.diff()
    pos_mf = raw_mf.where(delta > 0, 0.0)
    neg_mf = raw_mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change (momentum as percentage)."""
    prev = series.shift(period)
    return ((series - prev) / prev.replace(0, np.nan)) * 100.0


def psar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02,
         af_max: float = 0.20) -> pd.Series:
    """Parabolic SAR."""
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(close)
    sar = np.full(n, np.nan)

    if n < 2:
        return pd.Series(sar, index=df.index)

    # Initialize
    bull = close[1] > close[0]
    af = af_start
    if bull:
        sar[0] = low[0]
        ep = high[0]
    else:
        sar[0] = high[0]
        ep = low[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            # SAR can't be above prior two lows
            if i >= 2:
                sar[i] = min(sar[i], low[i - 1], low[i - 2])
            else:
                sar[i] = min(sar[i], low[i - 1])

            if low[i] < sar[i]:
                # Reversal to bear
                bull = False
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            # SAR can't be below prior two highs
            if i >= 2:
                sar[i] = max(sar[i], high[i - 1], high[i - 2])
            else:
                sar[i] = max(sar[i], high[i - 1])

            if high[i] > sar[i]:
                # Reversal to bull
                bull = True
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=df.index)


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, displacement: int = 26,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku Cloud.

    Returns (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span).
    Note: senkou spans are shifted forward by `displacement` periods.
    """
    high = df["High"]
    low = df["Low"]

    tenkan_sen = (high.rolling(tenkan, min_periods=tenkan).max() +
                  low.rolling(tenkan, min_periods=tenkan).min()) / 2.0
    kijun_sen = (high.rolling(kijun, min_periods=kijun).max() +
                 low.rolling(kijun, min_periods=kijun).min()) / 2.0

    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(displacement)
    senkou_b_val = ((high.rolling(senkou_b, min_periods=senkou_b).max() +
                     low.rolling(senkou_b, min_periods=senkou_b).min()) / 2.0).shift(displacement)
    chikou = df["Close"].shift(-displacement)

    return tenkan_sen, kijun_sen, senkou_a, senkou_b_val, chikou


def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling VWAP (Volume Weighted Average Price).

    For daily data, uses a rolling window as a proxy since true VWAP
    is an intraday concept.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol = tp * df["Volume"]
    cum_tp_vol = tp_vol.rolling(window=period, min_periods=period).sum()
    cum_vol = df["Volume"].rolling(window=period, min_periods=period).sum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)
