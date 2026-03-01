"""Regime classification and performance breakdown.

Classifies market regimes (bull/bear/sideways) and volatility regimes
(low/medium/high), then computes performance metrics for each regime period.
"""

import numpy as np
import pandas as pd


def classify_market_regime(
    benchmark_prices: pd.Series,
    sma_window: int = 200,
    sideways_band: float = 0.02,
) -> pd.Series:
    """Classify each day as 'bull', 'bear', or 'sideways' based on SMA position.

    Parameters
    ----------
    benchmark_prices : pd.Series
        Benchmark Close prices, indexed by date.
    sma_window : int
        Lookback window for the simple moving average.
    sideways_band : float
        Fraction (e.g. 0.02 = +/-2%) around SMA for 'sideways' classification.

    Returns
    -------
    pd.Series
        Regime labels indexed like the input. First ``sma_window`` days are
        'unknown' (insufficient data for SMA).
    """
    if benchmark_prices.empty:
        return pd.Series(dtype=str)

    sma = benchmark_prices.rolling(window=sma_window, min_periods=sma_window).mean()

    labels = pd.Series("unknown", index=benchmark_prices.index, dtype=str)

    valid = sma.notna()
    if not valid.any():
        return labels

    ratio = (benchmark_prices[valid] - sma[valid]) / sma[valid]

    bull_mask = ratio > sideways_band
    bear_mask = ratio < -sideways_band
    sideways_mask = ~bull_mask & ~bear_mask

    labels.loc[bull_mask.index[bull_mask]] = "bull"
    labels.loc[bear_mask.index[bear_mask]] = "bear"
    labels.loc[sideways_mask.index[sideways_mask]] = "sideways"

    return labels


def classify_volatility_regime(
    returns: pd.Series,
    window: int = 63,
    percentiles: tuple[float, float] = (33, 67),
) -> pd.Series:
    """Classify each day into a volatility regime using rolling realized vol.

    Parameters
    ----------
    returns : pd.Series
        Daily returns, indexed by date.
    window : int
        Rolling window for realized volatility (63 ~= 3 months).
    percentiles : tuple[float, float]
        Percentile boundaries between low/medium and medium/high vol.

    Returns
    -------
    pd.Series
        Labels: 'low_vol', 'medium_vol', 'high_vol', or 'unknown'
        (for the initial warmup period).
    """
    if returns.empty:
        return pd.Series(dtype=str)

    # Annualized rolling realized volatility
    rolling_vol = returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)

    labels = pd.Series("unknown", index=returns.index, dtype=str)

    valid = rolling_vol.dropna()
    if valid.empty:
        return labels

    low_thresh = np.percentile(valid.values, percentiles[0])
    high_thresh = np.percentile(valid.values, percentiles[1])

    low_mask = rolling_vol <= low_thresh
    high_mask = rolling_vol > high_thresh
    medium_mask = (rolling_vol > low_thresh) & (rolling_vol <= high_thresh)

    # Only assign where rolling_vol is valid (not NaN)
    valid_mask = rolling_vol.notna()
    labels.loc[valid_mask & low_mask] = "low_vol"
    labels.loc[valid_mask & medium_mask] = "medium_vol"
    labels.loc[valid_mask & high_mask] = "high_vol"

    return labels


def regime_performance(
    equity_curve: pd.Series,
    regime_labels: pd.Series,
    annual_factor: int = 252,
) -> pd.DataFrame:
    """Compute performance metrics for each regime.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio values indexed by date.
    regime_labels : pd.Series
        Regime labels indexed by date (must align with equity_curve).
    annual_factor : int
        Trading days per year for annualization.

    Returns
    -------
    pd.DataFrame
        One row per unique regime, with columns: total_return,
        annualized_return, annualized_volatility, sharpe_ratio,
        max_drawdown, trading_days, pct_of_time.
    """
    if equity_curve.empty or regime_labels.empty:
        return pd.DataFrame()

    # Align on common dates
    common = equity_curve.index.intersection(regime_labels.index)
    if len(common) < 2:
        return pd.DataFrame()

    equity = equity_curve.loc[common]
    labels = regime_labels.loc[common]
    daily_returns = equity.pct_change()

    total_days = len(common)
    unique_regimes = sorted(labels.unique())

    rows = []
    for regime in unique_regimes:
        mask = labels == regime
        regime_days = mask.sum()
        if regime_days < 1:
            continue

        regime_returns = daily_returns[mask].dropna()

        # Total return: compound the daily returns during this regime
        if len(regime_returns) == 0:
            total_ret = 0.0
        else:
            total_ret = (1 + regime_returns).prod() - 1.0

        # Annualized return
        if regime_days <= 1:
            ann_return = 0.0
        else:
            ann_return = (1 + total_ret) ** (annual_factor / regime_days) - 1.0

        # Annualized volatility
        if len(regime_returns) < 2:
            ann_vol = 0.0
        else:
            ann_vol = regime_returns.std() * np.sqrt(annual_factor)

        # Sharpe ratio (rf=0)
        if ann_vol == 0.0:
            sharpe = 0.0
        else:
            sharpe = ann_return / ann_vol

        # Max drawdown within regime periods
        # Build a sub-equity curve by compounding regime-day returns
        if len(regime_returns) == 0:
            mdd = 0.0
        else:
            sub_equity = (1 + regime_returns).cumprod()
            cummax = sub_equity.cummax()
            drawdown = (sub_equity - cummax) / cummax
            mdd = drawdown.min()

        rows.append(
            {
                "regime": regime,
                "total_return": total_ret,
                "annualized_return": ann_return,
                "annualized_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": mdd,
                "trading_days": regime_days,
                "pct_of_time": regime_days / total_days,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("regime")
    return df


def regime_summary(
    equity_curve: pd.Series,
    benchmark_prices: pd.Series,
    returns: pd.Series | None = None,
    sma_window: int = 200,
    vol_window: int = 63,
) -> dict:
    """Convenience function: classify regimes and compute performance for each.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio values indexed by date.
    benchmark_prices : pd.Series
        Benchmark Close prices indexed by date.
    returns : pd.Series, optional
        Daily portfolio returns. If None, computed from ``equity_curve``.
    sma_window : int
        Window for market-regime SMA.
    vol_window : int
        Window for volatility-regime rolling vol.

    Returns
    -------
    dict
        Keys: ``market_regime_perf`` (DataFrame), ``vol_regime_perf`` (DataFrame),
        ``market_labels`` (Series), ``vol_labels`` (Series).
    """
    if returns is None:
        returns = equity_curve.pct_change().dropna()

    market_labels = classify_market_regime(benchmark_prices, sma_window=sma_window)
    vol_labels = classify_volatility_regime(returns, window=vol_window)

    market_perf = regime_performance(equity_curve, market_labels)
    vol_perf = regime_performance(equity_curve, vol_labels)

    return {
        "market_regime_perf": market_perf,
        "vol_regime_perf": vol_perf,
        "market_labels": market_labels,
        "vol_labels": vol_labels,
    }
