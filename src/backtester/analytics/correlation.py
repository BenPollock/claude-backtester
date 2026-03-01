"""Correlation analysis, portfolio concentration, and sector exposure metrics."""

import numpy as np
import pandas as pd


def compute_correlation_matrix(
    price_data: dict[str, pd.DataFrame],
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of daily close-to-close returns.

    Args:
        price_data: Dict of ticker -> OHLCV DataFrame (must contain 'Close' column).
        tickers: Subset of tickers to include. If None, uses all keys from price_data.

    Returns:
        Symmetric correlation matrix DataFrame (tickers x tickers).
        Single-ticker case returns a 1x1 DataFrame with correlation 1.0.
    """
    if tickers is None:
        tickers = list(price_data.keys())

    if not tickers:
        return pd.DataFrame()

    # Build a DataFrame of daily returns for each ticker, aligned by date
    returns = {}
    for t in tickers:
        if t not in price_data:
            continue
        close = price_data[t]["Close"]
        returns[t] = close.pct_change().dropna()

    if not returns:
        return pd.DataFrame()

    returns_df = pd.DataFrame(returns)
    # Drop rows where any ticker has NaN (align dates)
    returns_df = returns_df.dropna()

    return returns_df.corr()


def compute_rolling_correlation(
    price_data: dict[str, pd.DataFrame],
    ticker_a: str,
    ticker_b: str,
    window: int = 63,
) -> pd.Series:
    """Rolling pairwise Pearson correlation between two tickers.

    Args:
        price_data: Dict of ticker -> OHLCV DataFrame.
        ticker_a: First ticker symbol.
        ticker_b: Second ticker symbol.
        window: Rolling window size in trading days (default 63 ~ one quarter).

    Returns:
        Series indexed by date with rolling correlation values.
    """
    close_a = price_data[ticker_a]["Close"]
    close_b = price_data[ticker_b]["Close"]

    ret_a = close_a.pct_change().dropna()
    ret_b = close_b.pct_change().dropna()

    # Align on common dates
    aligned = pd.DataFrame({"a": ret_a, "b": ret_b}).dropna()

    return aligned["a"].rolling(window=window).corr(aligned["b"]).dropna()


def compute_hhi(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index: sum of squared portfolio weights.

    Args:
        weights: Dict of ticker -> weight as fraction (e.g. 0.25).
            Weights should sum to 1.0 for meaningful interpretation.

    Returns:
        Float between 0 and 1.  HHI = 1.0 means fully concentrated in one
        position; HHI = 1/N means perfectly equal-weighted across N positions.
        Returns 0.0 for empty weights.
    """
    if not weights:
        return 0.0
    return sum(w ** 2 for w in weights.values())


def compute_portfolio_concentration(positions: dict[str, float]) -> dict:
    """Compute concentration metrics from a ticker -> market_value mapping.

    Args:
        positions: Dict of ticker -> market value (dollar amount).

    Returns:
        Dict with keys:
            - weights: dict of ticker -> weight (fraction of total)
            - hhi: Herfindahl-Hirschman Index
            - effective_n: Effective number of positions (1 / HHI)
            - max_weight: Largest single-position weight
            - max_weight_ticker: Ticker with the largest weight
    """
    if not positions:
        return {
            "weights": {},
            "hhi": 0.0,
            "effective_n": 0.0,
            "max_weight": 0.0,
            "max_weight_ticker": "",
        }

    total = sum(positions.values())
    if total == 0:
        return {
            "weights": {t: 0.0 for t in positions},
            "hhi": 0.0,
            "effective_n": 0.0,
            "max_weight": 0.0,
            "max_weight_ticker": "",
        }

    weights = {t: v / total for t, v in positions.items()}
    hhi = compute_hhi(weights)
    effective_n = 1.0 / hhi if hhi > 0 else 0.0
    max_ticker = max(weights, key=weights.get)

    return {
        "weights": weights,
        "hhi": hhi,
        "effective_n": effective_n,
        "max_weight": weights[max_ticker],
        "max_weight_ticker": max_ticker,
    }


def compute_sector_exposure(
    positions: dict[str, float],
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Aggregate position values by sector.

    Args:
        positions: Dict of ticker -> market value (dollar amount).
        sector_map: Dict of ticker -> sector name.
            Tickers not found in sector_map are assigned to "Unknown".

    Returns:
        DataFrame with columns: sector, total_value, weight, ticker_count.
        Sorted by total_value descending. Empty DataFrame (with correct columns)
        if positions is empty.
    """
    columns = ["sector", "total_value", "weight", "ticker_count"]

    if not positions:
        return pd.DataFrame(columns=columns)

    total_value = sum(positions.values())

    # Group by sector
    sector_agg: dict[str, dict] = {}
    for ticker, value in positions.items():
        sector = sector_map.get(ticker, "Unknown")
        if sector not in sector_agg:
            sector_agg[sector] = {"total_value": 0.0, "ticker_count": 0}
        sector_agg[sector]["total_value"] += value
        sector_agg[sector]["ticker_count"] += 1

    rows = []
    for sector, agg in sector_agg.items():
        weight = agg["total_value"] / total_value if total_value > 0 else 0.0
        rows.append({
            "sector": sector,
            "total_value": agg["total_value"],
            "weight": weight,
            "ticker_count": agg["ticker_count"],
        })

    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values("total_value", ascending=False).reset_index(drop=True)
    return df
