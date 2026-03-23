"""Monte Carlo simulation via bootstrap resampling of daily returns."""

import math

import numpy as np
import pandas as pd


def run_monte_carlo(
    equity_series: pd.Series,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Bootstrap daily returns to generate simulated equity paths.

    Args:
        equity_series: Daily equity values
        n_simulations: Number of simulation paths
        seed: Random seed for reproducibility

    Returns:
        2D array of shape (n_simulations, n_days) with simulated equity values
    """
    rng = np.random.default_rng(seed)
    returns = equity_series.pct_change().dropna().values
    n_days = len(returns)
    start_value = equity_series.iloc[0]

    # Bootstrap: sample with replacement
    sampled_indices = rng.integers(0, n_days, size=(n_simulations, n_days))
    sampled_returns = returns[sampled_indices]

    # Compute cumulative equity paths
    cum_returns = np.cumprod(1.0 + sampled_returns, axis=1)
    paths = start_value * cum_returns

    return paths


def monte_carlo_percentiles(
    paths: np.ndarray,
    percentiles: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute percentile bands from Monte Carlo paths.

    Returns dict with keys like "p5", "p25", "p50", "p75", "p95".
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    result = {}
    for p in percentiles:
        result[f"p{p}"] = np.percentile(paths, p, axis=0)

    # Also include final equity distribution
    result["final_values"] = paths[:, -1]
    return result


def monte_carlo_summary(
    paths: np.ndarray,
    start_value: float,
    n_days: int,
) -> dict:
    """Compute summary statistics from Monte Carlo simulation paths.

    Computes per-path CAGR, Sharpe, and max drawdown, then returns
    percentile summaries across all paths.

    Args:
        paths: 2D array of shape (n_simulations, n_days).
        start_value: Starting equity value.
        n_days: Number of trading days in the original backtest.

    Returns:
        Dict with median_cagr, p25_cagr, p75_cagr, median_sharpe,
        p25_max_dd, and other summary statistics.
    """
    n_sims = paths.shape[0]
    years = n_days / 252.0 if n_days > 0 else 1.0

    # Per-path CAGR
    final_values = paths[:, -1]
    cagrs = np.where(
        start_value > 0,
        (final_values / start_value) ** (1.0 / years) - 1.0,
        0.0,
    )

    # Per-path Sharpe (annualized from daily returns)
    daily_returns = np.diff(paths, axis=1) / paths[:, :-1]
    means = daily_returns.mean(axis=1)
    stds = daily_returns.std(axis=1)
    sharpes = np.where(stds > 0, (means / stds) * math.sqrt(252), 0.0)

    # Per-path max drawdown
    cummax = np.maximum.accumulate(paths, axis=1)
    drawdowns = np.where(cummax > 0, (paths - cummax) / cummax, 0.0)
    max_dds = drawdowns.min(axis=1)  # Most negative value per path

    return {
        "n_simulations": n_sims,
        "median_cagr": float(np.median(cagrs)),
        "p25_cagr": float(np.percentile(cagrs, 25)),
        "p75_cagr": float(np.percentile(cagrs, 75)),
        "p5_cagr": float(np.percentile(cagrs, 5)),
        "p95_cagr": float(np.percentile(cagrs, 95)),
        "median_sharpe": float(np.median(sharpes)),
        "p25_sharpe": float(np.percentile(sharpes, 25)),
        "p75_sharpe": float(np.percentile(sharpes, 75)),
        "p25_max_dd": float(np.percentile(max_dds, 25)),
        "median_max_dd": float(np.median(max_dds)),
        "p75_max_dd": float(np.percentile(max_dds, 75)),
        "median_final": float(np.median(final_values)),
        "p5_final": float(np.percentile(final_values, 5)),
        "p95_final": float(np.percentile(final_values, 95)),
    }
