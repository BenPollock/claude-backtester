"""Monte Carlo simulation via bootstrap resampling of daily returns."""

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
