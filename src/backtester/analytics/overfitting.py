"""Overfitting detection metrics (Gap 7).

- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Permutation test for strategy significance
"""

import math
import numpy as np
import pandas as pd


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    variance_of_sharpes: float,
    n_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014).

    Adjusts the observed Sharpe ratio for the number of trials
    (parameter combos tested), using the expected maximum of N
    draws from the null distribution.

    Args:
        observed_sharpe: The best Sharpe ratio found.
        num_trials: Number of parameter combos tested.
        variance_of_sharpes: Variance of Sharpe ratios across all combos.
        n_returns: Number of daily returns used.
        skewness: Skewness of daily returns (default 0).
        kurtosis: Excess kurtosis of daily returns (default 3 = normal).

    Returns:
        DSR value between 0 and 1. Higher means less likely to be
        due to overfitting.
    """
    if num_trials <= 1 or variance_of_sharpes <= 0 or n_returns <= 1:
        return 0.0

    # Expected max Sharpe from N iid normal draws (Euler-Mascheroni approx)
    euler_mascheroni = 0.5772156649
    e_max_sharpe = (
        math.sqrt(variance_of_sharpes)
        * ((1 - euler_mascheroni) * _norm_ppf(1 - 1.0 / num_trials)
           + euler_mascheroni * _norm_ppf(1 - 1.0 / (num_trials * math.e)))
    )

    # Standard error of Sharpe (with skew/kurtosis correction)
    se = math.sqrt(
        (1 + 0.5 * observed_sharpe ** 2
         - skewness * observed_sharpe
         + ((kurtosis - 3) / 4) * observed_sharpe ** 2)
        / (n_returns - 1)
    )

    if se <= 0:
        return 0.0

    z = (observed_sharpe - e_max_sharpe) / se
    return _norm_cdf(z)


def permutation_test(
    equity_series: pd.Series,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for strategy significance.

    Shuffles daily returns N times, recomputes Sharpe each time,
    returns p-value = fraction of null Sharpes >= observed Sharpe.

    Args:
        equity_series: Daily equity curve.
        n_permutations: Number of random shuffles.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'observed_sharpe', 'p_value', 'null_sharpes'.
    """
    returns = equity_series.pct_change().dropna().values
    if len(returns) < 2:
        return {"observed_sharpe": 0.0, "p_value": 1.0, "null_sharpes": []}

    observed = _sharpe_from_returns(returns)

    rng = np.random.default_rng(seed)
    null_sharpes = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(returns)
        null_sharpes.append(_sharpe_from_returns(shuffled))

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= observed).mean()

    return {
        "observed_sharpe": observed,
        "p_value": float(p_value),
        "null_sharpes": null_sharpes.tolist(),
    }


def _sharpe_from_returns(returns: np.ndarray) -> float:
    std = returns.std()
    if std == 0:
        return 0.0
    return (returns.mean() / std) * math.sqrt(252)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (percent-point function).

    Uses the rational approximation from Abramowitz and Stegun.
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        sign = -1.0
        p = 1.0 - p
    else:
        sign = 1.0
        p = p

    t = math.sqrt(-2.0 * math.log(1.0 - p))
    # Rational approximation coefficients
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
    return sign * result
