"""Overfitting detection metrics (Gap 7).

- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Permutation test for strategy significance (sign-flip method)
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
        observed_sharpe: The best (annualized) Sharpe ratio found.
        num_trials: Number of parameter combos tested.
        variance_of_sharpes: Variance of annualized Sharpe ratios across
            all combos (or estimated via estimate_sharpe_variance).
        n_returns: Number of daily returns used.
        skewness: Skewness of daily returns (default 0).
        kurtosis: Kurtosis of daily returns (default 3 = normal).

    Returns:
        DSR value between 0 and 1. Higher means less likely to be
        due to overfitting.
    """
    if n_returns <= 1:
        return 0.0
    if variance_of_sharpes <= 0:
        return 0.0

    # Convert annualized Sharpe to daily for the SE correction terms.
    # The Lo (2002) SE formula uses the daily Sharpe ratio:
    #   SE(SR_daily) = sqrt((1 + 0.5*SR_d^2 - skew*SR_d + ((k-3)/4)*SR_d^2) / (T-1))
    # Then SE(SR_annual) = SE(SR_daily) * sqrt(252)
    sr_daily = observed_sharpe / math.sqrt(252) if observed_sharpe != 0 else 0.0

    if num_trials <= 1:
        # No multiple-testing penalty: test if observed Sharpe is
        # significantly above zero using its standard error.
        se = math.sqrt(
            (1 + 0.5 * sr_daily ** 2
             - skewness * sr_daily
             + ((kurtosis - 3) / 4) * sr_daily ** 2)
            / (n_returns - 1)
        ) * math.sqrt(252)
        if se <= 0:
            return 0.0
        return _norm_cdf(observed_sharpe / se)

    # Expected max Sharpe from N iid normal draws (Euler-Mascheroni approx)
    euler_mascheroni = 0.5772156649
    e_max_sharpe = (
        math.sqrt(variance_of_sharpes)
        * ((1 - euler_mascheroni) * _norm_ppf(1 - 1.0 / num_trials)
           + euler_mascheroni * _norm_ppf(1 - 1.0 / (num_trials * math.e)))
    )

    # Standard error of the annualized Sharpe (with skew/kurtosis correction)
    se = math.sqrt(
        (1 + 0.5 * sr_daily ** 2
         - skewness * sr_daily
         + ((kurtosis - 3) / 4) * sr_daily ** 2)
        / (n_returns - 1)
    ) * math.sqrt(252)

    if se <= 0:
        return 0.0

    z = (observed_sharpe - e_max_sharpe) / se
    return _norm_cdf(z)


def estimate_sharpe_variance(returns: np.ndarray, n_bootstraps: int = 500,
                             seed: int = 42) -> float:
    """Estimate variance of the annualized Sharpe ratio via the Lo (2002) formula.

    For the DSR, we need the variance of annualized Sharpe ratios that
    would be observed across independent strategy trials. Under H0 (no
    skill), all strategies have true SR=0, and the observed variance comes
    from sampling error. The Lo (2002) formula gives:

        Var(SR_daily) = (1 + 0.5*SR_d^2 - skew*SR_d + ((k-3)/4)*SR_d^2) / (T-1)
        Var(SR_annual) = Var(SR_daily) * 252

    The n_bootstraps and seed parameters are retained for API compatibility
    but are no longer used (the analytical formula replaced bootstrap).
    """
    if len(returns) < 10:
        return 1.0

    n = len(returns)
    std = returns.std()
    if std == 0:
        return 1.0

    sr_daily = returns.mean() / std
    skew = float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0
    kurt = float(pd.Series(returns).kurtosis()) + 3.0 if len(returns) > 3 else 3.0

    # Lo (2002) variance of annualized Sharpe ratio
    var_sr_daily = (
        1 + 0.5 * sr_daily ** 2
        - skew * sr_daily
        + ((kurt - 3) / 4) * sr_daily ** 2
    ) / (n - 1)
    var_sr_annual = var_sr_daily * 252

    return max(float(var_sr_annual), 1e-10)


def permutation_test(
    equity_series: pd.Series,
    n_permutations: int = 1000,
    seed: int = 42,
    benchmark_series: pd.Series | None = None,
) -> dict:
    """Sign-flip permutation test for strategy significance.

    Tests whether the strategy generates returns significantly different
    from zero (or significantly above benchmark when provided).

    The previous implementation shuffled daily returns, but Sharpe ratio
    (mean/std * sqrt(252)) is invariant to shuffling because both mean
    and std are preserved. This fix uses a sign-flip test: under the null
    hypothesis of no edge, daily returns should be symmetric around zero.
    Randomly flipping signs changes the mean while preserving std,
    producing a proper null distribution.

    When a benchmark series is provided, tests excess returns
    (strategy - benchmark) instead of raw returns.

    Args:
        equity_series: Daily equity curve.
        n_permutations: Number of sign-flip iterations.
        seed: Random seed for reproducibility.
        benchmark_series: Optional benchmark equity curve for excess
            return testing.

    Returns:
        Dict with 'observed_sharpe', 'p_value', 'null_sharpes'.
    """
    strategy_returns = equity_series.pct_change().dropna()

    if len(strategy_returns) < 2:
        return {"observed_sharpe": 0.0, "p_value": 1.0, "null_sharpes": []}

    if benchmark_series is not None:
        # Test excess returns (alpha) over benchmark
        benchmark_returns = benchmark_series.pct_change().dropna()
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 2:
            return {"observed_sharpe": 0.0, "p_value": 1.0, "null_sharpes": []}
        returns = (strategy_returns.loc[common_idx].values
                   - benchmark_returns.loc[common_idx].values)
    else:
        returns = strategy_returns.values

    observed = _sharpe_from_returns(returns)

    # Sign-flip test: under H0 (no edge / no alpha), returns are
    # symmetric around zero. Randomly flip signs to generate null.
    rng = np.random.default_rng(seed)
    null_sharpes = []
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(returns))
        null_sharpes.append(_sharpe_from_returns(returns * signs))

    null_sharpes = np.array(null_sharpes)
    p_value = float((null_sharpes >= observed).mean())

    return {
        "observed_sharpe": observed,
        "p_value": p_value,
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
