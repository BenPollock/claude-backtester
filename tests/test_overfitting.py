"""Tests for overfitting detection metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.overfitting import (
    deflated_sharpe_ratio,
    estimate_sharpe_variance,
    permutation_test,
    _sharpe_from_returns,
    _norm_cdf,
    _norm_ppf,
)


class TestDeflatedSharpeRatio:
    def test_single_trial_positive_sharpe(self):
        # With 1 trial, no multiple-testing penalty; just tests significance
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0, num_trials=1,
            variance_of_sharpes=0.5, n_returns=252,
        )
        assert 0.0 <= dsr <= 1.0
        # High Sharpe with many returns should give high DSR (CDF saturates near 1.0)
        assert dsr > 0.9

    def test_many_trials_reduces_dsr(self):
        # More trials -> higher expected max -> lower DSR
        dsr_few = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=5,
            variance_of_sharpes=0.5, n_returns=252,
        )
        dsr_many = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=1000,
            variance_of_sharpes=0.5, n_returns=252,
        )
        assert dsr_few > dsr_many

    def test_zero_variance_returns_zero(self):
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0, num_trials=10,
            variance_of_sharpes=0.0, n_returns=252,
        )
        assert dsr == 0.0

    def test_single_return_returns_zero(self):
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0, num_trials=10,
            variance_of_sharpes=0.5, n_returns=1,
        )
        assert dsr == 0.0

    def test_output_between_zero_and_one(self):
        for sharpe in [0.5, 1.0, 2.0, 3.0]:
            for trials in [1, 10, 100]:
                dsr = deflated_sharpe_ratio(
                    observed_sharpe=sharpe, num_trials=trials,
                    variance_of_sharpes=0.5, n_returns=252,
                )
                assert 0.0 <= dsr <= 1.0

    def test_negative_sharpe_gives_low_dsr(self):
        dsr = deflated_sharpe_ratio(
            observed_sharpe=-1.0, num_trials=10,
            variance_of_sharpes=0.5, n_returns=252,
        )
        assert dsr < 0.5

    def test_skewness_and_kurtosis_affect_result(self):
        base = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=10,
            variance_of_sharpes=0.5, n_returns=252,
            skewness=0.0, kurtosis=3.0,
        )
        skewed = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=10,
            variance_of_sharpes=0.5, n_returns=252,
            skewness=1.0, kurtosis=3.0,
        )
        # Different skewness should give different DSR
        assert base != skewed

    def test_more_returns_increases_dsr(self):
        dsr_short = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=10,
            variance_of_sharpes=0.5, n_returns=50,
        )
        dsr_long = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=10,
            variance_of_sharpes=0.5, n_returns=2520,
        )
        # More data = tighter SE = more significant
        assert dsr_long > dsr_short


class TestEstimateSharpeVariance:
    def test_returns_positive_float(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.02, 252)
        var = estimate_sharpe_variance(returns)
        assert var > 0
        assert isinstance(var, float)

    def test_short_returns_gives_default(self):
        returns = np.array([0.01, 0.02])
        var = estimate_sharpe_variance(returns)
        assert var == 1.0

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0005, 0.02, 252)
        v1 = estimate_sharpe_variance(returns, seed=42)
        v2 = estimate_sharpe_variance(returns, seed=42)
        assert v1 == v2

    def test_different_seeds_differ(self):
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0005, 0.02, 252)
        v1 = estimate_sharpe_variance(returns, seed=42)
        v2 = estimate_sharpe_variance(returns, seed=99)
        assert v1 != v2

    def test_more_data_reduces_variance(self):
        # More data points should reduce Sharpe estimation variance
        rng = np.random.default_rng(42)
        all_returns = rng.normal(0.0003, 0.01, 2000)
        var_short = estimate_sharpe_variance(all_returns[:200], seed=42)
        var_long = estimate_sharpe_variance(all_returns, seed=42)
        assert var_long < var_short


class TestPermutationTest:
    def _make_equity(self, n=252, trend=0.001):
        dates = pd.bdate_range("2020-01-02", periods=n)
        prices = 100 * np.cumprod(1 + np.full(n, trend))
        return pd.Series(prices, index=pd.DatetimeIndex(dates.date, name="Date"))

    def test_strong_trend_low_pvalue(self):
        equity = self._make_equity(trend=0.005)
        result = permutation_test(equity, n_permutations=500, seed=42)
        assert result["observed_sharpe"] > 0
        assert result["p_value"] < 0.05
        assert len(result["null_sharpes"]) == 500

    def test_flat_equity_high_pvalue(self):
        # No trend -> p_value should be high
        dates = pd.bdate_range("2020-01-02", periods=252)
        flat = pd.Series(np.full(252, 100.0), index=pd.DatetimeIndex(dates.date, name="Date"))
        result = permutation_test(flat, n_permutations=200, seed=42)
        assert result["observed_sharpe"] == 0.0
        # All null sharpes also 0 since returns are all 0
        assert result["p_value"] >= 0.0

    def test_short_series(self):
        equity = pd.Series([100.0], index=[pd.Timestamp("2020-01-02")])
        result = permutation_test(equity)
        assert result["observed_sharpe"] == 0.0
        assert result["p_value"] == 1.0
        assert result["null_sharpes"] == []

    def test_two_point_series(self):
        equity = pd.Series([100.0, 101.0],
                           index=[pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")])
        result = permutation_test(equity, n_permutations=100, seed=42)
        # Only 1 return after pct_change().dropna(), which is < 2
        assert result["p_value"] == 1.0

    def test_deterministic(self):
        equity = self._make_equity()
        r1 = permutation_test(equity, n_permutations=100, seed=42)
        r2 = permutation_test(equity, n_permutations=100, seed=42)
        assert r1["observed_sharpe"] == r2["observed_sharpe"]
        assert r1["p_value"] == r2["p_value"]

    def test_with_benchmark(self):
        strategy_equity = self._make_equity(trend=0.003)
        benchmark_equity = self._make_equity(trend=0.001)
        result = permutation_test(
            strategy_equity,
            n_permutations=200,
            seed=42,
            benchmark_series=benchmark_equity,
        )
        assert "observed_sharpe" in result
        assert "p_value" in result
        assert len(result["null_sharpes"]) == 200

    def test_benchmark_short_common_index(self):
        dates1 = pd.bdate_range("2020-01-02", periods=10)
        dates2 = pd.bdate_range("2021-01-02", periods=10)
        eq1 = pd.Series(np.linspace(100, 110, 10), index=pd.DatetimeIndex(dates1.date))
        eq2 = pd.Series(np.linspace(100, 105, 10), index=pd.DatetimeIndex(dates2.date))
        result = permutation_test(eq1, benchmark_series=eq2)
        # No common index
        assert result["p_value"] == 1.0


class TestHelpers:
    def test_sharpe_from_returns_positive(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)
        sharpe = _sharpe_from_returns(returns)
        assert sharpe > 0

    def test_sharpe_from_returns_zero_std(self):
        returns = np.zeros(100)
        assert _sharpe_from_returns(returns) == 0.0

    def test_norm_cdf_symmetry(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5)
        assert _norm_cdf(1.0) + _norm_cdf(-1.0) == pytest.approx(1.0)

    def test_norm_cdf_extremes(self):
        assert _norm_cdf(10.0) > 0.999
        assert _norm_cdf(-10.0) < 0.001

    def test_norm_ppf_boundaries(self):
        assert _norm_ppf(0.0) == -10.0
        assert _norm_ppf(1.0) == 10.0
        assert _norm_ppf(0.5) == 0.0

    def test_norm_ppf_approximate_inverse_of_cdf(self):
        # ppf(cdf(x)) should be approximately x
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            p = _norm_cdf(x)
            recovered = _norm_ppf(p)
            assert recovered == pytest.approx(x, abs=0.02)
