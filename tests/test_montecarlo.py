"""Tests for Monte Carlo simulation module."""

import numpy as np
import pandas as pd

from backtester.analytics.montecarlo import run_monte_carlo, monte_carlo_percentiles


def make_equity(values, start="2020-01-02"):
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=pd.DatetimeIndex(dates.date, name="Date"), name="Equity")


class TestMonteCarlo:
    def test_output_shape(self):
        equity = make_equity(np.linspace(100, 120, 50))
        paths = run_monte_carlo(equity, n_simulations=100, seed=42)
        # Shape: (n_simulations, n_days-1) since pct_change drops first value
        assert paths.shape == (100, 49)

    def test_deterministic_with_seed(self):
        equity = make_equity(np.linspace(100, 120, 50))
        paths1 = run_monte_carlo(equity, n_simulations=50, seed=123)
        paths2 = run_monte_carlo(equity, n_simulations=50, seed=123)
        np.testing.assert_array_equal(paths1, paths2)

    def test_starts_from_initial_equity(self):
        equity = make_equity([200, 210, 220, 215, 225])
        paths = run_monte_carlo(equity, n_simulations=10, seed=42)
        # First column = start_value * (1 + sampled_return), so all paths
        # should be anchored near the starting equity of 200
        # The first step is start * (1+r), verify it's in a reasonable range
        assert np.all(paths[:, 0] > 0)
        # More specifically, the mean of first-step values should be close to
        # start_value * mean(1+return) ≈ 200 * ~1.02 = ~204
        assert 150 < paths[:, 0].mean() < 250

    def test_percentiles_keys_and_shape(self):
        equity = make_equity(np.linspace(100, 120, 50))
        paths = run_monte_carlo(equity, n_simulations=100, seed=42)
        result = monte_carlo_percentiles(paths)

        expected_keys = {"p5", "p25", "p50", "p75", "p95", "final_values"}
        assert set(result.keys()) == expected_keys

        # Percentile arrays should match path length (n_days)
        assert result["p50"].shape == (49,)
        # final_values should have one entry per simulation
        assert result["final_values"].shape == (100,)
