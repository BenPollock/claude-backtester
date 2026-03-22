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


class TestMonteCarloPercentileOrdering:
    """Percentile values should be monotonically ordered: p5 < p25 < p50 < p75 < p95."""

    def test_percentile_ordering_at_each_timestep(self):
        equity = make_equity(np.linspace(100, 150, 100))
        paths = run_monte_carlo(equity, n_simulations=5000, seed=42)
        result = monte_carlo_percentiles(paths)

        # At every timestep, lower percentiles should be <= higher percentiles
        for i in range(len(result["p5"])):
            assert result["p5"][i] <= result["p25"][i], f"p5 > p25 at step {i}"
            assert result["p25"][i] <= result["p50"][i], f"p25 > p50 at step {i}"
            assert result["p50"][i] <= result["p75"][i], f"p50 > p75 at step {i}"
            assert result["p75"][i] <= result["p95"][i], f"p75 > p95 at step {i}"



class TestMonteCarloEdgeCases:
    """Edge cases for Monte Carlo simulation."""

    def test_two_value_equity_series(self):
        """Minimum viable equity series: 2 values produce 1 return."""
        equity = make_equity([100, 110])
        paths = run_monte_carlo(equity, n_simulations=50, seed=42)
        # 1 return → paths shape is (n_simulations, 1)
        assert paths.shape == (50, 1)
        # All paths should use the single return (10%), so all values = 100 * 1.10 = 110
        np.testing.assert_allclose(paths[:, 0], 110.0)

    def test_constant_equity_series(self):
        """Flat equity (zero returns) should produce flat paths."""
        equity = make_equity([100, 100, 100, 100, 100])
        paths = run_monte_carlo(equity, n_simulations=20, seed=42)
        # All returns are 0%, so all paths should stay at 100
        np.testing.assert_allclose(paths, 100.0)

    def test_large_n_simulations_mean_close_to_original(self):
        """With many simulations, the mean final value should approximate
        the original final value (bootstrap preserves return distribution)."""
        equity = make_equity(np.linspace(100, 120, 50))
        paths = run_monte_carlo(equity, n_simulations=10000, seed=42)
        result = monte_carlo_percentiles(paths)

        original_final = equity.iloc[-1]
        sim_mean_final = result["final_values"].mean()
        # The mean should be within 5% of the original final value
        assert abs(sim_mean_final - original_final) / original_final < 0.05, (
            f"Mean final {sim_mean_final:.2f} too far from original {original_final:.2f}"
        )

    def test_all_paths_positive(self):
        """Equity paths from bootstrap should remain positive (cumulative product
        of (1+r) factors where r > -1)."""
        equity = make_equity(np.linspace(100, 80, 50))  # declining equity
        paths = run_monte_carlo(equity, n_simulations=500, seed=42)
        assert np.all(paths > 0), "All simulated equity values should be positive"

    def test_custom_percentiles(self):
        """monte_carlo_percentiles accepts custom percentile list."""
        equity = make_equity(np.linspace(100, 120, 30))
        paths = run_monte_carlo(equity, n_simulations=100, seed=42)
        result = monte_carlo_percentiles(paths, percentiles=[10, 50, 90])

        assert "p10" in result
        assert "p50" in result
        assert "p90" in result
        assert "p5" not in result  # not requested
        assert "final_values" in result  # always included


# ---------------------------------------------------------------------------
# Coverage-expanding tests
# ---------------------------------------------------------------------------


class TestMonteCarloPercentileAccuracy:
    """Verify percentile values are numerically correct."""

    def test_p50_is_median(self):
        """p50 should be the median of paths at each timestep."""
        equity = make_equity(np.linspace(100, 130, 30))
        paths = run_monte_carlo(equity, n_simulations=200, seed=42)
        result = monte_carlo_percentiles(paths)

        # At each timestep, p50 should equal np.median
        for i in range(paths.shape[1]):
            expected_median = np.median(paths[:, i])
            np.testing.assert_allclose(
                result["p50"][i], expected_median,
                err_msg=f"p50 != median at step {i}"
            )

    def test_p5_is_5th_percentile(self):
        """p5 should equal the 5th percentile at each timestep."""
        equity = make_equity(np.linspace(100, 130, 30))
        paths = run_monte_carlo(equity, n_simulations=500, seed=42)
        result = monte_carlo_percentiles(paths)

        for i in range(paths.shape[1]):
            expected = np.percentile(paths[:, i], 5)
            np.testing.assert_allclose(
                result["p5"][i], expected,
                err_msg=f"p5 != 5th percentile at step {i}"
            )

    def test_final_values_match_last_column(self):
        """final_values should be the last column of paths."""
        equity = make_equity(np.linspace(100, 120, 20))
        paths = run_monte_carlo(equity, n_simulations=100, seed=42)
        result = monte_carlo_percentiles(paths)

        np.testing.assert_array_equal(result["final_values"], paths[:, -1])


class TestMonteCarloSingleSimulation:
    """Monte Carlo with n_simulations=1."""

    def test_single_simulation_shape(self):
        equity = make_equity(np.linspace(100, 120, 10))
        paths = run_monte_carlo(equity, n_simulations=1, seed=42)
        assert paths.shape == (1, 9)

    def test_single_simulation_percentiles(self):
        """All percentiles should be the same for a single simulation."""
        equity = make_equity(np.linspace(100, 120, 10))
        paths = run_monte_carlo(equity, n_simulations=1, seed=42)
        result = monte_carlo_percentiles(paths)
        np.testing.assert_array_equal(result["p5"], result["p95"])


class TestMonteCarloNoSeed:
    """Monte Carlo without seed should still work (non-deterministic)."""

    def test_no_seed_produces_valid_paths(self):
        equity = make_equity(np.linspace(100, 120, 50))
        paths = run_monte_carlo(equity, n_simulations=50, seed=None)
        assert paths.shape == (50, 49)
        assert np.all(paths > 0)
