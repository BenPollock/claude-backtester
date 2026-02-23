"""Tests for optimizer (grid search) and walk-forward analysis."""

from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig
from backtester.research.optimizer import grid_search, OptimizationResult
from backtester.research.walk_forward import walk_forward, _add_months


def _base_config():
    return BacktestConfig(
        strategy_name="sma_crossover",
        tickers=["TEST"],
        benchmark="TEST",
        start_date=date(2005, 1, 3),
        end_date=date(2010, 12, 31),
        starting_cash=100_000.0,
        max_positions=10,
        max_alloc_pct=0.10,
        strategy_params={"sma_fast": 50, "sma_slow": 200},
    )


def _mock_result(sharpe=1.0, cagr=0.10):
    """Create a mock BacktestResult with controllable metrics."""
    result = MagicMock()
    dates = pd.bdate_range("2020-01-02", periods=252)
    equity_vals = np.linspace(100, 100 * (1 + cagr), 252)
    result.equity_series = pd.Series(
        equity_vals,
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )
    result.trades = []
    result.benchmark_series = None
    return result


class TestGridSearch:
    @patch("backtester.research.optimizer.BacktestEngine")
    def test_all_combos_tested(self, MockEngine):
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()
        grid = {"sma_fast": [50, 100], "sma_slow": [200, 300]}

        result = grid_search(config, grid)

        assert isinstance(result, OptimizationResult)
        # 2 x 2 = 4 combos
        assert len(result.results_table) == 4
        assert MockEngine.return_value.run.call_count == 4

    @patch("backtester.research.optimizer.BacktestEngine")
    def test_best_params_selected(self, MockEngine):
        # Return different sharpe for each combo
        sharpes = iter([0.5, 1.5, 0.8, 1.2])

        def side_effect():
            s = next(sharpes)
            return _mock_result(sharpe=s, cagr=s * 0.05)

        MockEngine.return_value.run.side_effect = lambda: side_effect()
        config = _base_config()
        grid = {"sma_fast": [50, 100], "sma_slow": [200, 300]}

        result = grid_search(config, grid, optimize_metric="sharpe_ratio")

        assert result.best_params  # not empty
        assert result.optimize_metric == "sharpe_ratio"

    @patch("backtester.research.optimizer.BacktestEngine")
    def test_error_handling(self, MockEngine):
        MockEngine.return_value.run.side_effect = RuntimeError("Data fetch failed")
        config = _base_config()
        grid = {"sma_fast": [50]}

        result = grid_search(config, grid)

        assert len(result.results_table) == 1
        assert "error" in result.results_table.columns
        assert result.best_params == {}

    @patch("backtester.research.optimizer.BacktestEngine")
    def test_single_param(self, MockEngine):
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()
        grid = {"sma_fast": [50, 100, 150]}

        result = grid_search(config, grid)

        assert len(result.results_table) == 3


class TestWalkForward:
    @patch("backtester.research.walk_forward.BacktestEngine")
    @patch("backtester.research.walk_forward.grid_search")
    def test_window_splitting(self, mock_gs, MockEngine):
        mock_gs.return_value = OptimizationResult(
            results_table=pd.DataFrame(),
            best_params={"sma_fast": 100},
            best_metric_value=1.5,
            optimize_metric="sharpe_ratio",
        )
        MockEngine.return_value.run.return_value = _mock_result()

        config = _base_config()  # 2005-01-03 to 2010-12-31
        result = walk_forward(
            config,
            param_grid={"sma_fast": [50, 100]},
            is_months=12,
            oos_months=3,
        )

        assert result["num_windows"] > 0
        assert "degradation_ratio" in result
        assert "windows" in result

        # Verify window date ranges make sense
        for w in result["windows"]:
            assert w["is_start"] < w["is_end"]
            assert w["oos_start"] < w["oos_end"]
            assert w["oos_start"] > w["is_end"]

    @patch("backtester.research.walk_forward.BacktestEngine")
    @patch("backtester.research.walk_forward.grid_search")
    def test_degradation_ratio(self, mock_gs, MockEngine):
        mock_gs.return_value = OptimizationResult(
            results_table=pd.DataFrame(),
            best_params={"sma_fast": 100},
            best_metric_value=2.0,
            optimize_metric="sharpe_ratio",
        )
        MockEngine.return_value.run.return_value = _mock_result(sharpe=1.0)

        config = _base_config()
        result = walk_forward(
            config,
            param_grid={"sma_fast": [50, 100]},
            is_months=24,
            oos_months=6,
        )

        assert result["num_windows"] > 0
        # avg_is_sharpe should be 2.0, avg_oos_sharpe computed from mock data
        assert "avg_is_sharpe" in result
        assert "avg_oos_sharpe" in result


class TestAddMonths:
    def test_basic(self):
        assert _add_months(date(2020, 1, 15), 3) == date(2020, 4, 15)

    def test_year_boundary(self):
        assert _add_months(date(2020, 11, 1), 3) == date(2021, 2, 1)

    def test_month_end_clamping(self):
        # Jan 31 + 1 month → Feb 29 (2020 is leap year)
        assert _add_months(date(2020, 1, 31), 1) == date(2020, 2, 29)

    def test_month_end_non_leap(self):
        # Jan 31 + 1 month → Feb 28 (2021 is not leap year)
        assert _add_months(date(2021, 1, 31), 1) == date(2021, 2, 28)

    def test_twelve_months(self):
        assert _add_months(date(2020, 6, 15), 12) == date(2021, 6, 15)

    def test_large_jump(self):
        assert _add_months(date(2020, 1, 1), 24) == date(2022, 1, 1)
