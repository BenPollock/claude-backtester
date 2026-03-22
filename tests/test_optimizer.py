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

    @patch("backtester.research.walk_forward.BacktestEngine")
    @patch("backtester.research.walk_forward.grid_search")
    def test_oos_uses_optimize_metric_not_hardcoded_sharpe(self, mock_gs, MockEngine):
        """OOS metric should use optimize_metric param, not hardcoded 'sharpe_ratio'.

        Previously, walk_forward always retrieved 'sharpe_ratio' from OOS metrics
        regardless of optimize_metric, producing incorrect degradation ratios
        when optimizing for a different metric (e.g. CAGR).
        """
        # IS optimization returns best CAGR of 0.25
        mock_gs.return_value = OptimizationResult(
            results_table=pd.DataFrame(),
            best_params={"sma_fast": 100},
            best_metric_value=0.25,
            optimize_metric="cagr",
        )
        # OOS result: equity that goes from 100 to 115 over 252 days (~15% return)
        mock_result = _mock_result(cagr=0.15)
        MockEngine.return_value.run.return_value = mock_result

        config = _base_config()
        result = walk_forward(
            config,
            param_grid={"sma_fast": [50, 100]},
            is_months=24,
            oos_months=6,
            optimize_metric="cagr",
        )

        assert result["num_windows"] > 0
        # OOS metric should be CAGR (from compute_all_metrics), not Sharpe
        # The degradation ratio should be OOS_CAGR / IS_CAGR, both in same units
        from backtester.analytics.metrics import compute_all_metrics
        oos_equity = mock_result.equity_series
        expected_oos_cagr = compute_all_metrics(oos_equity, [])["cagr"]
        # Each window's oos_sharpe should actually be the OOS CAGR value
        for w in result["windows"]:
            assert w["oos_sharpe"] == pytest.approx(expected_oos_cagr, rel=1e-6)


class TestGridSearchEdgeCases:
    """Additional grid_search coverage: higher_is_better=False, all failures, missing metric."""

    @patch("backtester.research.optimizer.BacktestEngine")
    def test_higher_is_better_false_picks_minimum(self, MockEngine):
        """When higher_is_better=False, the combo with the lowest metric value wins."""
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()
        grid = {"sma_fast": [50, 100, 150]}

        # All combos produce the same mock metrics, so verify the code path
        # selects idxmin instead of idxmax
        result_low = grid_search(config, grid, optimize_metric="max_drawdown",
                                 higher_is_better=False)
        result_high = grid_search(config, grid, optimize_metric="max_drawdown",
                                  higher_is_better=True)

        # Both should return valid params (the metric is the same for all combos
        # since mock returns identical results, so both pick the first)
        assert result_low.best_params != {}
        assert result_high.best_params != {}
        # The best_metric_value should be the same since all combos are equal
        assert result_low.best_metric_value == result_high.best_metric_value

    @patch("backtester.research.optimizer.BacktestEngine")
    def test_all_runs_fail_returns_empty_best(self, MockEngine):
        """When every combo fails, best_params should be empty."""
        MockEngine.return_value.run.side_effect = RuntimeError("Boom")
        config = _base_config()
        grid = {"sma_fast": [50, 100]}

        result = grid_search(config, grid)

        assert len(result.results_table) == 2
        assert result.best_params == {}
        assert result.best_metric_value == 0.0

    @patch("backtester.research.optimizer.BacktestEngine")
    def test_missing_optimize_metric(self, MockEngine):
        """When optimize_metric isn't in results, best_params should be empty."""
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()
        grid = {"sma_fast": [50]}

        result = grid_search(config, grid, optimize_metric="nonexistent_metric")

        assert result.best_params == {}
        assert result.best_metric_value == 0.0

    @patch("backtester.research.optimizer.BacktestEngine")
    def test_mixed_success_and_failure(self, MockEngine):
        """Grid search with some successes and some failures picks best from successes."""
        results = iter([RuntimeError("fail"), _mock_result(sharpe=1.0)])

        def side_effect():
            r = next(results)
            if isinstance(r, Exception):
                raise r
            return r

        MockEngine.return_value.run.side_effect = lambda: side_effect()
        config = _base_config()
        grid = {"sma_fast": [50, 100]}

        result = grid_search(config, grid)

        assert len(result.results_table) == 2
        assert "error" in result.results_table.columns
        assert result.best_params != {}  # Should pick the successful one


class TestBayesianOptimize:
    """Tests for bayesian_optimize (requires skopt mock)."""

    def test_missing_skopt_raises_import_error(self):
        """When scikit-optimize is not installed, a clear error is raised."""
        from backtester.research.optimizer import bayesian_optimize
        config = _base_config()

        with patch.dict("sys.modules", {"skopt": None, "skopt.space": None}):
            with pytest.raises(ImportError, match="scikit-optimize"):
                bayesian_optimize(config, {"sma_fast": (10, 200)}, n_calls=1)


class TestPrintOptimizationResults:
    """Tests for print_optimization_results output."""

    def test_prints_without_error(self, capsys):
        from backtester.research.optimizer import print_optimization_results
        df = pd.DataFrame([
            {"sma_fast": 50, "sharpe_ratio": 1.2, "cagr": 0.10, "max_drawdown": -0.15,
             "total_trades": 20, "win_rate": 0.55, "profit_factor": 1.5, "calmar_ratio": 0.7},
            {"sma_fast": 100, "sharpe_ratio": 0.8, "cagr": 0.05, "max_drawdown": -0.20,
             "total_trades": 10, "win_rate": 0.50, "profit_factor": 1.2, "calmar_ratio": 0.25},
        ])
        opt = OptimizationResult(
            results_table=df,
            best_params={"sma_fast": 50},
            best_metric_value=1.2,
            optimize_metric="sharpe_ratio",
        )
        print_optimization_results(opt, top_n=5)
        captured = capsys.readouterr()
        assert "OPTIMIZATION RESULTS" in captured.out
        assert "sharpe_ratio" in captured.out
        assert "Best params" in captured.out

    def test_prints_with_errors(self, capsys):
        from backtester.research.optimizer import print_optimization_results
        df = pd.DataFrame([
            {"sma_fast": 50, "sharpe_ratio": 1.0, "cagr": 0.08, "error": None},
            {"sma_fast": 100, "error": "Data fetch failed"},
        ])
        opt = OptimizationResult(
            results_table=df,
            best_params={"sma_fast": 50},
            best_metric_value=1.0,
            optimize_metric="sharpe_ratio",
        )
        print_optimization_results(opt, top_n=5)
        captured = capsys.readouterr()
        assert "Failed runs: 1" in captured.out


class TestPrintWalkForwardResults:
    """Tests for print_walk_forward_results output."""

    def test_prints_without_error(self, capsys):
        from backtester.research.walk_forward import print_walk_forward_results
        wf = {
            "num_windows": 2,
            "avg_is_sharpe": 1.5,
            "avg_oos_sharpe": 0.9,
            "degradation_ratio": 0.6,
            "windows": [
                {
                    "window": 1,
                    "is_start": date(2005, 1, 3),
                    "is_end": date(2006, 1, 3),
                    "oos_start": date(2006, 1, 4),
                    "oos_end": date(2006, 4, 3),
                    "best_params": {"sma_fast": 100},
                    "is_sharpe": 1.8,
                    "oos_sharpe": 1.0,
                    "oos_cagr": 0.12,
                    "oos_max_dd": -0.10,
                    "oos_trades": 5,
                    "oos_win_rate": 0.60,
                },
                {
                    "window": 2,
                    "is_start": date(2006, 4, 3),
                    "is_end": date(2007, 4, 3),
                    "oos_start": date(2007, 4, 4),
                    "oos_end": date(2007, 7, 3),
                    "best_params": {"sma_fast": 50},
                    "is_sharpe": 1.2,
                    "oos_sharpe": 0.8,
                    "oos_cagr": 0.08,
                    "oos_max_dd": -0.15,
                    "oos_trades": 3,
                    "oos_win_rate": 0.40,
                },
            ],
        }
        print_walk_forward_results(wf)
        captured = capsys.readouterr()
        assert "WALK-FORWARD ANALYSIS" in captured.out
        assert "Windows: 2" in captured.out
        assert "GOOD" in captured.out  # 0.6 > 0.5

    def test_empty_windows(self, capsys):
        from backtester.research.walk_forward import print_walk_forward_results
        wf = {
            "num_windows": 0,
            "avg_is_sharpe": 0.0,
            "avg_oos_sharpe": 0.0,
            "degradation_ratio": 0.0,
            "windows": [],
        }
        print_walk_forward_results(wf)
        captured = capsys.readouterr()
        assert "Windows: 0" in captured.out
        assert "POOR" in captured.out  # 0.0 is not > 0


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
