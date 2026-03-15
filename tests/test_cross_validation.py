"""Tests for purged K-Fold cross-validation."""

from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig
from backtester.research.cross_validation import purged_kfold_cv
from backtester.research.optimizer import OptimizationResult


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
    result = MagicMock()
    dates = pd.bdate_range("2020-01-02", periods=252)
    equity_vals = np.linspace(100_000, 100_000 * (1 + cagr), 252)
    result.equity_series = pd.Series(
        equity_vals, index=pd.DatetimeIndex(dates.date, name="Date")
    )
    result.trades = []
    return result


def _mock_opt_result(best_params=None, best_metric=1.5):
    if best_params is None:
        best_params = {"sma_fast": 100}
    return OptimizationResult(
        results_table=pd.DataFrame(),
        best_params=best_params,
        best_metric_value=best_metric,
        optimize_metric="sharpe_ratio",
    )


class TestPurgedKFoldCV:
    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_returns_correct_structure(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result()
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=3,
        )

        assert "folds" in result
        assert "n_splits" in result
        assert result["n_splits"] == 3
        assert "purge_days" in result
        assert "embargo_days" in result
        assert "avg_train_sharpe" in result
        assert "avg_test_sharpe" in result

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_number_of_folds(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result()
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=5,
        )

        assert len(result["folds"]) == 5

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_fold_results_contain_expected_keys(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result()
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=3,
        )

        for fold in result["folds"]:
            assert "fold" in fold
            assert "test_start" in fold
            assert "test_end" in fold
            assert "best_params" in fold
            assert "train_sharpe" in fold
            assert "test_sharpe" in fold
            assert "test_cagr" in fold
            assert "test_max_dd" in fold

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_average_sharpe_computed(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result(best_metric=2.0)
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=3,
        )

        assert result["avg_train_sharpe"] == pytest.approx(2.0)
        assert isinstance(result["avg_test_sharpe"], float)

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_purge_and_embargo_days_stored(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result()
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=3,
            purge_days=15,
            embargo_days=10,
        )

        assert result["purge_days"] == 15
        assert result["embargo_days"] == 10

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_grid_search_called_with_train_config(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result()
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()
        grid = {"sma_fast": [50, 100]}

        purged_kfold_cv(config, param_grid=grid, n_splits=3)

        # grid_search should have been called n_splits times
        assert mock_gs.call_count == 3
        # Each call should pass the param grid and optimize metric
        for call in mock_gs.call_args_list:
            # Second positional arg is param_grid
            assert call[0][1] == grid
            assert call[1].get("optimize_metric") == "sharpe_ratio"

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_train_failure_skips_fold(self, mock_gs, MockEngine):
        mock_gs.side_effect = RuntimeError("Train optimization failed")
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=3,
        )

        # All folds should fail, so no results
        assert len(result["folds"]) == 0
        assert result["avg_train_sharpe"] == 0.0
        assert result["avg_test_sharpe"] == 0.0

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_test_failure_records_zero_sharpe(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result()
        MockEngine.return_value.run.side_effect = RuntimeError("Test failed")
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=3,
        )

        for fold in result["folds"]:
            assert fold["test_sharpe"] == 0.0

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_empty_best_params_skips_fold(self, mock_gs, MockEngine):
        mock_gs.return_value = OptimizationResult(
            results_table=pd.DataFrame(),
            best_params={},
            best_metric_value=0.0,
            optimize_metric="sharpe_ratio",
        )
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=3,
        )

        assert len(result["folds"]) == 0

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_two_splits(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result()
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        result = purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 100]},
            n_splits=2,
        )

        assert len(result["folds"]) == 2
        assert result["folds"][0]["fold"] == 1
        assert result["folds"][1]["fold"] == 2

    @patch("backtester.research.cross_validation.BacktestEngine")
    @patch("backtester.research.cross_validation.grid_search")
    def test_best_params_passed_to_test(self, mock_gs, MockEngine):
        mock_gs.return_value = _mock_opt_result(best_params={"sma_fast": 75})
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        purged_kfold_cv(
            config,
            param_grid={"sma_fast": [50, 75, 100]},
            n_splits=2,
        )

        # BacktestEngine should be called with strategy_params including sma_fast=75
        for call in MockEngine.call_args_list:
            test_config = call[0][0]
            assert test_config.strategy_params["sma_fast"] == 75
