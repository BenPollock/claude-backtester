"""Tests for stress testing scenarios."""

from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.stress import (
    STRESS_SCENARIOS,
    run_stress_test,
    print_stress_results,
)
from backtester.config import BacktestConfig


def _base_config():
    return BacktestConfig(
        strategy_name="sma_crossover",
        tickers=["SPY"],
        benchmark="SPY",
        start_date=date(2005, 1, 3),
        end_date=date(2010, 12, 31),
        starting_cash=100_000.0,
        max_positions=10,
        max_alloc_pct=0.10,
        strategy_params={"sma_fast": 50, "sma_slow": 200},
    )


def _mock_result():
    result = MagicMock()
    dates = pd.bdate_range("2020-01-02", periods=252)
    equity_vals = np.linspace(100_000, 110_000, 252)
    result.equity_series = pd.Series(
        equity_vals, index=pd.DatetimeIndex(dates.date, name="Date")
    )
    result.trades = []
    return result


class TestStressScenarios:
    def test_predefined_scenarios_exist(self):
        assert len(STRESS_SCENARIOS) == 7
        assert "gfc_2008" in STRESS_SCENARIOS
        assert "covid_crash" in STRESS_SCENARIOS
        assert "dot_com_crash" in STRESS_SCENARIOS

    def test_scenario_dates_are_valid(self):
        for name, (start, end) in STRESS_SCENARIOS.items():
            assert isinstance(start, date)
            assert isinstance(end, date)
            assert start < end, f"Scenario {name} has start >= end"


class TestRunStressTest:
    @patch("backtester.engine.BacktestEngine")
    def test_runs_all_scenarios_by_default(self, MockEngine):
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        results = run_stress_test(config)

        assert len(results) == len(STRESS_SCENARIOS)
        assert MockEngine.return_value.run.call_count == len(STRESS_SCENARIOS)

    @patch("backtester.engine.BacktestEngine")
    def test_runs_specific_scenarios(self, MockEngine):
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        results = run_stress_test(config, scenarios=["gfc_2008", "covid_crash"])

        assert len(results) == 2
        assert results[0]["scenario"] == "gfc_2008"
        assert results[1]["scenario"] == "covid_crash"

    @patch("backtester.engine.BacktestEngine")
    def test_unknown_scenario_skipped(self, MockEngine):
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        results = run_stress_test(config, scenarios=["nonexistent", "gfc_2008"])

        assert len(results) == 1
        assert results[0]["scenario"] == "gfc_2008"

    @patch("backtester.engine.BacktestEngine")
    def test_result_contains_metrics(self, MockEngine):
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        results = run_stress_test(config, scenarios=["gfc_2008"])

        r = results[0]
        assert r["scenario"] == "gfc_2008"
        assert r["start"] == date(2007, 10, 1)
        assert r["end"] == date(2009, 3, 31)
        assert "total_return" in r
        assert "cagr" in r
        assert "sharpe_ratio" in r
        assert "max_drawdown" in r

    @patch("backtester.engine.BacktestEngine")
    def test_scenario_dates_applied_to_config(self, MockEngine):
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()

        run_stress_test(config, scenarios=["covid_crash"])

        # Verify engine was created with scenario dates
        call_args = MockEngine.call_args[0][0]
        assert call_args.start_date == date(2020, 2, 1)
        assert call_args.end_date == date(2020, 4, 30)

    @patch("backtester.engine.BacktestEngine")
    def test_engine_failure_captured_as_error(self, MockEngine):
        MockEngine.return_value.run.side_effect = RuntimeError("Data not available")
        config = _base_config()

        results = run_stress_test(config, scenarios=["gfc_2008"])

        assert len(results) == 1
        assert "error" in results[0]
        assert "Data not available" in results[0]["error"]

    @patch("backtester.engine.BacktestEngine")
    def test_empty_scenarios_list(self, MockEngine):
        config = _base_config()

        results = run_stress_test(config, scenarios=[])

        assert results == []
        assert MockEngine.return_value.run.call_count == 0

    @patch("backtester.engine.BacktestEngine")
    def test_base_config_not_mutated(self, MockEngine):
        MockEngine.return_value.run.return_value = _mock_result()
        config = _base_config()
        original_start = config.start_date
        original_end = config.end_date

        run_stress_test(config, scenarios=["gfc_2008"])

        assert config.start_date == original_start
        assert config.end_date == original_end


class TestPrintStressResults:
    def test_prints_normal_results(self, capsys):
        results = [
            {
                "scenario": "gfc_2008",
                "start": date(2007, 10, 1),
                "end": date(2009, 3, 31),
                "total_return": -0.35,
                "max_drawdown": -0.50,
                "sharpe_ratio": -0.8,
            }
        ]
        print_stress_results(results)
        captured = capsys.readouterr()
        assert "STRESS TEST RESULTS" in captured.out
        assert "gfc_2008" in captured.out

    def test_prints_error_results(self, capsys):
        results = [
            {
                "scenario": "gfc_2008",
                "start": date(2007, 10, 1),
                "end": date(2009, 3, 31),
                "error": "Data not available",
            }
        ]
        print_stress_results(results)
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_empty_results(self, capsys):
        print_stress_results([])
        captured = capsys.readouterr()
        assert "STRESS TEST RESULTS" in captured.out
