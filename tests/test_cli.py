"""Tests for the CLI layer using Click's CliRunner."""

import json
import tempfile
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from click.testing import CliRunner

from backtester.cli import cli, _build_config
from backtester.config import BacktestConfig, RegimeFilter, StopConfig
from backtester.engine import BacktestEngine
from backtester.result import BacktestResult
from backtester.portfolio.portfolio import Portfolio
from tests.conftest import make_price_df, MockDataSource


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_result():
    """Create a minimal BacktestResult for mocking engine.run()."""
    portfolio = Portfolio(cash=100_000.0)
    # Record a couple equity snapshots so equity_series works
    portfolio.record_equity(date(2020, 1, 2))
    portfolio.record_equity(date(2020, 1, 3))
    benchmark_eq = [(date(2020, 1, 2), 100_000.0), (date(2020, 1, 3), 100_050.0)]
    return BacktestResult(
        config=BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["SPY"],
            benchmark="SPY",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
        ),
        portfolio=portfolio,
        benchmark_equity=benchmark_eq,
    )


class TestListStrategies:
    """Tests for the list-strategies command."""

    def test_lists_registered_strategies(self, runner):
        """list-strategies should show names of all registered strategies."""
        result = runner.invoke(cli, ["list-strategies"])
        assert result.exit_code == 0
        assert "sma_crossover" in result.output

    def test_output_format(self, runner):
        """Output should have a header and bullet-style listing."""
        result = runner.invoke(cli, ["list-strategies"])
        assert result.exit_code == 0
        assert "Available strategies:" in result.output
        # Each strategy is listed with a "  - " prefix
        lines = result.output.strip().split("\n")
        strategy_lines = [l for l in lines if l.startswith("  - ")]
        assert len(strategy_lines) >= 1


class TestBuildConfig:
    """Tests for the _build_config helper function."""

    def test_basic_config_construction(self):
        """_build_config should produce a valid BacktestConfig."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers="SPY,QQQ",
            market="us_ca",
            universe="index",
            benchmark="SPY",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=50_000.0,
            max_positions=5,
            max_alloc=0.20,
            fee=1.0,
            slippage_bps=5.0,
            params='{"sma_fast": 50, "sma_slow": 200}',
            cache_dir="/tmp/test-cache",
        )
        assert isinstance(config, BacktestConfig)
        assert config.strategy_name == "sma_crossover"
        assert config.tickers == ["SPY", "QQQ"]
        assert config.benchmark == "SPY"
        assert config.start_date == date(2020, 1, 2)
        assert config.end_date == date(2020, 12, 31)
        assert config.starting_cash == 50_000.0
        assert config.max_positions == 5
        assert config.max_alloc_pct == 0.20
        assert config.fee_per_trade == 1.0
        assert config.slippage_bps == 5.0
        assert config.strategy_params == {"sma_fast": 50, "sma_slow": 200}
        assert config.data_cache_dir == "/tmp/test-cache"

    def test_tickers_uppercased_and_stripped(self):
        """Tickers should be uppercased and whitespace-stripped."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers=" spy , qqq , aapl ",
            market="us_ca",
            universe="index",
            benchmark="spy",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=10_000.0,
            max_positions=10,
            max_alloc=0.10,
            fee=0.05,
            slippage_bps=10.0,
            params="{}",
            cache_dir="/tmp/test-cache",
        )
        assert config.tickers == ["SPY", "QQQ", "AAPL"]
        assert config.benchmark == "SPY"

    def test_regime_filter_constructed(self):
        """When regime-benchmark is set, a RegimeFilter should be created."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers="SPY",
            market="us_ca",
            universe="index",
            benchmark="SPY",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=10_000.0,
            max_positions=10,
            max_alloc=0.10,
            fee=0.05,
            slippage_bps=10.0,
            params="{}",
            cache_dir="/tmp/test-cache",
            regime_benchmark="SPY",
            regime_fast=50,
            regime_slow=150,
        )
        assert config.regime_filter is not None
        assert isinstance(config.regime_filter, RegimeFilter)
        assert config.regime_filter.benchmark == "SPY"
        assert config.regime_filter.fast_period == 50
        assert config.regime_filter.slow_period == 150

    def test_no_regime_filter_by_default(self):
        """Without regime-benchmark, regime_filter should be None."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers="SPY",
            market="us_ca",
            universe="index",
            benchmark="SPY",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=10_000.0,
            max_positions=10,
            max_alloc=0.10,
            fee=0.05,
            slippage_bps=10.0,
            params="{}",
            cache_dir="/tmp/test-cache",
        )
        assert config.regime_filter is None

    def test_stop_config_constructed(self):
        """When stop params are provided, a StopConfig should be created."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers="SPY",
            market="us_ca",
            universe="index",
            benchmark="SPY",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=10_000.0,
            max_positions=10,
            max_alloc=0.10,
            fee=0.05,
            slippage_bps=10.0,
            params="{}",
            cache_dir="/tmp/test-cache",
            stop_loss=0.05,
            take_profit=0.20,
            trailing_stop=0.08,
        )
        assert config.stop_config is not None
        assert isinstance(config.stop_config, StopConfig)
        assert config.stop_config.stop_loss_pct == 0.05
        assert config.stop_config.take_profit_pct == 0.20
        assert config.stop_config.trailing_stop_pct == 0.08

    def test_stop_config_atr_multiples(self):
        """ATR-based stop params should populate stop_config."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers="SPY",
            market="us_ca",
            universe="index",
            benchmark="SPY",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=10_000.0,
            max_positions=10,
            max_alloc=0.10,
            fee=0.05,
            slippage_bps=10.0,
            params="{}",
            cache_dir="/tmp/test-cache",
            stop_loss_atr=2.0,
            take_profit_atr=3.0,
        )
        assert config.stop_config is not None
        assert config.stop_config.stop_loss_atr == 2.0
        assert config.stop_config.take_profit_atr == 3.0

    def test_no_stop_config_by_default(self):
        """Without any stop params, stop_config should be None."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers="SPY",
            market="us_ca",
            universe="index",
            benchmark="SPY",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=10_000.0,
            max_positions=10,
            max_alloc=0.10,
            fee=0.05,
            slippage_bps=10.0,
            params="{}",
            cache_dir="/tmp/test-cache",
        )
        assert config.stop_config is None

    def test_position_sizing_params(self):
        """Position sizing params should be passed through to config."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers="SPY",
            market="us_ca",
            universe="index",
            benchmark="SPY",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=10_000.0,
            max_positions=10,
            max_alloc=0.10,
            fee=0.05,
            slippage_bps=10.0,
            params="{}",
            cache_dir="/tmp/test-cache",
            position_sizing="atr",
            risk_pct=0.02,
            atr_multiple=3.0,
        )
        assert config.position_sizing == "atr"
        assert config.sizing_risk_pct == 0.02
        assert config.sizing_atr_multiple == 3.0

    def test_empty_params_yields_empty_dict(self):
        """Passing '{}' for params should produce an empty strategy_params dict."""
        from datetime import datetime
        config = _build_config(
            strategy="sma_crossover",
            tickers="SPY",
            market="us_ca",
            universe="index",
            benchmark="SPY",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 12, 31),
            cash=10_000.0,
            max_positions=10,
            max_alloc=0.10,
            fee=0.05,
            slippage_bps=10.0,
            params="{}",
            cache_dir="/tmp/test-cache",
        )
        assert config.strategy_params == {}

    def test_universe_fallback_when_no_tickers(self):
        """When tickers is None, _build_config should call UniverseProvider."""
        from datetime import datetime
        with patch("backtester.cli.UniverseProvider", create=True) as mock_cls:
            # We need to patch where UniverseProvider is imported (lazy import inside _build_config)
            with patch("backtester.data.universe.UniverseProvider") as mock_univ_cls:
                mock_provider = MagicMock()
                mock_provider.get_tickers.return_value = ["AAPL", "MSFT"]
                mock_univ_cls.return_value = mock_provider

                config = _build_config(
                    strategy="sma_crossover",
                    tickers=None,
                    market="us",
                    universe="all",
                    benchmark="SPY",
                    start=datetime(2020, 1, 2),
                    end=datetime(2020, 12, 31),
                    cash=10_000.0,
                    max_positions=10,
                    max_alloc=0.10,
                    fee=0.05,
                    slippage_bps=10.0,
                    params="{}",
                    cache_dir="/tmp/test-cache",
                )
                assert config.tickers == ["AAPL", "MSFT"]
                mock_provider.get_tickers.assert_called_once_with(market="us", universe="all")


class TestRunCommand:
    """Tests for the 'run' CLI command."""

    def test_run_invokes_engine(self, runner, mock_result):
        """'run' command should construct config and invoke BacktestEngine.run()."""
        with patch("backtester.cli.BacktestEngine") as mock_engine_cls, \
             patch("backtester.cli.print_report") as mock_report, \
             patch("backtester.cli.plot_results") as mock_plot:

            mock_engine = MagicMock()
            mock_engine.run.return_value = mock_result
            mock_engine_cls.return_value = mock_engine

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--cash", "50000",
                "--max-positions", "5",
                "--max-alloc", "0.20",
                "--params", '{"sma_fast": 50, "sma_slow": 200}',
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            mock_engine_cls.assert_called_once()
            mock_engine.run.assert_called_once()
            mock_report.assert_called_once_with(mock_result)
            mock_plot.assert_called_once_with(mock_result)

    def test_run_passes_correct_config(self, runner, mock_result):
        """'run' should build BacktestConfig with the correct field values."""
        captured_config = {}

        def capture_engine(config):
            captured_config["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture_engine), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY,QQQ",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--cash", "50000",
                "--max-positions", "5",
                "--max-alloc", "0.20",
                "--fee", "1.50",
                "--slippage-bps", "15.0",
                "--params", '{"sma_fast": 50}',
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured_config["config"]
            assert config.strategy_name == "sma_crossover"
            assert config.tickers == ["SPY", "QQQ"]
            assert config.benchmark == "SPY"
            assert config.start_date == date(2020, 1, 2)
            assert config.end_date == date(2020, 12, 31)
            assert config.starting_cash == 50_000.0
            assert config.max_positions == 5
            assert config.max_alloc_pct == 0.20
            assert config.fee_per_trade == 1.50
            assert config.slippage_bps == 15.0
            assert config.strategy_params == {"sma_fast": 50}

    def test_run_with_invalid_strategy_exits_nonzero(self, runner):
        """'run' with a non-existent strategy should exit with an error."""
        # The engine constructor calls get_strategy which raises ValueError
        # for unknown strategies. We don't mock the engine here.
        with patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "nonexistent_strategy",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
            ])

            assert result.exit_code != 0

    def test_run_missing_required_options(self, runner):
        """'run' without required options should fail."""
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "Error" in result.output

    def test_run_with_regime_filter(self, runner, mock_result):
        """'run' with regime filter options should build config with RegimeFilter."""
        captured_config = {}

        def capture_engine(config):
            captured_config["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture_engine), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--regime-benchmark", "SPY",
                "--regime-fast", "50",
                "--regime-slow", "150",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured_config["config"]
            assert config.regime_filter is not None
            assert config.regime_filter.benchmark == "SPY"
            assert config.regime_filter.fast_period == 50
            assert config.regime_filter.slow_period == 150

    def test_run_with_stop_config(self, runner, mock_result):
        """'run' with stop options should build config with StopConfig."""
        captured_config = {}

        def capture_engine(config):
            captured_config["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture_engine), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--stop-loss", "0.05",
                "--take-profit", "0.20",
                "--trailing-stop", "0.08",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured_config["config"]
            assert config.stop_config is not None
            assert config.stop_config.stop_loss_pct == 0.05
            assert config.stop_config.take_profit_pct == 0.20
            assert config.stop_config.trailing_stop_pct == 0.08

    def test_run_with_position_sizing(self, runner, mock_result):
        """'run' with position sizing options should be reflected in config."""
        captured_config = {}

        def capture_engine(config):
            captured_config["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture_engine), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--position-sizing", "atr",
                "--risk-pct", "0.02",
                "--atr-multiple", "3.0",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured_config["config"]
            assert config.position_sizing == "atr"
            assert config.sizing_risk_pct == 0.02
            assert config.sizing_atr_multiple == 3.0

    def test_run_with_export_log(self, runner, mock_result):
        """'run' with --export-log should call export_activity_log_csv."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            log_path = f.name

        with patch("backtester.cli.BacktestEngine") as mock_engine_cls, \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"), \
             patch("backtester.cli.export_activity_log_csv") as mock_export:

            mock_engine = MagicMock()
            mock_engine.run.return_value = mock_result
            mock_engine_cls.return_value = mock_engine

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--export-log", log_path,
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            mock_export.assert_called_once_with(mock_result, log_path)
            assert f"Activity log exported to {log_path}" in result.output


class TestOptimizeCommand:
    """Tests for the 'optimize' CLI command."""

    def test_optimize_parses_grid_json(self, runner):
        """'optimize' should parse --grid JSON and pass to grid_search."""
        captured_args = {}

        def fake_grid_search(base_config, param_grid, optimize_metric="sharpe_ratio"):
            captured_args["config"] = base_config
            captured_args["grid"] = param_grid
            captured_args["metric"] = optimize_metric
            # Return a minimal OptimizationResult
            from backtester.research.optimizer import OptimizationResult
            return OptimizationResult(
                results_table=pd.DataFrame(),
                best_params={"sma_fast": 50},
                best_metric_value=1.5,
                optimize_metric=optimize_metric,
            )

        with patch("backtester.cli.BacktestEngine"), \
             patch("backtester.research.optimizer.grid_search", side_effect=fake_grid_search) as mock_gs, \
             patch("backtester.research.optimizer.print_optimization_results"):

            # We need to patch at the import site inside the optimize function
            with patch("backtester.cli.json.loads", wraps=json.loads):
                result = runner.invoke(cli, [
                    "optimize",
                    "--strategy", "sma_crossover",
                    "--tickers", "SPY",
                    "--benchmark", "SPY",
                    "--start", "2020-01-02",
                    "--end", "2020-12-31",
                    "--grid", '{"sma_fast": [50, 100], "sma_slow": [200, 300]}',
                    "--metric", "cagr",
                ])

                assert result.exit_code == 0, f"CLI failed: {result.output}"
                assert captured_args["grid"] == {"sma_fast": [50, 100], "sma_slow": [200, 300]}
                assert captured_args["metric"] == "cagr"

    def test_optimize_builds_correct_base_config(self, runner):
        """'optimize' should build base config with correct strategy and tickers."""
        captured_args = {}

        def fake_grid_search(base_config, param_grid, optimize_metric="sharpe_ratio"):
            captured_args["config"] = base_config
            from backtester.research.optimizer import OptimizationResult
            return OptimizationResult(
                results_table=pd.DataFrame(),
                best_params={},
                best_metric_value=0.0,
                optimize_metric=optimize_metric,
            )

        with patch("backtester.research.optimizer.grid_search", side_effect=fake_grid_search), \
             patch("backtester.research.optimizer.print_optimization_results"):

            result = runner.invoke(cli, [
                "optimize",
                "--strategy", "sma_crossover",
                "--tickers", "SPY,QQQ",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--cash", "75000",
                "--max-positions", "8",
                "--grid", '{"sma_fast": [50]}',
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured_args["config"]
            assert config.strategy_name == "sma_crossover"
            assert config.tickers == ["SPY", "QQQ"]
            assert config.starting_cash == 75_000.0
            assert config.max_positions == 8

    def test_optimize_missing_grid_fails(self, runner):
        """'optimize' without --grid should fail."""
        result = runner.invoke(cli, [
            "optimize",
            "--strategy", "sma_crossover",
            "--tickers", "SPY",
            "--benchmark", "SPY",
            "--start", "2020-01-02",
            "--end", "2020-12-31",
        ])
        assert result.exit_code != 0


class TestWalkForwardCommand:
    """Tests for the 'walk-forward' CLI command."""

    def test_walk_forward_invokes_correctly(self, runner):
        """'walk-forward' should parse options and call walk_forward()."""
        captured_args = {}

        def fake_walk_forward(base_config, param_grid, is_months=12,
                              oos_months=3, anchored=False,
                              optimize_metric="sharpe_ratio"):
            captured_args["config"] = base_config
            captured_args["grid"] = param_grid
            captured_args["is_months"] = is_months
            captured_args["oos_months"] = oos_months
            captured_args["anchored"] = anchored
            captured_args["metric"] = optimize_metric
            return {
                "num_windows": 0,
                "windows": [],
                "oos_metrics": {},
                "aggregate_sharpe": 0.0,
            }

        with patch("backtester.research.walk_forward.walk_forward", side_effect=fake_walk_forward), \
             patch("backtester.research.walk_forward.print_walk_forward_results"):

            result = runner.invoke(cli, [
                "walk-forward",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2018-01-02",
                "--end", "2020-12-31",
                "--grid", '{"sma_fast": [50, 100]}',
                "--is-months", "6",
                "--oos-months", "2",
                "--anchored",
                "--metric", "cagr",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert captured_args["grid"] == {"sma_fast": [50, 100]}
            assert captured_args["is_months"] == 6
            assert captured_args["oos_months"] == 2
            assert captured_args["anchored"] is True
            assert captured_args["metric"] == "cagr"

    def test_walk_forward_defaults(self, runner):
        """'walk-forward' should use defaults for is-months, oos-months, anchored."""
        captured_args = {}

        def fake_walk_forward(base_config, param_grid, is_months=12,
                              oos_months=3, anchored=False,
                              optimize_metric="sharpe_ratio"):
            captured_args["is_months"] = is_months
            captured_args["oos_months"] = oos_months
            captured_args["anchored"] = anchored
            captured_args["metric"] = optimize_metric
            return {
                "num_windows": 0,
                "windows": [],
                "oos_metrics": {},
                "aggregate_sharpe": 0.0,
            }

        with patch("backtester.research.walk_forward.walk_forward", side_effect=fake_walk_forward), \
             patch("backtester.research.walk_forward.print_walk_forward_results"):

            result = runner.invoke(cli, [
                "walk-forward",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2018-01-02",
                "--end", "2020-12-31",
                "--grid", '{"sma_fast": [50]}',
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert captured_args["is_months"] == 12
            assert captured_args["oos_months"] == 3
            assert captured_args["anchored"] is False
            assert captured_args["metric"] == "sharpe_ratio"


class TestNewConfigFlags:
    """Tests for Group A/B CLI flags that map to BacktestConfig fields."""

    def _make_base_kwargs(self):
        from datetime import datetime
        return dict(
            strategy="sma_crossover", tickers="SPY", market="us_ca",
            universe="index", benchmark="SPY",
            start=datetime(2020, 1, 2), end=datetime(2020, 12, 31),
            cash=10_000.0, max_positions=10, max_alloc=0.10,
            fee=0.05, slippage_bps=10.0, params="{}", cache_dir="/tmp/test",
        )

    def test_allow_short_flag(self):
        """--allow-short should set allow_short=True in config."""
        config = _build_config(**self._make_base_kwargs(), allow_short=True)
        assert config.allow_short is True

    def test_allow_short_default_false(self):
        """allow_short should default to False."""
        config = _build_config(**self._make_base_kwargs())
        assert config.allow_short is False

    def test_short_borrow_rate(self):
        """--short-borrow-rate should set short_borrow_rate in config."""
        config = _build_config(**self._make_base_kwargs(), short_borrow_rate=0.05)
        assert config.short_borrow_rate == 0.05

    def test_margin_requirement(self):
        """--margin-requirement should set margin_requirement in config."""
        config = _build_config(**self._make_base_kwargs(), margin_requirement=2.0)
        assert config.margin_requirement == 2.0

    def test_slippage_model(self):
        """--slippage-model should set slippage_model in config."""
        config = _build_config(**self._make_base_kwargs(), slippage_model="volume")
        assert config.slippage_model == "volume"

    def test_slippage_impact(self):
        """--slippage-impact should set slippage_impact_factor in config."""
        config = _build_config(**self._make_base_kwargs(), slippage_impact=0.25)
        assert config.slippage_impact_factor == 0.25

    def test_fee_model(self):
        """--fee-model should set fee_model in config."""
        config = _build_config(**self._make_base_kwargs(), fee_model="percentage")
        assert config.fee_model == "percentage"

    def test_fee_model_composite_us(self):
        """--fee-model composite_us should be accepted."""
        config = _build_config(**self._make_base_kwargs(), fee_model="composite_us")
        assert config.fee_model == "composite_us"

    def test_vol_target(self):
        """--vol-target should set sizing_target_vol in config."""
        config = _build_config(**self._make_base_kwargs(), vol_target=0.15)
        assert config.sizing_target_vol == 0.15

    def test_vol_lookback(self):
        """--vol-lookback should set sizing_vol_lookback in config."""
        config = _build_config(**self._make_base_kwargs(), vol_lookback=30)
        assert config.sizing_vol_lookback == 30

    def test_monte_carlo_runs(self):
        """--monte-carlo-runs should set monte_carlo_runs in config."""
        config = _build_config(**self._make_base_kwargs(), monte_carlo_runs=500)
        assert config.monte_carlo_runs == 500

    def test_regime_condition(self):
        """--regime-condition should set condition on RegimeFilter."""
        config = _build_config(
            **self._make_base_kwargs(),
            regime_benchmark="SPY",
            regime_condition="fast_below_slow",
        )
        assert config.regime_filter is not None
        assert config.regime_filter.condition == "fast_below_slow"

    def test_regime_condition_default(self):
        """Regime condition should default to fast_above_slow."""
        config = _build_config(
            **self._make_base_kwargs(),
            regime_benchmark="SPY",
        )
        assert config.regime_filter.condition == "fast_above_slow"


class TestNewFlagsCLIInvocation:
    """Tests that new flags are accepted by CLI commands and produce correct config."""

    def test_run_short_selling_flags(self, runner, mock_result):
        """run with short selling flags should set correct config fields."""
        captured = {}

        def capture(config):
            captured["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--allow-short",
                "--short-borrow-rate", "0.05",
                "--margin-requirement", "2.0",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.allow_short is True
            assert config.short_borrow_rate == 0.05
            assert config.margin_requirement == 2.0

    def test_run_slippage_model_volume(self, runner, mock_result):
        """run with --slippage-model volume and --slippage-impact should set config."""
        captured = {}

        def capture(config):
            captured["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--slippage-model", "volume",
                "--slippage-impact", "0.25",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.slippage_model == "volume"
            assert config.slippage_impact_factor == 0.25

    def test_run_fee_model(self, runner, mock_result):
        """run with --fee-model should set config.fee_model."""
        captured = {}

        def capture(config):
            captured["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--fee-model", "composite_us",
                "--fee", "5.0",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.fee_model == "composite_us"
            assert config.fee_per_trade == 5.0

    def test_run_vol_parity_flags(self, runner, mock_result):
        """run with vol parity flags should set config fields."""
        captured = {}

        def capture(config):
            captured["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--position-sizing", "vol_parity",
                "--vol-target", "0.15",
                "--vol-lookback", "30",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.position_sizing == "vol_parity"
            assert config.sizing_target_vol == 0.15
            assert config.sizing_vol_lookback == 30

    def test_run_monte_carlo_runs_flag(self, runner, mock_result):
        """run with --monte-carlo-runs should set config.monte_carlo_runs."""
        captured = {}

        def capture(config):
            captured["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--monte-carlo-runs", "500",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.monte_carlo_runs == 500

    def test_run_regime_condition_flag(self, runner, mock_result):
        """run with --regime-condition should set the condition on RegimeFilter."""
        captured = {}

        def capture(config):
            captured["config"] = config
            engine = MagicMock()
            engine.run.return_value = mock_result
            return engine

        with patch("backtester.cli.BacktestEngine", side_effect=capture), \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--regime-benchmark", "SPY",
                "--regime-condition", "fast_below_slow",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.regime_filter is not None
            assert config.regime_filter.condition == "fast_below_slow"


class TestAnalyticsOutputFlags:
    """Tests for Group C analytics output flags (run only)."""

    def test_tearsheet_flag(self, runner, mock_result):
        """--tearsheet should call generate_tearsheet."""
        with patch("backtester.cli.BacktestEngine") as mock_engine_cls, \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"), \
             patch("backtester.analytics.tearsheet.generate_tearsheet", return_value="/tmp/test.html") as mock_ts:

            mock_engine = MagicMock()
            mock_engine.run.return_value = mock_result
            mock_engine_cls.return_value = mock_engine

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--tearsheet", "/tmp/test.html",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            mock_ts.assert_called_once_with(mock_result, output_path="/tmp/test.html")
            assert "Tearsheet written to" in result.output

    def test_report_regime_flag(self, runner):
        """--report-regime should call regime_summary and print output."""
        portfolio = Portfolio(cash=100_000.0)
        portfolio.record_equity(date(2020, 1, 2))
        portfolio.record_equity(date(2020, 1, 3))
        benchmark_eq = [(date(2020, 1, 2), 100_000.0), (date(2020, 1, 3), 100_050.0)]
        bench_prices = pd.Series([300.0, 301.0],
                                 index=pd.DatetimeIndex([date(2020, 1, 2), date(2020, 1, 3)]))
        result_obj = BacktestResult(
            config=BacktestConfig(
                strategy_name="sma_crossover", tickers=["SPY"], benchmark="SPY",
                start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
                starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            ),
            portfolio=portfolio,
            benchmark_equity=benchmark_eq,
            benchmark_prices=bench_prices,
        )

        mock_summary = {
            "market_regime_perf": pd.DataFrame({"return": [0.05]}, index=["bull"]),
            "vol_regime_perf": pd.DataFrame({"return": [0.03]}, index=["low_vol"]),
        }

        with patch("backtester.cli.BacktestEngine") as mock_engine_cls, \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"), \
             patch("backtester.analytics.regime.regime_summary", return_value=mock_summary) as mock_rs:

            mock_engine = MagicMock()
            mock_engine.run.return_value = result_obj
            mock_engine_cls.return_value = mock_engine

            runner = CliRunner()
            cli_result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--report-regime",
            ])

            assert cli_result.exit_code == 0, f"CLI failed: {cli_result.output}"
            mock_rs.assert_called_once()
            assert "Regime Performance" in cli_result.output

    def test_report_regime_no_benchmark(self, runner, mock_result):
        """--report-regime without benchmark prices should print a message."""
        with patch("backtester.cli.BacktestEngine") as mock_engine_cls, \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"):

            mock_engine = MagicMock()
            mock_engine.run.return_value = mock_result
            mock_engine_cls.return_value = mock_engine

            result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--report-regime",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert "requires benchmark data" in result.output

    def test_report_correlation_flag(self, runner):
        """--report-correlation should call compute_correlation_matrix."""
        portfolio = Portfolio(cash=100_000.0)
        portfolio.record_equity(date(2020, 1, 2))
        portfolio.record_equity(date(2020, 1, 3))
        benchmark_eq = [(date(2020, 1, 2), 100_000.0), (date(2020, 1, 3), 100_050.0)]

        df1 = make_price_df(start_price=100.0)
        df2 = make_price_df(start_price=200.0)
        universe = {"SPY": df1, "QQQ": df2}

        result_obj = BacktestResult(
            config=BacktestConfig(
                strategy_name="sma_crossover", tickers=["SPY", "QQQ"], benchmark="SPY",
                start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
                starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            ),
            portfolio=portfolio,
            benchmark_equity=benchmark_eq,
            universe_data=universe,
        )

        mock_corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["SPY", "QQQ"], columns=["SPY", "QQQ"],
        )

        with patch("backtester.cli.BacktestEngine") as mock_engine_cls, \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"), \
             patch("backtester.analytics.correlation.compute_correlation_matrix", return_value=mock_corr) as mock_cc:

            mock_engine = MagicMock()
            mock_engine.run.return_value = result_obj
            mock_engine_cls.return_value = mock_engine

            runner = CliRunner()
            cli_result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY,QQQ",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--report-correlation",
            ])

            assert cli_result.exit_code == 0, f"CLI failed: {cli_result.output}"
            mock_cc.assert_called_once_with(universe)
            assert "Correlation Matrix" in cli_result.output

    def test_report_signal_decay_flag(self, runner):
        """--report-signal-decay should call signal_decay_summary."""
        portfolio = Portfolio(cash=100_000.0)
        portfolio.record_equity(date(2020, 1, 2))
        portfolio.record_equity(date(2020, 1, 3))

        df1 = make_price_df(start_price=100.0)
        universe = {"SPY": df1}

        # Create a mock trade
        mock_trade = MagicMock()
        mock_trade.symbol = "SPY"
        mock_trade.entry_date = date(2020, 1, 2)
        mock_trade.exit_date = date(2020, 1, 3)
        mock_trade.entry_price = 100.0
        portfolio.trade_log.append(mock_trade)

        result_obj = BacktestResult(
            config=BacktestConfig(
                strategy_name="sma_crossover", tickers=["SPY"], benchmark="SPY",
                start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
                starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            ),
            portfolio=portfolio,
            universe_data=universe,
        )

        mock_summary = {
            "total_signals": 1,
            "avg_decay": pd.Series([0.01, 0.02], index=["T+1", "T+2"]),
            "optimal_holding": {"optimal_days": 2, "peak_return": 0.02},
        }

        with patch("backtester.cli.BacktestEngine") as mock_engine_cls, \
             patch("backtester.cli.print_report"), \
             patch("backtester.cli.plot_results"), \
             patch("backtester.analytics.signal_decay.signal_decay_summary", return_value=mock_summary) as mock_sd:

            mock_engine = MagicMock()
            mock_engine.run.return_value = result_obj
            mock_engine_cls.return_value = mock_engine

            runner = CliRunner()
            cli_result = runner.invoke(cli, [
                "run",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--report-signal-decay",
            ])

            assert cli_result.exit_code == 0, f"CLI failed: {cli_result.output}"
            mock_sd.assert_called_once()
            assert "Signal Decay Analysis" in cli_result.output
            assert "Optimal holding period: 2 days" in cli_result.output


class TestCommandParity:
    """Tests that optimize and walk-forward accept the same flags as run."""

    def _make_fake_grid_search(self, captured):
        def fake(base_config, param_grid, optimize_metric="sharpe_ratio"):
            captured["config"] = base_config
            from backtester.research.optimizer import OptimizationResult
            return OptimizationResult(
                results_table=pd.DataFrame(), best_params={},
                best_metric_value=0.0, optimize_metric=optimize_metric,
            )
        return fake

    def _make_fake_walk_forward(self, captured):
        def fake(base_config, param_grid, is_months=12, oos_months=3,
                 anchored=False, optimize_metric="sharpe_ratio"):
            captured["config"] = base_config
            return {"num_windows": 0, "windows": [], "oos_metrics": {},
                    "aggregate_sharpe": 0.0}
        return fake

    def test_optimize_accepts_regime_flags(self, runner):
        """optimize should accept regime filter flags."""
        captured = {}
        with patch("backtester.research.optimizer.grid_search",
                   side_effect=self._make_fake_grid_search(captured)), \
             patch("backtester.research.optimizer.print_optimization_results"):
            result = runner.invoke(cli, [
                "optimize",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--grid", '{"sma_fast": [50]}',
                "--regime-benchmark", "SPY",
                "--regime-fast", "50",
                "--regime-slow", "150",
                "--regime-condition", "fast_below_slow",
            ])
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.regime_filter is not None
            assert config.regime_filter.condition == "fast_below_slow"

    def test_optimize_accepts_stop_flags(self, runner):
        """optimize should accept stop config flags."""
        captured = {}
        with patch("backtester.research.optimizer.grid_search",
                   side_effect=self._make_fake_grid_search(captured)), \
             patch("backtester.research.optimizer.print_optimization_results"):
            result = runner.invoke(cli, [
                "optimize",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--grid", '{"sma_fast": [50]}',
                "--stop-loss", "0.05",
                "--take-profit", "0.20",
            ])
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.stop_config is not None
            assert config.stop_config.stop_loss_pct == 0.05

    def test_optimize_accepts_short_selling_flags(self, runner):
        """optimize should accept short selling flags."""
        captured = {}
        with patch("backtester.research.optimizer.grid_search",
                   side_effect=self._make_fake_grid_search(captured)), \
             patch("backtester.research.optimizer.print_optimization_results"):
            result = runner.invoke(cli, [
                "optimize",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2020-01-02",
                "--end", "2020-12-31",
                "--grid", '{"sma_fast": [50]}',
                "--allow-short",
                "--fee-model", "percentage",
                "--slippage-model", "volume",
            ])
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.allow_short is True
            assert config.fee_model == "percentage"
            assert config.slippage_model == "volume"

    def test_walk_forward_accepts_regime_flags(self, runner):
        """walk-forward should accept regime filter flags."""
        captured = {}
        with patch("backtester.research.walk_forward.walk_forward",
                   side_effect=self._make_fake_walk_forward(captured)), \
             patch("backtester.research.walk_forward.print_walk_forward_results"):
            result = runner.invoke(cli, [
                "walk-forward",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2018-01-02",
                "--end", "2020-12-31",
                "--grid", '{"sma_fast": [50]}',
                "--regime-benchmark", "SPY",
                "--regime-condition", "fast_below_slow",
            ])
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.regime_filter is not None
            assert config.regime_filter.condition == "fast_below_slow"

    def test_walk_forward_accepts_stop_and_sizing_flags(self, runner):
        """walk-forward should accept stop and position sizing flags."""
        captured = {}
        with patch("backtester.research.walk_forward.walk_forward",
                   side_effect=self._make_fake_walk_forward(captured)), \
             patch("backtester.research.walk_forward.print_walk_forward_results"):
            result = runner.invoke(cli, [
                "walk-forward",
                "--strategy", "sma_crossover",
                "--tickers", "SPY",
                "--benchmark", "SPY",
                "--start", "2018-01-02",
                "--end", "2020-12-31",
                "--grid", '{"sma_fast": [50]}',
                "--stop-loss", "0.05",
                "--position-sizing", "vol_parity",
                "--vol-target", "0.15",
                "--vol-lookback", "30",
            ])
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            config = captured["config"]
            assert config.stop_config is not None
            assert config.position_sizing == "vol_parity"
            assert config.sizing_target_vol == 0.15
            assert config.sizing_vol_lookback == 30


class TestEngineWiring:
    """Tests that engine correctly wires config fields to execution models."""

    def test_fee_factory_per_trade(self):
        """fee_model='per_trade' should create PerTradeFee."""
        from backtester.execution.fees import PerTradeFee
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["SPY"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            fee_model="per_trade", fee_per_trade=1.0,
        )
        engine = BacktestEngine(config)
        assert isinstance(engine._broker._fees, PerTradeFee)

    def test_fee_factory_percentage(self):
        """fee_model='percentage' should create PercentageFee."""
        from backtester.execution.fees import PercentageFee
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["SPY"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            fee_model="percentage", fee_per_trade=5.0,
        )
        engine = BacktestEngine(config)
        assert isinstance(engine._broker._fees, PercentageFee)

    def test_fee_factory_composite_us(self):
        """fee_model='composite_us' should create CompositeFee."""
        from backtester.execution.fees import CompositeFee
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["SPY"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            fee_model="composite_us", fee_per_trade=5.0,
        )
        engine = BacktestEngine(config)
        assert isinstance(engine._broker._fees, CompositeFee)

    def test_volume_slippage_uses_config_impact(self):
        """VolumeSlippage should use config's slippage_impact_factor."""
        from backtester.execution.slippage import VolumeSlippage
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["SPY"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            slippage_model="volume", slippage_impact_factor=0.25,
        )
        engine = BacktestEngine(config)
        assert isinstance(engine._broker._slippage, VolumeSlippage)
        assert engine._broker._slippage._impact == 0.25

    def test_vol_parity_uses_config_params(self):
        """VolatilityParity should use config's target_vol and lookback."""
        from backtester.execution.position_sizing import VolatilityParity
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["SPY"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            position_sizing="vol_parity",
            sizing_target_vol=0.15, sizing_vol_lookback=30,
        )
        engine = BacktestEngine(config)
        assert isinstance(engine._sizer, VolatilityParity)
        assert engine._sizer._target_vol == 0.15
        assert engine._sizer._lookback == 30

    def test_regime_fast_below_slow(self):
        """fast_below_slow regime condition should return True when fast < slow."""
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["SPY"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
            regime_filter=RegimeFilter(
                benchmark="SPY", indicator="sma",
                fast_period=50, slow_period=200,
                condition="fast_below_slow",
            ),
        )
        engine = BacktestEngine(config)
        # fast < slow → True for fast_below_slow
        row = pd.Series({"regime_fast": 100.0, "regime_slow": 200.0})
        assert engine._check_regime(row) == True
        # fast > slow → False for fast_below_slow
        row2 = pd.Series({"regime_fast": 250.0, "regime_slow": 200.0})
        assert engine._check_regime(row2) == False


class TestVerboseFlag:
    """Tests for the --verbose / -v global flag."""

    def test_verbose_flag_accepted(self, runner):
        """The -v flag should be accepted without error."""
        result = runner.invoke(cli, ["-v", "list-strategies"])
        assert result.exit_code == 0
        assert "sma_crossover" in result.output
