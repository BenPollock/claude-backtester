"""Tests for multi-strategy portfolio support."""

import tempfile
from dataclasses import replace
from datetime import date

import pytest
import pandas as pd

from backtester.config import BacktestConfig
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.analytics.metrics import compute_all_metrics
from backtester.research.multi_strategy import (
    StrategyAllocation,
    MultiStrategyConfig,
    MultiStrategyResult,
    run_multi_strategy,
    compute_attribution,
    print_multi_strategy_report,
)
from tests.conftest import make_price_df, MockDataSource


def _make_data_manager(tmpdir, tickers=None, days=252, start_price=100.0,
                       daily_return=0.0005):
    """Create a DataManager backed by MockDataSource with synthetic data."""
    tickers = tickers or ["TEST"]
    source = MockDataSource()
    df = make_price_df(start="2020-01-02", days=days, start_price=start_price,
                       daily_return=daily_return)
    for t in tickers:
        source.add(t, df)
    return DataManager(cache_dir=tmpdir, source=source)


def _make_base_config(tmpdir, tickers=None, starting_cash=100_000.0):
    """Build a base BacktestConfig for testing."""
    tickers = tickers or ["TEST"]
    return BacktestConfig(
        strategy_name="sma_crossover",
        tickers=tickers,
        benchmark=tickers[0],
        start_date=date(2020, 1, 2),
        end_date=date(2020, 12, 31),
        starting_cash=starting_cash,
        max_positions=10,
        max_alloc_pct=0.20,
        fee_per_trade=0.0,
        slippage_bps=0.0,
        data_cache_dir=tmpdir,
        strategy_params={"sma_fast": 20, "sma_slow": 50},
    )


class TestMultiStrategyConfig:
    """Validation and immutability of MultiStrategyConfig."""

    def test_config_is_frozen(self):
        """MultiStrategyConfig should reject attribute assignment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            alloc = StrategyAllocation(
                strategy_name="sma_crossover",
                params={"sma_fast": 20, "sma_slow": 50},
                weight=1.0,
            )
            mc = MultiStrategyConfig(allocations=(alloc,), base_config=base)

            with pytest.raises(AttributeError):
                mc.allocations = ()

    def test_strategy_allocation_is_frozen(self):
        alloc = StrategyAllocation(
            strategy_name="sma_crossover",
            params={"sma_fast": 20, "sma_slow": 50},
            weight=0.5,
        )
        with pytest.raises(AttributeError):
            alloc.weight = 0.8

    def test_weights_exceeding_one_rejected(self):
        """Weights summing to more than 1.0 should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.6),
                StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 100}, 0.6),
            )
            with pytest.raises(ValueError, match="exceeds 1.0"):
                MultiStrategyConfig(allocations=allocs, base_config=base)

    def test_negative_weight_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, -0.3),
            )
            with pytest.raises(ValueError, match="Negative weight"):
                MultiStrategyConfig(allocations=allocs, base_config=base)

    def test_weights_summing_to_one_ok(self):
        """Exact 1.0 total weight should be accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.5),
                StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 100}, 0.5),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            assert len(mc.allocations) == 2

    def test_weights_below_one_ok(self):
        """Weights below 1.0 are fine (remainder is idle cash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.3),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            assert len(mc.allocations) == 1


class TestSingleStrategyAllocation:
    """A single strategy at weight=1.0 should match a normal backtest."""

    def test_single_strategy_matches_normal_run(self):
        """Running one strategy via multi-strategy with weight=1.0 should
        produce the same equity as running BacktestEngine directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            # Direct run
            engine = BacktestEngine(base, data_manager=dm)
            direct_result = engine.run()

            # Multi-strategy run (weight=1.0, same strategy + params)
            alloc = StrategyAllocation(
                strategy_name="sma_crossover",
                params={"sma_fast": 20, "sma_slow": 50},
                weight=1.0,
            )
            mc = MultiStrategyConfig(allocations=(alloc,), base_config=base)

            # Need a fresh DataManager because the engine may have consumed data
            dm2 = _make_data_manager(tmpdir)
            multi_result = run_multi_strategy(mc, data_manager=dm2)

            direct_eq = direct_result.equity_series
            multi_eq = multi_result.combined_equity_curve

            # Equity curves should match (same strategy, same cash)
            assert len(direct_eq) == len(multi_eq)
            pd.testing.assert_series_equal(
                direct_eq.reset_index(drop=True),
                multi_eq.reset_index(drop=True),
                check_names=False,
                atol=0.01,
            )


class TestTwoStrategyAllocation:
    """Two strategies with 50/50 split."""

    def test_combined_equity_is_sum_of_individuals(self):
        """Combined equity curve should equal the sum of individual curves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir, starting_cash=100_000.0)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.5),
                StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 100}, 0.5),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            # Sum individual equity curves
            individual_sum = pd.Series(0.0, index=result.combined_equity_curve.index)
            for name, res in result.per_strategy_results.items():
                eq = res.equity_series.reindex(individual_sum.index).ffill().bfill()
                individual_sum = individual_sum + eq

            pd.testing.assert_series_equal(
                result.combined_equity_curve,
                individual_sum,
                check_names=False,
                atol=0.01,
            )

    def test_per_strategy_weights_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.5),
                StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 100}, 0.5),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            # Two allocations of the same strategy get _1 and _2 suffixes
            assert len(result.per_strategy_weights) == 2
            assert "sma_crossover_1" in result.per_strategy_weights
            assert "sma_crossover_2" in result.per_strategy_weights
            for w in result.per_strategy_weights.values():
                assert w == 0.5


class TestAttribution:
    """Attribution analysis tests."""

    def test_contribution_sums_to_100(self):
        """Contribution percentages should sum to approximately 100%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.5),
                StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 100}, 0.5),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            total_contrib = result.attribution["contribution_pct"].sum()
            assert abs(total_contrib - 100.0) < 0.1

    def test_attribution_has_correct_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 1.0),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            expected_cols = {"strategy_name", "weight", "cagr", "sharpe",
                             "pnl", "contribution_pct"}
            assert set(result.attribution.columns) == expected_cols

    def test_attribution_single_strategy_is_100(self):
        """A single strategy should have 100% contribution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 1.0),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            assert len(result.attribution) == 1
            assert abs(result.attribution.iloc[0]["contribution_pct"] - 100.0) < 0.1


class TestPerStrategyMetrics:
    """Per-strategy metrics computed correctly."""

    def test_per_strategy_metrics_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.6),
                StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 100}, 0.4),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            for name, metrics in result.per_strategy_metrics.items():
                assert "cagr" in metrics
                assert "sharpe_ratio" in metrics
                assert "max_drawdown" in metrics
                assert "total_trades" in metrics

    def test_combined_metrics_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 1.0),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            m = result.combined_metrics
            assert "cagr" in m
            assert "sharpe_ratio" in m
            assert "max_drawdown" in m
            assert "total_trades" in m


class TestEdgeCases:
    """Edge cases: zero trades, idle cash."""

    def test_strategy_with_zero_trades(self):
        """A strategy that produces no trades should still contribute to results.

        Use very long SMA periods so no crossover occurs within the data window.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir, days=252)

            allocs = (
                StrategyAllocation(
                    "sma_crossover",
                    {"sma_fast": 200, "sma_slow": 250},  # longer than data -> no signals
                    weight=0.5,
                ),
                StrategyAllocation(
                    "sma_crossover",
                    {"sma_fast": 20, "sma_slow": 50},
                    weight=0.5,
                ),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            # Should still produce a combined result without error
            assert result is not None
            assert len(result.combined_equity_curve) > 0

    def test_uninvested_cash_remainder(self):
        """When weights < 1.0, the remainder should stay as flat cash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir, starting_cash=100_000.0)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.5),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            # First day's combined equity should be close to full starting cash
            # (50% allocated + 50% idle cash)
            first_eq = result.combined_equity_curve.iloc[0]
            assert abs(first_eq - 100_000.0) < 1.0

    def test_combined_trades_merges_all(self):
        """combined_trades should contain trades from all strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.5),
                StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 100}, 0.5),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            # Combined trades should be the union of all per-strategy trades
            total_individual = sum(
                len(r.trades) for r in result.per_strategy_results.values()
            )
            assert len(result.combined_trades) == total_individual


class TestReport:
    """Verify report printing doesn't raise."""

    def test_print_multi_strategy_report_no_error(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 0.5),
                StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 100}, 0.5),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            # Should not raise
            print_multi_strategy_report(result)

            captured = capsys.readouterr()
            assert "MULTI-STRATEGY PORTFOLIO REPORT" in captured.out
            assert "Combined Portfolio" in captured.out
            assert "Per-Strategy Breakdown" in captured.out
            assert "Attribution Analysis" in captured.out

    def test_print_report_single_strategy(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_config(tmpdir)
            dm = _make_data_manager(tmpdir)

            allocs = (
                StrategyAllocation("sma_crossover", {"sma_fast": 20, "sma_slow": 50}, 1.0),
            )
            mc = MultiStrategyConfig(allocations=allocs, base_config=base)
            result = run_multi_strategy(mc, data_manager=dm)

            print_multi_strategy_report(result)
            captured = capsys.readouterr()
            assert "sma_crossover" in captured.out
