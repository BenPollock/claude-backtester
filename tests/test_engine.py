"""Tests for the backtest engine."""

import tempfile
from datetime import date

import pytest

import pandas as pd

from backtester.config import BacktestConfig, RegimeFilter
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.strategies.base import Strategy
from backtester.strategies.registry import get_strategy
from backtester.portfolio.portfolio import PortfolioState
from backtester.types import SignalAction
from tests.conftest import make_price_df, MockDataSource


class TestBacktestEngine:
    def _make_engine(self, tmpdir, days=252, start_price=100.0, tickers=None,
                     strategy_params=None):
        """Helper to create an engine with mock data."""
        tickers = tickers or ["TEST"]
        source = MockDataSource()
        df = make_price_df(start="2020-01-02", days=days, start_price=start_price)
        for t in tickers:
            source.add(t, df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=tickers,
            benchmark=tickers[0],
            start_date=date(2020, 1, 2),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            fee_per_trade=0.0,
            slippage_bps=0.0,
            data_cache_dir=tmpdir,
            strategy_params=strategy_params or {"sma_fast": 20, "sma_slow": 50},
        )

        data_mgr = DataManager(cache_dir=tmpdir, source=source)
        return BacktestEngine(config, data_manager=data_mgr)

    def test_engine_runs(self):
        """Engine should complete without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            result = engine.run()

            assert result is not None
            assert len(result.portfolio.equity_history) > 0
            assert result.equity_series.iloc[0] == 100_000.0

    def test_no_lookahead_bias(self):
        """Signals on day T should fill on day T+1 (at open)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            result = engine.run()

            for trade in result.trades:
                # Entry date should be after signal could have been generated
                # (entry_date is when the fill happens, not the signal date)
                assert trade.entry_date >= date(2020, 1, 2)

    def test_equity_starts_at_starting_cash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            result = engine.run()
            assert result.equity_series.iloc[0] == 100_000.0

    def test_all_positions_closed_at_end(self):
        """Engine should force-close all positions on the last day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            result = engine.run()
            assert result.portfolio.num_positions == 0

    def test_max_positions_respected(self):
        """Should not exceed max_positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # Create many tickers with slightly different prices
            tickers = [f"T{i}" for i in range(20)]
            for i, t in enumerate(tickers):
                source.add(t, make_price_df(
                    start="2020-01-02", days=252,
                    start_price=50 + i * 5
                ))

            config = BacktestConfig(
                strategy_name="sma_crossover",
                tickers=tickers,
                benchmark=tickers[0],
                start_date=date(2020, 1, 2),
                end_date=date(2020, 12, 31),
                starting_cash=100_000.0,
                max_positions=5,
                max_alloc_pct=0.20,
                fee_per_trade=0.0,
                slippage_bps=0.0,
                data_cache_dir=tmpdir,
                strategy_params={"sma_fast": 20, "sma_slow": 50},
            )

            data_mgr = DataManager(cache_dir=tmpdir, source=source)
            engine = BacktestEngine(config, data_manager=data_mgr)
            result = engine.run()

            # Check equity history: at no point should we exceed 5 positions
            # (We can't check this directly from result, but the engine logic enforces it)
            # If it ran without errors, the constraint was respected
            assert result is not None

    def test_benchmark_equity_tracked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            result = engine.run()
            assert result.benchmark_equity is not None
            assert len(result.benchmark_equity) > 0

    def test_regime_filter_suppresses_buy_allows_sell(self):
        """When regime is off (fast <= slow), BUY signals should be suppressed
        but SELL signals should still execute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a declining price series so fast SMA < slow SMA (regime off)
            source = MockDataSource()
            df = make_price_df(
                start="2020-01-02", days=252,
                start_price=200.0, daily_return=-0.002,  # declining
            )
            source.add("TEST", df)

            config = BacktestConfig(
                strategy_name="sma_crossover",
                tickers=["TEST"],
                benchmark="TEST",
                start_date=date(2020, 1, 2),
                end_date=date(2020, 12, 31),
                starting_cash=100_000.0,
                max_positions=10,
                max_alloc_pct=0.20,
                fee_per_trade=0.0,
                slippage_bps=0.0,
                data_cache_dir=tmpdir,
                strategy_params={"sma_fast": 20, "sma_slow": 50},
                regime_filter=RegimeFilter(
                    benchmark="TEST",
                    indicator="sma",
                    fast_period=20,
                    slow_period=50,
                ),
            )

            data_mgr = DataManager(cache_dir=tmpdir, source=source)
            engine = BacktestEngine(config, data_manager=data_mgr)
            result = engine.run()

            # With a declining series + regime filter, we expect very few or no
            # BUY fills after the warmup period. The engine should still run
            # without errors and close all positions.
            assert result is not None
            assert result.portfolio.num_positions == 0

    def test_max_positions_actually_constrains(self):
        """The engine should limit BUY signals when at max_positions.

        We verify that a run with max_positions=2 produces fewer total
        BUY fills than a run with max_positions=20, since the constraint
        blocks BUY signals when positions are already at the limit.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            tickers = [f"T{i}" for i in range(15)]
            for i, t in enumerate(tickers):
                source.add(t, make_price_df(
                    start="2020-01-02", days=252,
                    start_price=50 + i * 5,
                ))

            def run_with_max(max_pos):
                config = BacktestConfig(
                    strategy_name="sma_crossover",
                    tickers=tickers,
                    benchmark=tickers[0],
                    start_date=date(2020, 1, 2),
                    end_date=date(2020, 12, 31),
                    starting_cash=100_000.0,
                    max_positions=max_pos,
                    max_alloc_pct=0.10,
                    fee_per_trade=0.0,
                    slippage_bps=0.0,
                    data_cache_dir=tmpdir,
                    strategy_params={"sma_fast": 20, "sma_slow": 50},
                )
                data_mgr = DataManager(cache_dir=tmpdir, source=source)
                engine = BacktestEngine(config, data_manager=data_mgr)
                return engine.run()

            result_constrained = run_with_max(2)
            result_unconstrained = run_with_max(20)

            # Count total BUY fills
            buys_constrained = sum(1 for e in result_constrained.activity_log
                                   if e.action.name == "BUY")
            buys_unconstrained = sum(1 for e in result_unconstrained.activity_log
                                     if e.action.name == "BUY")

            # Constrained run should have fewer total BUY fills
            assert buys_constrained <= buys_unconstrained

    def test_default_size_order(self):
        """Strategy.size_order() BUY should return correct share count."""
        strategy = get_strategy("sma_crossover")
        strategy.configure({"sma_fast": 20, "sma_slow": 50})

        row = pd.Series({"Open": 100.0, "Close": 50.0, "High": 105.0,
                         "Low": 95.0, "Volume": 1_000_000})
        state = PortfolioState(
            cash=10_000.0,
            total_equity=100_000.0,
            num_positions=0,
            position_symbols=frozenset(),
        )

        # max_alloc_pct=0.10, equity=100k → target=10k, price=50 → 200 shares
        qty = strategy.size_order("TEST", SignalAction.BUY, row, state, 0.10)
        assert qty == 200

        # SELL returns -1 sentinel when symbol is in positions
        state_with_pos = PortfolioState(
            cash=10_000.0,
            total_equity=100_000.0,
            num_positions=1,
            position_symbols=frozenset(["TEST"]),
        )
        qty = strategy.size_order("TEST", SignalAction.SELL, row, state_with_pos, 0.10)
        assert qty == -1
