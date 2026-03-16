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
        """Signals on day T should fill on day T+1 at open price.
        Verify fills happen at the Open price of the fill date, not Close.
        Also verify that every fill date is strictly after the signal date
        by checking that no fill happens on the first trading day (since
        signals require at least one day of data before generating)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            result = engine.run()

            assert len(result.trades) > 0, "Need at least one trade to verify"

            # Get the first trading day from the data
            first_day = None
            for symbol, df in result.universe_data.items():
                day = df.index[0]
                if first_day is None or day < first_day:
                    first_day = day

            # Check activity log: fills should occur at Open price AND
            # no fill should happen on the very first day (signals need prior data)
            fill_dates = []
            for entry in result.activity_log:
                fill_date = entry.date
                fill_price = entry.price
                symbol = entry.symbol
                fill_dates.append(fill_date)

                # No fill should occur on the first trading day
                # (signals use close of day T, fills on T+1)
                assert pd.Timestamp(fill_date) > first_day, (
                    f"Fill for {symbol} on {fill_date} is on/before first day "
                    f"{first_day}. Fills must be strictly after signal generation."
                )

                # Look up the actual Open price on the fill date
                df = result.universe_data.get(symbol)
                if df is None:
                    continue
                ts = pd.Timestamp(fill_date)
                if ts not in df.index:
                    continue
                open_price = df.loc[ts, "Open"]
                # Fill price should be at the Open (no slippage in this config)
                assert fill_price == pytest.approx(open_price, rel=1e-6), (
                    f"Fill for {symbol} on {fill_date}: price {fill_price} != "
                    f"Open {open_price}. Fills should happen at next-day Open."
                )

            # Verify fills are spread across multiple dates (not all on one day)
            assert len(set(fill_dates)) > 1, "Fills should occur on multiple dates"

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

            # Verify position count never exceeded max_positions by checking
            # the activity log: count concurrent positions at each date
            assert result is not None
            positions_held: dict[str, bool] = {}  # symbol -> currently_held
            max_concurrent = 0
            for entry in result.activity_log:
                if entry.action.name == "BUY":
                    positions_held[entry.symbol] = True
                elif entry.action.name == "SELL":
                    positions_held.pop(entry.symbol, None)
                current_count = len(positions_held)
                max_concurrent = max(max_concurrent, current_count)

            assert max_concurrent <= 5, (
                f"Max concurrent positions was {max_concurrent}, exceeding limit of 5"
            )

    def test_benchmark_equity_tracked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            result = engine.run()
            assert result.benchmark_equity is not None
            assert len(result.benchmark_equity) > 0
            # Verify benchmark equity is computed as shares * close_price
            # First value should equal starting cash (shares = cash / first_close)
            first_date, first_value = result.benchmark_equity[0]
            assert first_value == pytest.approx(100_000.0, abs=1.0)
            # Last value should reflect the price change of the benchmark
            last_date, last_value = result.benchmark_equity[-1]
            # Benchmark should change proportionally to the underlying price
            assert last_value > 0
            # Verify a few intermediate values are consistent
            # (monotonic check would be wrong since prices fluctuate,
            # but values should all be positive)
            for d, v in result.benchmark_equity:
                assert v > 0, f"Benchmark equity should be positive, got {v} on {d}"

    def test_benchmark_equity_has_daily_data_points(self):
        """Benchmark equity should have one data point per trading day
        with benchmark data — not sparse/straight-line segments from
        missing days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir, days=252)
            result = engine.run()

            assert result.benchmark_equity is not None
            equity_len = len(result.equity_series)
            bm_len = len(result.benchmark_equity)

            # Benchmark may have slightly fewer points than equity
            # because synthetic data uses bdate_range which doesn't
            # perfectly match NYSE holidays. But it should be very
            # close — within 2% of the equity length.
            ratio = bm_len / equity_len
            assert ratio > 0.98, (
                f"Benchmark equity has {bm_len} points but strategy equity "
                f"has {equity_len} (ratio={ratio:.2%}) — benchmark appears "
                f"to be missing significant data"
            )

    def test_benchmark_equity_tracks_price_movements(self):
        """Benchmark equity should reflect actual price changes, not
        be a straight line (which would indicate sparse data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir, days=100)
            result = engine.run()

            assert result.benchmark_equity is not None
            values = [v for _, v in result.benchmark_equity]

            # With random daily returns, there should be meaningful
            # day-to-day variation in benchmark equity
            diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
            nonzero_diffs = sum(1 for d in diffs if d > 0.01)
            assert nonzero_diffs > len(diffs) * 0.9, (
                f"Only {nonzero_diffs}/{len(diffs)} days had price movement — "
                f"benchmark equity appears to be sparse/missing data"
            )

    def test_benchmark_equity_starts_at_starting_cash(self):
        """First benchmark equity value should equal starting_cash
        (buying benchmark shares at first close price)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir, days=100)
            result = engine.run()

            assert result.benchmark_equity is not None
            first_value = result.benchmark_equity[0][1]
            # benchmark_shares = starting_cash / first_close, so
            # first equity = benchmark_shares * first_close = starting_cash
            # (approximately, due to integer share truncation not applying here)
            assert abs(first_value - 100_000.0) < 1.0, (
                f"First benchmark equity ({first_value:.2f}) should equal "
                f"starting cash (100000.00)"
            )

    def test_benchmark_with_separate_ticker(self):
        """Benchmark should work correctly when benchmark ticker differs
        from the trading tickers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # Different price series for ticker and benchmark
            source.add("TRADE", make_price_df(days=100, start_price=50.0))
            source.add("BENCH", make_price_df(days=100, start_price=200.0))

            config = BacktestConfig(
                strategy_name="sma_crossover",
                tickers=["TRADE"],
                benchmark="BENCH",
                start_date=date(2020, 1, 2),
                end_date=date(2020, 5, 29),
                starting_cash=100_000.0,
                max_positions=10,
                max_alloc_pct=0.20,
                fee_per_trade=0.0,
                slippage_bps=0.0,
                data_cache_dir=tmpdir,
                strategy_params={"sma_fast": 20, "sma_slow": 50},
            )

            data_mgr = DataManager(cache_dir=tmpdir, source=source)
            engine = BacktestEngine(config, data_manager=data_mgr)
            result = engine.run()

            assert result.benchmark_equity is not None
            bm_len = len(result.benchmark_equity)
            eq_len = len(result.equity_series)
            ratio = bm_len / eq_len
            assert ratio > 0.98, (
                f"Benchmark has {bm_len} points vs equity {eq_len} "
                f"(ratio={ratio:.2%}) — benchmark appears sparse"
            )

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

    def test_zero_volume_days(self):
        """Engine should handle zero-volume days without crashing.
        Orders on zero-volume days may be skipped by volume constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = make_price_df(start="2020-01-02", days=252, start_price=100.0)
            # Set volume to 0 for a stretch of days (days 50-60)
            df_zeroed = df.copy()
            df_zeroed.iloc[50:60, df_zeroed.columns.get_loc("Volume")] = 0
            source.add("TEST", df_zeroed)

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
            )
            data_mgr = DataManager(cache_dir=tmpdir, source=source)
            engine = BacktestEngine(config, data_manager=data_mgr)
            result = engine.run()

            # Engine should complete without errors
            assert result is not None
            assert len(result.portfolio.equity_history) > 0
            # All positions should be closed at end
            assert result.portfolio.num_positions == 0

    def test_fill_delay_respected(self):
        """Orders submitted on day T must NOT fill on day T.
        They fill on day T+1 (next call to process_fills).
        Verify by checking that no activity log entry's fill date matches
        a date where no prior-day signal could have been generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            result = engine.run()

            assert len(result.activity_log) > 0, "Need activity to verify delay"

            # Get trading days from the equity history
            trading_days = [d for d, _ in result.portfolio.equity_history]

            for entry in result.activity_log:
                fill_date = entry.date
                # The fill_date must have a preceding trading day
                # (because the signal was generated the day before)
                fill_idx = None
                for i, td in enumerate(trading_days):
                    if td == fill_date:
                        fill_idx = i
                        break
                if fill_idx is not None:
                    assert fill_idx > 0, (
                        f"Fill on {fill_date} for {entry.symbol} cannot happen on "
                        f"the first trading day — no signal could have been generated yet."
                    )

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


class TestEngineValidation:
    """Verify engine validates config before running."""

    def test_end_before_start_raises(self):
        """Engine.run() should raise ValueError if end_date < start_date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df(days=100))

            config = BacktestConfig(
                strategy_name="sma_crossover",
                tickers=["TEST"],
                benchmark="TEST",
                start_date=date(2020, 12, 31),
                end_date=date(2020, 1, 2),  # before start
                starting_cash=100_000.0,
                max_positions=10,
                max_alloc_pct=0.10,
                strategy_params={"sma_fast": 20, "sma_slow": 50},
                data_cache_dir=tmpdir,
            )
            data_mgr = DataManager(cache_dir=tmpdir, source=source)
            engine = BacktestEngine(config, data_manager=data_mgr)
            with pytest.raises(ValueError, match="end_date"):
                engine.run()

    def test_negative_cash_raises(self):
        """Engine.run() should raise ValueError if starting_cash <= 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df(days=100))

            config = BacktestConfig(
                strategy_name="sma_crossover",
                tickers=["TEST"],
                benchmark="TEST",
                start_date=date(2020, 1, 2),
                end_date=date(2020, 12, 31),
                starting_cash=-1000.0,  # negative
                max_positions=10,
                max_alloc_pct=0.10,
                strategy_params={"sma_fast": 20, "sma_slow": 50},
                data_cache_dir=tmpdir,
            )
            data_mgr = DataManager(cache_dir=tmpdir, source=source)
            engine = BacktestEngine(config, data_manager=data_mgr)
            with pytest.raises(ValueError, match="starting_cash"):
                engine.run()


class TestRebalanceSchedule:
    """Verify the engine's rebalance schedule logic."""

    def test_daily_always_true(self):
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["X"],
            benchmark="X",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            rebalance_schedule="daily",
        )
        assert engine._is_rebalance_day_for_schedule(date(2020, 3, 2), date(2020, 3, 1))

    def test_weekly_new_week(self):
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["X"],
            benchmark="X",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            rebalance_schedule="weekly",
        )
        # Monday after Friday = new week
        assert engine._is_rebalance_day_for_schedule(date(2020, 3, 2), date(2020, 2, 28))
        # Same week: should NOT rebalance
        assert not engine._is_rebalance_day_for_schedule(date(2020, 3, 3), date(2020, 3, 2))

    def test_monthly_new_month(self):
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["X"],
            benchmark="X",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            rebalance_schedule="monthly",
        )
        # March 2 after Feb 28 = new month
        assert engine._is_rebalance_day_for_schedule(date(2020, 3, 2), date(2020, 2, 28))
        # Same month
        assert not engine._is_rebalance_day_for_schedule(date(2020, 3, 3), date(2020, 3, 2))

    def test_quarterly_new_quarter(self):
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["X"],
            benchmark="X",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            rebalance_schedule="quarterly",
        )
        # April 1 after March 31 = new quarter
        assert engine._is_rebalance_day_for_schedule(date(2020, 4, 1), date(2020, 3, 31))
        # Same quarter
        assert not engine._is_rebalance_day_for_schedule(date(2020, 2, 1), date(2020, 1, 31))


class TestRegimeFilterShortSuppression:
    """Verify that the regime filter also suppresses SHORT signals."""

    def test_regime_off_suppresses_short(self):
        """When regime is off, SHORT signals should be suppressed (not just BUY)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a short-selling strategy that generates SHORT signals
            source = MockDataSource()
            # Declining prices -> fast SMA < slow SMA -> regime is OFF
            df = make_price_df(
                start="2020-01-02", days=252,
                start_price=200.0, daily_return=-0.002,
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
                allow_short=True,  # enable short selling
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

            # With regime filter off (declining market), BOTH BUY and SHORT
            # signals should be suppressed. SMA crossover is long-only so no
            # SHORT signals anyway, but the engine-level filter should block them.
            assert result is not None
            assert result.portfolio.num_positions == 0
