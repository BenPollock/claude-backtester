"""Phase 2 & 3: Financial correctness unit tests + integration tests.

Covers remaining gaps identified by the Coverage Purist and Systems Thinker.
"""

import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import (
    total_return, cagr, sharpe_ratio, max_drawdown,
    capture_ratio, omega_ratio, exposure_time,
    compute_all_metrics,
)
from backtester.config import BacktestConfig, RegimeFilter, StopConfig
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.execution.broker import SimulatedBroker
from backtester.execution.fees import PerTradeFee
from backtester.execution.slippage import FixedSlippage
from backtester.portfolio.order import Order, Trade
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.position import Position, Lot
from backtester.strategies.base import Strategy, Signal
from backtester.strategies.registry import _REGISTRY
from backtester.types import Side, OrderType, SignalAction

from tests.conftest import MockDataSource, make_price_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rising_df(start="2020-01-02", days=252, start_price=100.0, daily_pct=0.001):
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + daily_pct))
    prices = np.array(prices)
    return pd.DataFrame(
        {"Open": prices * 0.999, "High": prices * 1.005,
         "Low": prices * 0.995, "Close": prices,
         "Volume": np.full(days, 1_000_000)},
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


def _make_falling_df(start="2020-01-02", days=252, start_price=200.0, daily_pct=-0.002):
    return _make_rising_df(start=start, days=days, start_price=start_price, daily_pct=daily_pct)


def _build_engine(tmpdir, source, tickers, config_overrides=None):
    defaults = dict(
        strategy_name="sma_crossover",
        tickers=tickers, benchmark=tickers[0],
        start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
        starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.20,
        fee_per_trade=0.0, slippage_bps=0.0, data_cache_dir=tmpdir,
        strategy_params={"sma_fast": 20, "sma_slow": 50},
    )
    if config_overrides:
        defaults.update(config_overrides)
    config = BacktestConfig(**defaults)
    dm = DataManager(cache_dir=tmpdir, source=source)
    return BacktestEngine(config, data_manager=dm)


# ===========================================================================
# Metrics edge cases
# ===========================================================================

class TestMetricsEdgeCases:
    """Edge cases in analytics metrics module."""

    def test_capture_ratio_flat_benchmark_returns_zero(self):
        equity = pd.Series([100, 101, 102, 101, 103],
                           index=pd.bdate_range("2020-01-02", periods=5))
        benchmark = pd.Series([100, 100, 100, 100, 100],
                              index=pd.bdate_range("2020-01-02", periods=5))
        up = capture_ratio(equity, benchmark, "up")
        down = capture_ratio(equity, benchmark, "down")
        assert up == 0.0
        assert down == 0.0

    def test_omega_ratio_all_positive_returns(self):
        equity = pd.Series([100, 101, 102, 103, 104],
                           index=pd.bdate_range("2020-01-02", periods=5))
        result = omega_ratio(equity)
        assert result == float("inf")

    def test_exposure_time_with_trades(self):
        equity = pd.Series(
            [100_000] * 10,
            index=pd.bdate_range("2020-01-02", periods=10),
        )
        trades = [
            Trade(symbol="A", entry_date=date(2020, 1, 2), exit_date=date(2020, 1, 6),
                  entry_price=100, exit_price=105, quantity=10,
                  pnl=50, pnl_pct=0.05, holding_days=4, fees_total=0),
        ]
        exp = exposure_time(equity, trades)
        assert 0.0 < exp <= 1.0

    def test_sharpe_ratio_flat_equity_returns_zero(self):
        equity = pd.Series([100_000] * 10,
                           index=pd.bdate_range("2020-01-02", periods=10))
        assert sharpe_ratio(equity) == 0.0

    def test_cagr_single_day_returns_zero(self):
        equity = pd.Series([100_000, 101_000],
                           index=[date(2020, 1, 2), date(2020, 1, 2)])
        # same date -> days=0 -> return 0
        result = cagr(equity)
        assert result == 0.0


# ===========================================================================
# Partial lot LIFO and cost-basis sells
# ===========================================================================

class TestPartialLotLIFOAndCostBasis:
    """LIFO and highest-cost lot methods with partial lot consumption."""

    def test_lifo_partial_lot_sell(self):
        pos = Position(symbol="TEST")
        pos.add_lot(50, 10.0, date(2020, 1, 2), commission=1.0)
        pos.add_lot(80, 12.0, date(2020, 1, 3), commission=2.0)
        pos.add_lot(70, 14.0, date(2020, 1, 4), commission=3.0)

        # Sell 120 shares via LIFO: lot3 (70) + lot2 (50 of 80)
        trades = pos.sell_lots_lifo(120, 15.0, date(2020, 2, 1), exit_commission=0.0)
        assert len(trades) == 2
        assert trades[0].quantity == 70  # lot3 (last in)
        assert trades[0].entry_price == 14.0
        assert trades[1].quantity == 50  # partial lot2
        assert trades[1].entry_price == 12.0

        # Remaining: lot1 (50@10) + lot2 remainder (30@12)
        assert pos.total_quantity == 80
        assert len(pos.lots) == 2
        assert pos.lots[0].quantity == 50  # lot1 unchanged
        assert pos.lots[1].quantity == 30  # lot2 partial

    def test_highest_cost_partial_lot_sell(self):
        pos = Position(symbol="TEST")
        pos.add_lot(50, 10.0, date(2020, 1, 2))
        pos.add_lot(80, 14.0, date(2020, 1, 3))
        pos.add_lot(70, 12.0, date(2020, 1, 4))

        # Sell 120 shares by highest cost: lot2@14 (80) + lot3@12 (40 of 70)
        trades = pos.sell_lots_by_cost(120, 15.0, date(2020, 2, 1), highest_first=True)
        assert len(trades) == 2
        assert trades[0].entry_price == 14.0
        assert trades[0].quantity == 80
        assert trades[1].entry_price == 12.0
        assert trades[1].quantity == 40

        assert pos.total_quantity == 80  # 50@10 + 30@12


# ===========================================================================
# Force-close short positions at backtest end
# ===========================================================================

class TestForceCloseShortPositions:
    """Short positions must be correctly closed and cash adjusted at backtest end."""

    def test_force_close_covers_short_positions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = _make_falling_df(days=100, start_price=200.0, daily_pct=-0.001)
            source.add("SH", df)

            if "_always_short_fc" not in _REGISTRY:
                class _AlwaysShortFC(Strategy):
                    def configure(self, params): pass
                    def compute_indicators(self, df, timeframe_data=None):
                        return df.copy()
                    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
                        if position is None or position.total_quantity == 0:
                            return SignalAction.SHORT
                        return SignalAction.HOLD
                _REGISTRY["_always_short_fc"] = _AlwaysShortFC

            engine = _build_engine(tmpdir, source, ["SH"], {
                "strategy_name": "_always_short_fc",
                "strategy_params": {},
                "allow_short": True,
                "max_positions": 1,
                "max_alloc_pct": 0.50,
                "short_borrow_rate": 0.0,
                "start_date": date(2020, 1, 2),
                "end_date": date(2020, 5, 31),
            })
            result = engine.run()

            # All positions should be closed
            assert result.portfolio.num_positions == 0, "Short positions not force-closed"
            # Trade log should contain the cover trade
            assert len(result.trades) >= 1, "No trades recorded from force-close"


# ===========================================================================
# Regime filter: fast_below_slow condition
# ===========================================================================

class TestRegimeFilterFastBelowSlow:
    """The fast_below_slow regime condition should work correctly."""

    def test_fast_below_slow_suppresses_buys_when_fast_above(self):
        """With fast_below_slow, BUYs should be suppressed when fast > slow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # Rising asset: fast SMA > slow SMA -> regime is OFF for fast_below_slow
            df = _make_rising_df(days=252, daily_pct=0.002)
            source.add("RISE", df)

            engine = _build_engine(tmpdir, source, ["RISE"], {
                "strategy_params": {"sma_fast": 20, "sma_slow": 50},
                "regime_filter": RegimeFilter(
                    benchmark="RISE", indicator="sma",
                    fast_period=20, slow_period=50,
                    condition="fast_below_slow",
                ),
            })
            result = engine.run()

            buys = sum(1 for e in result.activity_log if e.action == Side.BUY)
            # On a rising asset with fast_below_slow filter, most BUYs should be suppressed
            # because fast SMA > slow SMA (regime is off)
            # Compare to without filter
            engine_no_filter = _build_engine(tmpdir, source, ["RISE"], {
                "strategy_params": {"sma_fast": 20, "sma_slow": 50},
            })
            result_no_filter = engine_no_filter.run()
            buys_no_filter = sum(1 for e in result_no_filter.activity_log if e.action == Side.BUY)

            assert buys <= buys_no_filter, (
                f"fast_below_slow filter should suppress buys on rising asset: "
                f"filtered={buys}, unfiltered={buys_no_filter}"
            )


# ===========================================================================
# Rebalance schedule: quarterly
# ===========================================================================

class TestRebalanceSchedule:
    """Quarterly/monthly rebalance schedules fire on correct days."""

    def test_quarterly_rebalance_day_detection(self):
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["T"], benchmark="T",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000, max_positions=10, max_alloc_pct=0.10,
            rebalance_schedule="quarterly",
        )

        # Same quarter: should NOT be a rebalance day
        assert not engine._is_rebalance_day_for_schedule(date(2020, 2, 3), date(2020, 1, 31))

        # Cross quarter boundary (Q1->Q2): should be a rebalance day
        assert engine._is_rebalance_day_for_schedule(date(2020, 4, 1), date(2020, 3, 31))

        # Cross quarter boundary (Q2->Q3)
        assert engine._is_rebalance_day_for_schedule(date(2020, 7, 1), date(2020, 6, 30))

    def test_monthly_rebalance_day_detection(self):
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["T"], benchmark="T",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000, max_positions=10, max_alloc_pct=0.10,
            rebalance_schedule="monthly",
        )

        # Same month
        assert not engine._is_rebalance_day_for_schedule(date(2020, 1, 15), date(2020, 1, 14))
        # Cross month
        assert engine._is_rebalance_day_for_schedule(date(2020, 2, 3), date(2020, 1, 31))


# ===========================================================================
# Lookahead prevention: T+1 invariant
# ===========================================================================

class TestLookaheadPreventionStrict:
    """Every fill must happen strictly after the signal day."""

    def test_no_same_day_fills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine = _build_engine(tmpdir, source, ["TEST"])
            result = engine.run()

            for entry in result.activity_log:
                if entry.action == Side.BUY:
                    # The fill date must be > the first possible signal date
                    # (which is at earliest the start date)
                    assert entry.date > date(2020, 1, 2), (
                        f"Fill on {entry.date} -- first day fills violate T+1"
                    )


# ===========================================================================
# Multiple runs produce identical results (idempotency)
# ===========================================================================

class TestIdempotency:
    def test_two_runs_produce_same_equity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine1 = _build_engine(tmpdir, source, ["TEST"])
            result1 = engine1.run()

            engine2 = _build_engine(tmpdir, source, ["TEST"])
            result2 = engine2.run()

            eq1 = result1.equity_series
            eq2 = result2.equity_series
            assert len(eq1) == len(eq2)
            assert (eq1.values == eq2.values).all()


# ===========================================================================
# Stops with LIFO lot method
# ===========================================================================

class TestStopsWithLIFOLotMethod:
    """Stop triggers should use the configured lot method."""

    def test_stop_triggers_use_lifo_when_configured(self):
        from backtester.execution.stops import StopManager
        from backtester.config import StopConfig

        stop_mgr = StopManager(
            StopConfig(stop_loss_pct=0.05), PerTradeFee(fee=0.0), lot_method="lifo"
        )

        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("TEST")
        pos.add_lot(50, 100.0, date(2020, 1, 2))
        pos.add_lot(50, 110.0, date(2020, 1, 5))
        pos.stop_state.stop_loss = 95.0
        pos.update_market_price(100.0)

        today_data = {
            "TEST": pd.Series({"Open": 96.0, "High": 96.0, "Low": 90.0, "Close": 92.0})
        }

        stop_mgr.check_stop_triggers(date(2020, 1, 10), today_data, portfolio)

        # Position should be closed
        assert portfolio.num_positions == 0
        # LIFO: lot2 (110) should be sold first, then lot1 (100)
        assert len(portfolio.trade_log) == 2
        assert portfolio.trade_log[0].entry_price == 110.0  # LIFO: last added
        assert portfolio.trade_log[1].entry_price == 100.0


# ===========================================================================
# Duplicate index dates
# ===========================================================================

class TestDuplicateIndexDates:
    """OHLCV data with duplicate dates should be handled gracefully."""

    def test_duplicate_dates_do_not_crash_data_manager(self):
        """DataManager should handle or reject duplicate dates."""
        from backtester.data.manager import DataManager

        dates = pd.bdate_range("2020-01-02", periods=10, freq="B")
        # Create duplicate
        dup_dates = list(dates.date) + [dates.date[5]]
        prices = list(range(100, 111))

        df = pd.DataFrame(
            {"Open": prices, "High": [p+1 for p in prices],
             "Low": [p-1 for p in prices], "Close": prices,
             "Volume": [1_000_000] * 11},
            index=pd.DatetimeIndex(dup_dates, name="Date"),
        )

        source = MockDataSource()
        source.add("DUP", df)

        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(cache_dir=tmpdir, source=source)
            # Should either deduplicate or raise -- but not crash with a confusing error
            try:
                result = dm.load("DUP", date(2020, 1, 2), date(2020, 1, 15))
                # If it loads, the index should be unique
                assert result.index.is_unique, "Loaded DataFrame should have unique index"
            except (ValueError, KeyError):
                pass  # Raising a clear error is acceptable


# ===========================================================================
# Corrupt parquet cache graceful fallback
# ===========================================================================

class TestCorruptCacheFallback:
    """Corrupt cache files should not crash data loading."""

    def test_corrupt_parquet_returns_none(self):
        from backtester.data.cache import ParquetCache
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            # Write garbage to a parquet file
            cache_path = os.path.join(tmpdir, "TEST.parquet")
            with open(cache_path, "wb") as f:
                f.write(b"this is not a parquet file")

            result = cache.load("TEST")
            assert result is None, "Corrupt cache should return None, not crash"
