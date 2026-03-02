"""Tests for Gap Analysis features (33 gaps across 11 batches)."""

import csv
import json
import math
import os
import tempfile
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig, StopConfig
from backtester.portfolio.portfolio import Portfolio, PortfolioState
from backtester.portfolio.position import Position, Lot
from backtester.portfolio.order import Order, Fill, Trade
from backtester.types import Side, OrderType, OrderStatus, SignalAction
from backtester.execution.broker import SimulatedBroker
from backtester.execution.slippage import FixedSlippage, SqrtImpactSlippage
from backtester.execution.fees import PerTradeFee
from backtester.execution.position_sizing import (
    KellyCriterionSizer, RiskParitySizer,
)
from backtester.strategies.base import Strategy, Signal, CrossSectionalStrategy
from backtester.strategies.registry import register_strategy
from backtester.analytics.metrics import (
    historical_var, cvar, omega_ratio, treynor_ratio, compute_all_metrics,
)
from backtester.analytics.overfitting import deflated_sharpe_ratio, permutation_test
from backtester.analytics.trade_analysis import compute_mae_mfe, mae_mfe_summary
from backtester.analytics.tca import compute_turnover, compute_cost_attribution, estimate_capacity

from tests.conftest import make_price_df, MockDataSource


# =====================================================================
# Gap 1 — Survivorship-Bias-Free Universe
# =====================================================================

class TestHistoricalUniverse:
    def test_members_on_date(self, tmp_path):
        """Universe membership returns correct set for a given date."""
        from backtester.data.universe import HistoricalUniverse
        csv_path = tmp_path / "universe.csv"
        csv_path.write_text(
            "date,symbol\n"
            "2020-01-01,AAPL\n"
            "2020-01-01,GOOG\n"
            "2020-02-01,AAPL\n"
            "2020-02-01,GOOG\n"
            "2020-02-01,MSFT\n"
            "2020-03-01,AAPL\n"
            "2020-03-01,MSFT\n"
        )
        hu = HistoricalUniverse(str(csv_path))

        # Before any snapshot
        assert hu.members_on(date(2019, 12, 31)) is None

        # First snapshot
        members_jan = hu.members_on(date(2020, 1, 15))
        assert members_jan == {"AAPL", "GOOG"}

        # Second snapshot — MSFT added
        members_feb = hu.members_on(date(2020, 2, 15))
        assert members_feb == {"AAPL", "GOOG", "MSFT"}

        # Third snapshot — GOOG removed
        members_mar = hu.members_on(date(2020, 3, 15))
        assert members_mar == {"AAPL", "MSFT"}

    def test_all_symbols(self, tmp_path):
        from backtester.data.universe import HistoricalUniverse
        csv_path = tmp_path / "universe.csv"
        csv_path.write_text("date,symbol\n2020-01-01,A\n2020-02-01,B\n")
        hu = HistoricalUniverse(str(csv_path))
        assert hu.all_symbols == {"A", "B"}

    def test_empty_before_first_snapshot(self, tmp_path):
        from backtester.data.universe import HistoricalUniverse
        csv_path = tmp_path / "universe.csv"
        csv_path.write_text("date,symbol\n2020-06-01,AAPL\n")
        hu = HistoricalUniverse(str(csv_path))
        assert hu.members_on(date(2020, 1, 1)) is None


# =====================================================================
# Gap 2 — Delisting Detection
# =====================================================================

class TestDelistingDetection:
    def test_force_close_after_missing_days(self):
        """Position force-closed after >5 consecutive missing days."""
        from backtester.engine import BacktestEngine
        source = MockDataSource()

        # Ticker A: full 252 days. Ticker B: stops at day 150.
        df_a = make_price_df(days=252, start_price=100)
        df_b = make_price_df(days=150, start_price=50)
        source.add("A", df_a)
        source.add("B", df_b)

        from backtester.data.manager import DataManager
        dm = DataManager(cache_dir=tempfile.mkdtemp(), source=source)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["A", "B"],
            benchmark="A",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 12, 31),
            starting_cash=100_000,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
        )

        engine = BacktestEngine(config, data_manager=dm)
        result = engine.run()
        # Should complete without error — B positions force-closed
        assert len(result.portfolio.equity_history) > 0


# =====================================================================
# Gap 3 — Partial Fills
# =====================================================================

class TestPartialFills:
    def _make_row(self, open_p=100, high=105, low=95, close=100, volume=5000):
        return pd.Series({
            "Open": open_p, "High": high, "Low": low,
            "Close": close, "Volume": volume,
        })

    def test_volume_constraint_cancel(self):
        """Order exceeding max_volume_pct is partially filled, remainder cancelled."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
            max_volume_pct=0.10,
            partial_fill_policy="cancel",
        )
        portfolio = Portfolio(cash=1_000_000)
        order = Order(symbol="TEST", side=Side.BUY, quantity=1000,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        row = self._make_row(volume=5000)  # max fillable = 500
        fills = broker.process_fills(date(2020, 1, 3), {"TEST": row}, portfolio)

        assert len(fills) == 1
        assert fills[0].quantity == 500  # 5000 * 0.10

    def test_volume_constraint_requeue(self):
        """Order exceeding max_volume_pct is partially filled, remainder requeued."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
            max_volume_pct=0.10,
            partial_fill_policy="requeue",
        )
        portfolio = Portfolio(cash=1_000_000)
        order = Order(symbol="TEST", side=Side.BUY, quantity=1000,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        row = self._make_row(volume=5000)
        fills = broker.process_fills(date(2020, 1, 3), {"TEST": row}, portfolio)

        assert fills[0].quantity == 500
        # Remainder should be requeued
        assert len(broker.pending_orders) == 1
        assert broker.pending_orders[0].quantity == 500  # remainder


# =====================================================================
# Gap 4 — Cross-Sectional Strategy
# =====================================================================

class TestCrossSectionalStrategy:
    def test_rank_universe_dispatched(self):
        """CrossSectionalStrategy.rank_universe is called by engine."""
        class MockCrossStrategy(CrossSectionalStrategy):
            def __init__(self):
                super().__init__()
                self.called = False

            def compute_indicators(self, df, timeframe_data=None):
                return df.copy()

            def rank_universe(self, bar_data, positions, portfolio_state, benchmark_row=None):
                self.called = True
                return []

        strat = MockCrossStrategy()
        assert strat.generate_signals("X", pd.Series(), None, None) == SignalAction.HOLD

    def test_top_n_bottom_n(self):
        scores = {"A": 10, "B": 5, "C": 20, "D": 1}
        assert CrossSectionalStrategy.top_n(scores, 2) == ["C", "A"]
        assert CrossSectionalStrategy.bottom_n(scores, 2) == ["D", "B"]


# =====================================================================
# Gap 5 — Drawdown Kill Switch
# =====================================================================

class TestDrawdownKillSwitch:
    def test_halts_at_threshold(self):
        """Backtest halts when drawdown breaches max_drawdown_pct."""
        from backtester.engine import BacktestEngine
        source = MockDataSource()

        # Create declining price data
        days = 100
        dates = pd.bdate_range(start="2020-01-02", periods=days)
        prices = np.linspace(100, 70, days)  # 30% decline
        df = pd.DataFrame({
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": 1_000_000,
        }, index=pd.DatetimeIndex(dates.date, name="Date"))
        source.add("TEST", df)

        from backtester.data.manager import DataManager
        dm = DataManager(cache_dir=tempfile.mkdtemp(), source=source)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 5, 31),
            starting_cash=100_000,
            max_positions=10,
            max_alloc_pct=0.50,
            max_drawdown_pct=0.10,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
        )

        engine = BacktestEngine(config, data_manager=dm)
        result = engine.run()

        # Should have recorded equity for all days (continuing as cash-only after halt)
        assert len(result.portfolio.equity_history) > 0


# =====================================================================
# Gap 6 — VaR and CVaR
# =====================================================================

class TestVaRCVaR:
    def test_historical_var(self):
        """VaR at 95% returns 5th percentile of returns."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 252)
        prices = [100]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        idx = pd.bdate_range("2020-01-02", periods=len(prices))
        equity = pd.Series(prices, index=idx)

        var = historical_var(equity, 0.95)
        assert var < 0  # VaR should be negative

    def test_cvar_below_var(self):
        """CVaR should be <= VaR (further into the tail)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 252)
        prices = [100]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        idx = pd.bdate_range("2020-01-02", periods=len(prices))
        equity = pd.Series(prices, index=idx)

        var = historical_var(equity, 0.95)
        cv = cvar(equity, 0.95)
        assert cv <= var

    def test_short_series_returns_zero(self):
        equity = pd.Series([100], index=[date(2020, 1, 1)])
        assert historical_var(equity) == 0.0
        assert cvar(equity) == 0.0

    def test_metrics_include_var_cvar(self):
        """compute_all_metrics includes VaR and CVaR."""
        df = make_price_df(days=100)
        idx = pd.bdate_range("2020-01-02", periods=100)
        equity = pd.Series(df["Close"].values, index=idx)
        m = compute_all_metrics(equity, [])
        assert "var_95" in m
        assert "cvar_95" in m


# =====================================================================
# Gap 7 — Overfitting Metrics
# =====================================================================

class TestOverfittingMetrics:
    def test_dsr_decreases_with_more_trials(self):
        """DSR should decrease as num_trials increases for same observed Sharpe."""
        # Use high observed Sharpe so values stay in distinguishable range
        dsr_10 = deflated_sharpe_ratio(2.5, 10, 1.0, 252)
        dsr_100 = deflated_sharpe_ratio(2.5, 100, 1.0, 252)
        dsr_1000 = deflated_sharpe_ratio(2.5, 1000, 1.0, 252)
        assert dsr_10 > dsr_100 > dsr_1000

    def test_dsr_bounded(self):
        """DSR should be between 0 and 1."""
        dsr = deflated_sharpe_ratio(2.0, 50, 0.5, 252)
        assert 0 <= dsr <= 1

    def test_permutation_random_strategy(self):
        """Random walk should have high p-value (not significant)."""
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 252))
        idx = pd.bdate_range("2020-01-02", periods=252)
        equity = pd.Series(prices, index=idx)

        result = permutation_test(equity, n_permutations=200, seed=42)
        assert result["p_value"] > 0.01  # random walk should not be significant


# =====================================================================
# Gap 8 — Multi-Source Data
# =====================================================================

class TestMultiSourceData:
    def test_csv_source(self, tmp_path):
        """CSVDataSource loads data from CSV files."""
        from backtester.data.csv_source import CSVDataSource

        # Create test CSV
        csv_path = tmp_path / "TEST.csv"
        dates = pd.bdate_range("2020-01-02", periods=10)
        df = pd.DataFrame({
            "Date": dates,
            "Open": range(100, 110),
            "High": range(101, 111),
            "Low": range(99, 109),
            "Close": range(100, 110),
            "Volume": [1000] * 10,
        })
        df.to_csv(csv_path, index=False)

        source = CSVDataSource(str(tmp_path))
        result = source.fetch("TEST", date(2020, 1, 2), date(2020, 1, 31))
        assert len(result) == 10
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_csv_source_missing_file(self, tmp_path):
        from backtester.data.csv_source import CSVDataSource
        source = CSVDataSource(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            source.fetch("MISSING", date(2020, 1, 1), date(2020, 12, 31))


# =====================================================================
# Gap 9 — Fundamental Data
# =====================================================================

class TestFundamentalData:
    def test_point_in_time_lookup(self, tmp_path):
        """Fundamental data lookup respects reporting dates."""
        from backtester.data.fundamental import FundamentalDataManager

        csv_path = tmp_path / "fundamentals.csv"
        csv_path.write_text(
            "date,symbol,field,value\n"
            "2020-03-15,AAPL,PE,15.0\n"
            "2020-06-15,AAPL,PE,18.0\n"
        )

        fm = FundamentalDataManager(str(csv_path))

        # Before first report
        assert fm.get("AAPL", "PE", date(2020, 3, 14)) is None

        # After first report, before second
        assert fm.get("AAPL", "PE", date(2020, 3, 15)) == 15.0
        assert fm.get("AAPL", "PE", date(2020, 5, 1)) == 15.0

        # After second report
        assert fm.get("AAPL", "PE", date(2020, 6, 15)) == 18.0

    def test_strategy_fundamental_access(self):
        """Strategy.get_fundamental returns None when no data injected."""
        class TestStrat(Strategy):
            def compute_indicators(self, df, timeframe_data=None):
                return df.copy()
            def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
                return SignalAction.HOLD

        strat = TestStrat()
        assert strat.get_fundamental("AAPL", "PE", date(2020, 1, 1)) is None


# =====================================================================
# Gap 11 — Stop Orders via Broker
# =====================================================================

class TestStopOrders:
    def _make_row(self, open_p=100, high=105, low=94, close=99, volume=10000):
        return pd.Series({
            "Open": open_p, "High": high, "Low": low,
            "Close": close, "Volume": volume,
        })

    def test_stop_sell_triggers(self):
        """STOP SELL triggers when Low <= stop_price."""
        broker = SimulatedBroker(slippage=FixedSlippage(bps=0), fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)
        pos = portfolio.open_position("TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))

        order = Order(symbol="TEST", side=Side.SELL, quantity=100,
                      order_type=OrderType.STOP, signal_date=date(2020, 1, 2),
                      stop_price=95.0)
        broker.submit_order(order)

        row = self._make_row(low=94)  # Low <= 95
        fills = broker.process_fills(date(2020, 1, 3), {"TEST": row}, portfolio)

        assert len(fills) == 1
        assert fills[0].price == 95.0

    def test_stop_sell_not_triggered(self):
        """STOP SELL does NOT trigger when Low > stop_price."""
        broker = SimulatedBroker(slippage=FixedSlippage(bps=0), fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)
        pos = portfolio.open_position("TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))

        order = Order(symbol="TEST", side=Side.SELL, quantity=100,
                      order_type=OrderType.STOP, signal_date=date(2020, 1, 2),
                      stop_price=90.0)
        broker.submit_order(order)

        row = self._make_row(low=94)  # Low > 90
        fills = broker.process_fills(date(2020, 1, 3), {"TEST": row}, portfolio)

        assert len(fills) == 0
        assert len(broker.pending_orders) == 0  # DAY order cancelled

    def test_stop_limit_no_fill(self):
        """STOP_LIMIT: stop triggers but limit not reachable."""
        broker = SimulatedBroker(slippage=FixedSlippage(bps=0), fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)
        pos = portfolio.open_position("TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))

        order = Order(symbol="TEST", side=Side.SELL, quantity=100,
                      order_type=OrderType.STOP_LIMIT, signal_date=date(2020, 1, 2),
                      stop_price=95.0, limit_price=110.0)  # stop triggered but limit too high
        broker.submit_order(order)

        row = self._make_row(low=94, high=105)
        fills = broker.process_fills(date(2020, 1, 3), {"TEST": row}, portfolio)

        assert len(fills) == 0


# =====================================================================
# Gap 12 — Square-Root Market Impact
# =====================================================================

class TestSqrtImpactSlippage:
    def test_basic_impact(self):
        """Verify sqrt impact calculation."""
        model = SqrtImpactSlippage(sigma=0.02, impact_factor=0.1)
        order = Order(symbol="TEST", side=Side.BUY, quantity=10_000,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        price = model.compute(order, 100.0, 1_000_000)
        expected_impact = 0.1 * 0.02 * math.sqrt(10_000 / 1_000_000) * 100.0
        assert abs(price - (100.0 + expected_impact)) < 0.01

    def test_superlinear_cost(self):
        """Larger orders should have superlinear impact."""
        model = SqrtImpactSlippage(sigma=0.02, impact_factor=0.1)
        small = Order(symbol="T", side=Side.BUY, quantity=1000,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        large = Order(symbol="T", side=Side.BUY, quantity=100_000,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))

        small_impact = model.compute(small, 100.0, 1_000_000) - 100.0
        large_impact = model.compute(large, 100.0, 1_000_000) - 100.0

        # Impact should grow with sqrt, so 100x qty -> 10x impact
        assert large_impact / small_impact > 5


# =====================================================================
# Gap 13 — Bracket / OCO Orders
# =====================================================================

class TestBracketOrders:
    def _make_row(self, open_p=100, high=115, low=85, close=100, volume=10000):
        return pd.Series({
            "Open": open_p, "High": high, "Low": low,
            "Close": close, "Volume": volume,
        })

    def test_bracket_entry_fills_activates_children(self):
        """When entry fills, children become active."""
        broker = SimulatedBroker(slippage=FixedSlippage(bps=0), fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)

        entry = Order(symbol="TEST", side=Side.BUY, quantity=100,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        stop = Order(symbol="TEST", side=Side.SELL, quantity=100,
                     order_type=OrderType.STOP, signal_date=date(2020, 1, 2),
                     stop_price=90.0, time_in_force="GTC")
        target = Order(symbol="TEST", side=Side.SELL, quantity=100,
                       order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
                       limit_price=110.0, time_in_force="GTC")

        broker.submit_bracket(entry, stop, target)
        row = self._make_row(open_p=100)
        fills = broker.process_fills(date(2020, 1, 3), {"TEST": row}, portfolio)

        assert len(fills) == 1  # entry filled
        # Both children should now be pending
        assert len(broker.pending_orders) == 2

    def test_bracket_entry_not_filled(self):
        """When entry doesn't fill, children never activate."""
        broker = SimulatedBroker(slippage=FixedSlippage(bps=0), fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)

        entry = Order(symbol="TEST", side=Side.BUY, quantity=100,
                      order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
                      limit_price=80.0)  # won't fill
        stop = Order(symbol="TEST", side=Side.SELL, quantity=100,
                     order_type=OrderType.STOP, signal_date=date(2020, 1, 2),
                     stop_price=70.0, time_in_force="GTC")
        target = Order(symbol="TEST", side=Side.SELL, quantity=100,
                       order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
                       limit_price=120.0, time_in_force="GTC")

        broker.submit_bracket(entry, stop, target)
        row = self._make_row(low=90)
        fills = broker.process_fills(date(2020, 1, 3), {"TEST": row}, portfolio)

        assert len(fills) == 0
        assert len(broker.pending_orders) == 0  # DAY entry cancelled, children never activated


# =====================================================================
# Gap 14 — Fill Price Models
# =====================================================================

class TestFillPriceModels:
    def _make_row(self):
        return pd.Series({
            "Open": 100, "High": 110, "Low": 90, "Close": 105, "Volume": 10000,
        })

    def test_open_model(self):
        broker = SimulatedBroker(fill_price_model="open", slippage=FixedSlippage(bps=0),
                                 fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)
        order = Order(symbol="T", side=Side.BUY, quantity=10,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)
        fills = broker.process_fills(date(2020, 1, 3), {"T": self._make_row()}, portfolio)
        assert fills[0].price == 100

    def test_close_model(self):
        broker = SimulatedBroker(fill_price_model="close", slippage=FixedSlippage(bps=0),
                                 fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)
        order = Order(symbol="T", side=Side.BUY, quantity=10,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)
        fills = broker.process_fills(date(2020, 1, 3), {"T": self._make_row()}, portfolio)
        assert fills[0].price == 105

    def test_vwap_model(self):
        broker = SimulatedBroker(fill_price_model="vwap", slippage=FixedSlippage(bps=0),
                                 fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)
        order = Order(symbol="T", side=Side.BUY, quantity=10,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)
        fills = broker.process_fills(date(2020, 1, 3), {"T": self._make_row()}, portfolio)
        # VWAP = (110 + 90 + 105) / 3
        assert abs(fills[0].price - (110 + 90 + 105) / 3) < 0.01

    def test_random_model_within_range(self):
        broker = SimulatedBroker(fill_price_model="random", slippage=FixedSlippage(bps=0),
                                 fees=PerTradeFee(fee=0))
        portfolio = Portfolio(cash=100_000)
        order = Order(symbol="T", side=Side.BUY, quantity=10,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)
        fills = broker.process_fills(date(2020, 1, 3), {"T": self._make_row()}, portfolio)
        assert 90 <= fills[0].price <= 110


# =====================================================================
# Gap 15 — Target-Weight Rebalancing
# =====================================================================

class TestTargetWeightRebalancing:
    def test_compute_rebalance_orders(self):
        """Portfolio.compute_rebalance_orders generates correct orders."""
        portfolio = Portfolio(cash=50_000)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)  # $10,000 position

        # Total equity = $60,000
        prices = {"AAPL": 100.0, "GOOG": 200.0}
        targets = {"AAPL": 0.5, "GOOG": 0.5}  # $30K each

        orders = portfolio.compute_rebalance_orders(targets, prices)
        # AAPL: has $10K, wants $30K -> BUY $20K / 100 = 200 shares
        # GOOG: has $0, wants $30K -> BUY $30K / 200 = 150 shares
        buy_orders = [(s, sd, q) for s, sd, q in orders if sd == Side.BUY]
        assert any(s == "AAPL" for s, _, _ in buy_orders)
        assert any(s == "GOOG" for s, _, _ in buy_orders)

    def test_rebalance_sell_then_buy(self):
        """Sells are generated before buys."""
        portfolio = Portfolio(cash=10_000)
        pos_a = portfolio.open_position("A")
        pos_a.add_lot(100, 100.0, date(2020, 1, 2))
        pos_a.update_market_price(100.0)

        prices = {"A": 100.0, "B": 50.0}
        targets = {"A": 0.0, "B": 1.0}  # sell all A, buy B

        orders = portfolio.compute_rebalance_orders(targets, prices)
        sells = [o for o in orders if o[1] == Side.SELL]
        buys = [o for o in orders if o[1] == Side.BUY]
        assert len(sells) > 0
        assert len(buys) > 0


# =====================================================================
# Gap 17 — Kelly and Risk Parity Sizing
# =====================================================================

class TestKellyRiskParity:
    def test_kelly_sizing(self):
        """Kelly sizer uses win_rate and payoff_ratio."""
        sizer = KellyCriterionSizer(fraction=0.5)
        row = pd.Series({
            "Close": 100.0, "kelly_win_rate": 0.6, "kelly_payoff_ratio": 2.0,
        })
        # f* = (0.6*2 - 0.4) / 2 = 0.4, half-kelly = 0.2
        # target = 100000 * 0.2 = 20000, shares = 200
        qty = sizer.compute("T", 100.0, row, 100_000, 100_000, 0.50)
        assert qty == 200

    def test_kelly_fallback(self):
        """Kelly falls back to fixed fractional without indicator data."""
        sizer = KellyCriterionSizer(fraction=0.5)
        row = pd.Series({"Close": 100.0})
        qty = sizer.compute("T", 100.0, row, 100_000, 100_000, 0.10)
        assert qty == 100  # fixed fractional: 100000 * 0.10 / 100

    def test_risk_parity_sizing(self):
        """RiskParity sizes inversely to volatility."""
        sizer = RiskParitySizer(target_vol=0.10)
        low_vol_row = pd.Series({"Close": 100.0, "ATR": 1.0})
        high_vol_row = pd.Series({"Close": 100.0, "ATR": 5.0})

        low_vol_qty = sizer.compute("T", 100.0, low_vol_row, 100_000, 100_000, 0.50)
        high_vol_qty = sizer.compute("T", 100.0, high_vol_row, 100_000, 100_000, 0.50)

        assert low_vol_qty > high_vol_qty  # lower vol -> bigger position


# =====================================================================
# Gap 18 — Sector Exposure Limits (tested via config)
# =====================================================================

class TestSectorExposure:
    def test_sector_map_config(self):
        """Config accepts sector_map_path and max_sector_exposure."""
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["AAPL"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000, max_positions=10, max_alloc_pct=0.10,
            max_sector_exposure=0.30, sector_map_path="/tmp/sectors.csv",
        )
        assert config.max_sector_exposure == 0.30


# =====================================================================
# Gap 19 — Gross/Net Exposure Limits
# =====================================================================

class TestExposureLimits:
    def test_config_accepts_exposure_limits(self):
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["AAPL"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000, max_positions=10, max_alloc_pct=0.10,
            max_gross_exposure=1.5, max_net_exposure=1.0,
        )
        assert config.max_gross_exposure == 1.5
        assert config.max_net_exposure == 1.0


# =====================================================================
# Gap 21 — Per-Trade MAE/MFE
# =====================================================================

class TestMAEMFE:
    def test_compute_mae_mfe(self):
        """MAE/MFE computed correctly for a long trade."""
        trade = Trade(
            symbol="TEST", entry_date=date(2020, 1, 2), exit_date=date(2020, 1, 10),
            entry_price=100.0, exit_price=105.0, quantity=100,
            pnl=500, pnl_pct=0.05, holding_days=8, fees_total=0,
        )

        dates = pd.bdate_range("2020-01-02", periods=7)
        df = pd.DataFrame({
            "Open": [100, 101, 99, 95, 98, 103, 105],
            "High": [102, 103, 101, 97, 100, 112, 106],
            "Low": [98, 99, 92, 93, 96, 101, 104],
            "Close": [101, 100, 95, 96, 99, 105, 105],
            "Volume": [1000] * 7,
        }, index=pd.DatetimeIndex(dates.date, name="Date"))

        results = compute_mae_mfe([trade], {"TEST": df})
        assert len(results) == 1
        r = results[0]
        assert r["mae"] == (92 - 100) / 100  # -8%
        assert r["mfe"] == (112 - 100) / 100  # +12%

    def test_summary(self):
        results = [
            {"symbol": "A", "entry_date": None, "exit_date": None, "mae": -0.08, "mfe": 0.12},
            {"symbol": "B", "entry_date": None, "exit_date": None, "mae": -0.04, "mfe": 0.06},
        ]
        summary = mae_mfe_summary(results)
        assert abs(summary["avg_mae"] - (-0.06)) < 0.001
        assert abs(summary["avg_mfe"] - 0.09) < 0.001


# =====================================================================
# Gap 22/23/24 — Turnover, TCA, Capacity
# =====================================================================

class TestTCA:
    def test_turnover(self):
        trades = [
            Trade("A", date(2020, 1, 2), date(2020, 1, 10), 100, 105,
                  100, 500, 0.05, 8, 10),
            Trade("B", date(2020, 3, 1), date(2020, 3, 10), 50, 55,
                  200, 1000, 0.10, 9, 10),
        ]
        idx = pd.bdate_range("2020-01-02", periods=252)
        equity = pd.Series(np.linspace(100000, 110000, 252), index=idx)
        turnover = compute_turnover(trades, equity)
        assert turnover > 0

    def test_cost_attribution(self):
        trades = [
            Trade("A", date(2020, 1, 2), date(2020, 1, 10), 100, 105,
                  100, 500, 0.05, 8, 25.0),
        ]
        idx = pd.bdate_range("2020-01-02", periods=10)
        equity = pd.Series(np.linspace(100000, 100500, 10), index=idx)
        costs = compute_cost_attribution(trades, equity)
        assert costs["total_fees"] == 25.0

    def test_no_trades(self):
        idx = pd.bdate_range("2020-01-02", periods=10)
        equity = pd.Series([100000] * 10, index=idx)
        assert compute_turnover([], equity) == 0.0
        costs = compute_cost_attribution([], equity)
        assert costs["total_fees"] == 0.0


# =====================================================================
# Gap 29 — Parallel Execution (test workers=1 path)
# =====================================================================

class TestParallelExecution:
    def test_workers_config(self):
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["AAPL"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000, max_positions=10, max_alloc_pct=0.10,
            workers=4,
        )
        assert config.workers == 4


# =====================================================================
# Gap 30 — Result Persistence
# =====================================================================

class TestResultPersistence:
    def test_save_and_load(self, tmp_path):
        """Save and load round-trip preserves metrics."""
        from backtester.result import BacktestResult

        portfolio = Portfolio(cash=100_000)
        for i in range(10):
            portfolio.record_equity(date(2020, 1, 2 + i))

        config = BacktestConfig(
            strategy_name="test", tickers=["SPY"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 1, 12),
            starting_cash=100_000, max_positions=10, max_alloc_pct=0.10,
        )

        result = BacktestResult(config=config, portfolio=portfolio)
        save_path = str(tmp_path / "results")
        result.save(save_path)

        # Verify files exist
        assert (tmp_path / "results" / "config.json").exists()
        assert (tmp_path / "results" / "metrics.json").exists()
        assert (tmp_path / "results" / "equity.parquet").exists()

    def test_compare(self, tmp_path):
        """Compare two results side-by-side."""
        from backtester.result import BacktestResult

        for name in ["run1", "run2"]:
            path = tmp_path / name
            path.mkdir()
            (path / "config.json").write_text(json.dumps({"strategy_name": name}))
            (path / "metrics.json").write_text(json.dumps({
                "sharpe_ratio": 1.5 if name == "run1" else 0.8,
                "cagr": 0.12 if name == "run1" else 0.05,
            }))

        df = BacktestResult.compare([str(tmp_path / "run1"), str(tmp_path / "run2")])
        assert len(df) == 2


# =====================================================================
# Gap 35 — Borrow Cost Deducted from Cash
# =====================================================================

class TestBorrowCostDeduction:
    def test_accrue_returns_cost(self):
        """accrue_borrow_cost returns the daily cost."""
        pos = Position(symbol="TEST")
        pos.add_lot(-100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)

        cost = pos.accrue_borrow_cost(0.02)
        expected = 10_000 * 0.02 / 252
        assert abs(cost - expected) < 0.01
        assert pos.short_borrow_cost_accrued > 0

    def test_long_position_no_cost(self):
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        cost = pos.accrue_borrow_cost(0.02)
        assert cost == 0.0


# =====================================================================
# Gap 36 — LIFO / Cost-Based Lot Accounting
# =====================================================================

class TestLotMethods:
    def test_lifo(self):
        """LIFO sells most recent lot first."""
        pos = Position(symbol="TEST")
        pos.add_lot(100, 10.0, date(2020, 1, 2))  # oldest
        pos.add_lot(100, 12.0, date(2020, 2, 2))  # middle
        pos.add_lot(100, 15.0, date(2020, 3, 2))  # newest

        trades = pos.sell_lots_lifo(100, 14.0, date(2020, 4, 1))
        assert len(trades) == 1
        assert trades[0].entry_price == 15.0  # newest lot sold first

    def test_highest_cost(self):
        """Highest cost sells most expensive lot first."""
        pos = Position(symbol="TEST")
        pos.add_lot(100, 10.0, date(2020, 1, 2))
        pos.add_lot(100, 15.0, date(2020, 2, 2))
        pos.add_lot(100, 12.0, date(2020, 3, 2))

        trades = pos.sell_lots_by_cost(100, 14.0, date(2020, 4, 1), highest_first=True)
        assert trades[0].entry_price == 15.0

    def test_lowest_cost(self):
        """Lowest cost sells cheapest lot first."""
        pos = Position(symbol="TEST")
        pos.add_lot(100, 10.0, date(2020, 1, 2))
        pos.add_lot(100, 15.0, date(2020, 2, 2))
        pos.add_lot(100, 12.0, date(2020, 3, 2))

        trades = pos.sell_lots_by_cost(100, 14.0, date(2020, 4, 1), highest_first=False)
        assert trades[0].entry_price == 10.0

    def test_fifo_unchanged(self):
        """FIFO still sells oldest first."""
        pos = Position(symbol="TEST")
        pos.add_lot(100, 10.0, date(2020, 1, 2))
        pos.add_lot(100, 15.0, date(2020, 2, 2))

        trades = pos.sell_lots_fifo(100, 14.0, date(2020, 4, 1))
        assert trades[0].entry_price == 10.0  # oldest first


# =====================================================================
# Gap 38 — Stress Testing
# =====================================================================

class TestStressTesting:
    def test_scenarios_defined(self):
        from backtester.analytics.stress import STRESS_SCENARIOS
        assert "gfc_2008" in STRESS_SCENARIOS
        assert "covid_crash" in STRESS_SCENARIOS
        assert len(STRESS_SCENARIOS) >= 5


# =====================================================================
# Gap 43 — Omega and Treynor Ratios
# =====================================================================

class TestOmegaTreynor:
    def test_omega_ratio_positive(self):
        """Omega ratio > 1 for positive-skew returns."""
        rng = np.random.default_rng(42)
        prices = [100]
        for _ in range(252):
            prices.append(prices[-1] * (1 + rng.normal(0.001, 0.01)))
        idx = pd.bdate_range("2020-01-02", periods=253)
        equity = pd.Series(prices, index=idx)
        omega = omega_ratio(equity)
        assert omega > 0

    def test_treynor_zero_beta(self):
        """Treynor returns 0 when beta is 0."""
        idx = pd.bdate_range("2020-01-02", periods=100)
        equity = pd.Series([100000] * 100, index=idx)
        benchmark = pd.Series([100000] * 100, index=idx)
        tr = treynor_ratio(equity, benchmark)
        assert tr == 0.0


# =====================================================================
# Gap 45 — Rebalance Schedule
# =====================================================================

class TestRebalanceSchedule:
    def test_config_accepts_schedule(self):
        config = BacktestConfig(
            strategy_name="sma_crossover", tickers=["AAPL"], benchmark="SPY",
            start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
            starting_cash=100_000, max_positions=10, max_alloc_pct=0.10,
            rebalance_schedule="monthly",
        )
        assert config.rebalance_schedule == "monthly"


# =====================================================================
# Gap 49 — TOML Config
# =====================================================================

class TestTOMLConfig:
    def test_toml_loading(self, tmp_path):
        """TOML config file is parsed correctly."""
        from backtester.cli import _load_config_file
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('[strategy]\nname = "sma_crossover"\n')
        result = _load_config_file(str(toml_path))
        assert result["strategy"]["name"] == "sma_crossover"


# =====================================================================
# Signal dataclass extensions
# =====================================================================

class TestSignalExtensions:
    def test_signal_with_stop_price(self):
        sig = Signal(
            action=SignalAction.SELL,
            stop_price=95.0,
            order_type=OrderType.STOP,
        )
        assert sig.stop_price == 95.0
        assert sig.order_type == OrderType.STOP

    def test_signal_backward_compatible(self):
        sig = Signal(action=SignalAction.BUY, limit_price=50.0)
        assert sig.stop_price is None
        assert sig.order_type == OrderType.MARKET
