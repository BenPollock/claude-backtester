"""Tests for the activity log (TradeLogEntry) feature."""

import csv
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from backtester.execution.broker import SimulatedBroker
from backtester.execution.slippage import FixedSlippage
from backtester.execution.fees import PerTradeFee
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.order import Order, TradeLogEntry
from backtester.types import Side, OrderType


def make_row(open_price=100.0, close=100.0, volume=1_000_000):
    return pd.Series({"Open": open_price, "High": 105.0, "Low": 95.0, "Close": close, "Volume": volume})


def _broker_no_costs():
    return SimulatedBroker(slippage=FixedSlippage(bps=0), fees=PerTradeFee(fee=0))


class TestActivityLogBuy:
    def test_buy_creates_entry(self):
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=50,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 3), {"AAPL": make_row(open_price=200.0)}, portfolio)

        assert len(portfolio.activity_log) == 1
        entry = portfolio.activity_log[0]
        assert entry.action == Side.BUY
        assert entry.symbol == "AAPL"
        assert entry.quantity == 50
        assert entry.price == 200.0
        assert entry.avg_cost_basis is None

    def test_buy_value_equals_qty_times_price(self):
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=30,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 3), {"AAPL": make_row(open_price=150.0)}, portfolio)

        entry = portfolio.activity_log[0]
        assert entry.value == entry.quantity * entry.price

    def test_buy_records_fees_and_slippage(self):
        broker = SimulatedBroker(slippage=FixedSlippage(bps=100), fees=PerTradeFee(fee=5.0))
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=10,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 3), {"AAPL": make_row(open_price=100.0)}, portfolio)

        entry = portfolio.activity_log[0]
        assert entry.fees == 5.0
        assert entry.slippage == 1.0  # 100 bps of 100 = 1.0


class TestActivityLogSell:
    def test_sell_creates_entry_with_cost_basis(self):
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=50_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))

        order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 6), {"AAPL": make_row(open_price=160.0)}, portfolio)

        assert len(portfolio.activity_log) == 1
        entry = portfolio.activity_log[0]
        assert entry.action == Side.SELL
        assert entry.quantity == 100
        assert entry.price == 160.0
        assert entry.avg_cost_basis == 150.0

    def test_sell_value_equals_qty_times_price(self):
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=50_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(40, 100.0, date(2020, 1, 2))

        order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 6), {"AAPL": make_row(open_price=120.0)}, portfolio)

        entry = portfolio.activity_log[0]
        assert entry.value == entry.quantity * entry.price

    def test_sell_cost_basis_is_weighted_avg(self):
        """With multiple lots, avg_cost_basis should be the weighted average."""
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=50_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(60, 100.0, date(2020, 1, 2))
        pos.add_lot(40, 200.0, date(2020, 1, 3))
        # Weighted avg = (60*100 + 40*200) / 100 = 140.0

        order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 6))
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 7), {"AAPL": make_row(open_price=150.0)}, portfolio)

        entry = portfolio.activity_log[0]
        assert entry.avg_cost_basis == 140.0


class TestActivityLogCSVExport:
    def test_csv_export(self):
        from backtester.analytics.report import export_activity_log_csv
        from backtester.config import BacktestConfig
        from backtester.engine import BacktestResult

        portfolio = Portfolio(cash=100_000.0)
        portfolio.activity_log.append(TradeLogEntry(
            date=date(2020, 1, 3), symbol="AAPL", action=Side.BUY,
            quantity=50, price=200.0, value=10_000.0,
            avg_cost_basis=None, fees=5.0, slippage=0.2,
        ))
        portfolio.activity_log.append(TradeLogEntry(
            date=date(2020, 2, 3), symbol="AAPL", action=Side.SELL,
            quantity=50, price=220.0, value=11_000.0,
            avg_cost_basis=200.0, fees=5.0, slippage=0.22,
        ))
        portfolio.equity_history = [(date(2020, 1, 3), 100_000.0), (date(2020, 2, 3), 101_000.0)]

        config = BacktestConfig(
            strategy_name="test", tickers=["AAPL"], benchmark="SPY",
            start_date=date(2020, 1, 1), end_date=date(2020, 12, 31),
            starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
        )
        result = BacktestResult(config=config, portfolio=portfolio)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "activity.csv")
            export_activity_log_csv(result, filepath)

            with open(filepath) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["symbol"] == "AAPL"
            assert rows[0]["action"] == "BUY"
            assert rows[0]["avg_cost_basis"] == ""
            assert rows[1]["action"] == "SELL"
            assert rows[1]["avg_cost_basis"] == "200.0000"


class TestActivityLogShortEntry:
    """Activity log entries for SHORT entry orders."""

    def test_short_entry_creates_sell_activity(self):
        """A short entry (reason='short_entry') should log a SELL activity
        with avg_cost_basis=None (since it's an opening trade, not a close).
        """
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.SELL, quantity=50,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
                      reason="short_entry")
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 3), {"AAPL": make_row(open_price=150.0)}, portfolio)

        assert len(portfolio.activity_log) == 1
        entry = portfolio.activity_log[0]
        assert entry.action == Side.SELL
        assert entry.symbol == "AAPL"
        assert entry.quantity == 50
        assert entry.price == 150.0
        assert entry.avg_cost_basis is None  # opening trade, no cost basis
        assert entry.value == 50 * 150.0

    def test_short_entry_portfolio_state(self):
        """After a short entry, portfolio should have a short position and
        increased cash (from selling borrowed shares).
        """
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.SELL, quantity=50,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
                      reason="short_entry")
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 3), {"AAPL": make_row(open_price=150.0)}, portfolio)

        assert portfolio.has_position("AAPL")
        assert portfolio.positions["AAPL"].is_short
        assert portfolio.positions["AAPL"].total_quantity == -50
        assert portfolio.cash == 100_000.0 + 50 * 150.0


class TestActivityLogCoverExit:
    """Activity log entries for COVER (buy-to-close short) orders."""

    def test_cover_creates_buy_activity_with_cost_basis(self):
        """A cover (reason='cover') should log a BUY activity with
        avg_cost_basis set to the short entry price.
        """
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)
        # Set up short position manually
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))

        order = Order(symbol="AAPL", side=Side.BUY, quantity=-1,  # sentinel: cover all
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3),
                      reason="cover")
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 6), {"AAPL": make_row(open_price=140.0)}, portfolio)

        assert len(portfolio.activity_log) == 1
        entry = portfolio.activity_log[0]
        assert entry.action == Side.BUY
        assert entry.symbol == "AAPL"
        assert entry.quantity == 100
        assert entry.price == 140.0
        assert entry.avg_cost_basis == 150.0  # original short entry price
        assert entry.value == 100 * 140.0

    def test_cover_removes_position(self):
        """After covering all shares, position should be removed."""
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))

        order = Order(symbol="AAPL", side=Side.BUY, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3),
                      reason="cover")
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 6), {"AAPL": make_row(open_price=140.0)}, portfolio)

        assert not portfolio.has_position("AAPL")


class TestActivityLogOrdering:
    """Activity log entries should appear in chronological order."""

    def test_activity_log_chronological_order(self):
        """Multiple fills across different dates should produce activity log
        entries in chronological order.
        """
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)

        # Day 1: BUY AAPL
        order1 = Order(symbol="AAPL", side=Side.BUY, quantity=50,
                       order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order1)
        broker.process_fills(date(2020, 1, 3), {"AAPL": make_row(open_price=100.0)}, portfolio)

        # Day 2: BUY MSFT
        order2 = Order(symbol="MSFT", side=Side.BUY, quantity=20,
                       order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order2)
        broker.process_fills(date(2020, 1, 6), {"MSFT": make_row(open_price=200.0)}, portfolio)

        # Day 3: SELL AAPL
        order3 = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                       order_type=OrderType.MARKET, signal_date=date(2020, 1, 6))
        broker.submit_order(order3)
        broker.process_fills(date(2020, 1, 7), {"AAPL": make_row(open_price=110.0)}, portfolio)

        assert len(portfolio.activity_log) == 3
        dates = [e.date for e in portfolio.activity_log]
        assert dates == sorted(dates), "Activity log entries must be in chronological order"
        assert portfolio.activity_log[0].action == Side.BUY
        assert portfolio.activity_log[0].symbol == "AAPL"
        assert portfolio.activity_log[1].action == Side.BUY
        assert portfolio.activity_log[1].symbol == "MSFT"
        assert portfolio.activity_log[2].action == Side.SELL
        assert portfolio.activity_log[2].symbol == "AAPL"


# ---------------------------------------------------------------------------
# Coverage-expanding tests
# ---------------------------------------------------------------------------


class TestActivityLogCSVExportEdgeCases:
    """Edge cases for CSV export of activity log."""

    def test_csv_export_empty_log(self):
        """Empty activity log should produce CSV with header only."""
        from backtester.analytics.report import export_activity_log_csv
        from backtester.config import BacktestConfig
        from backtester.engine import BacktestResult

        portfolio = Portfolio(cash=100_000.0)
        portfolio.equity_history = [(date(2020, 1, 3), 100_000.0)]

        config = BacktestConfig(
            strategy_name="test", tickers=["AAPL"], benchmark="SPY",
            start_date=date(2020, 1, 1), end_date=date(2020, 12, 31),
            starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.10,
        )
        result = BacktestResult(config=config, portfolio=portfolio)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "activity.csv")
            export_activity_log_csv(result, filepath)

            with open(filepath) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 0


class TestActivityLogMultipleFillsSameDay:
    """Multiple fills on the same date should all appear in the log."""

    def test_two_buys_same_day(self):
        """Two BUY orders filled on the same date produce two log entries."""
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)

        order1 = Order(symbol="AAPL", side=Side.BUY, quantity=10,
                       order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        order2 = Order(symbol="MSFT", side=Side.BUY, quantity=20,
                       order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order1)
        broker.submit_order(order2)

        market_data = {
            "AAPL": make_row(open_price=100.0),
            "MSFT": make_row(open_price=200.0),
        }
        broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(portfolio.activity_log) == 2
        # Both on same date
        assert portfolio.activity_log[0].date == date(2020, 1, 3)
        assert portfolio.activity_log[1].date == date(2020, 1, 3)

    def test_buy_then_sell_same_day(self):
        """A BUY and SELL on different symbols on the same fill day."""
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))

        buy_order = Order(symbol="MSFT", side=Side.BUY, quantity=10,
                          order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        sell_order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                           order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(buy_order)
        broker.submit_order(sell_order)

        market_data = {
            "AAPL": make_row(open_price=110.0),
            "MSFT": make_row(open_price=200.0),
        }
        broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        assert len(portfolio.activity_log) == 2
        actions = [e.action for e in portfolio.activity_log]
        assert Side.BUY in actions
        assert Side.SELL in actions


class TestActivityLogPartialSell:
    """Partial sell should record only the sold quantity."""

    def test_partial_sell_logs_correct_quantity(self):
        """Selling fewer shares than held should log only the sold amount."""
        broker = _broker_no_costs()
        portfolio = Portfolio(cash=50_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))

        order = Order(symbol="AAPL", side=Side.SELL, quantity=30,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order)
        broker.process_fills(date(2020, 1, 6), {"AAPL": make_row(open_price=120.0)}, portfolio)

        assert len(portfolio.activity_log) == 1
        entry = portfolio.activity_log[0]
        assert entry.quantity == 30
        assert entry.price == 120.0
        # Position still has 70 shares
        assert portfolio.positions["AAPL"].total_quantity == 70
