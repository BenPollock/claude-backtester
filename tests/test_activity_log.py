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
