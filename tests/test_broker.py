"""Tests for SimulatedBroker."""

from datetime import date

import pandas as pd

from backtester.execution.broker import SimulatedBroker
from backtester.execution.slippage import FixedSlippage
from backtester.execution.fees import PerTradeFee
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.order import Order
from backtester.types import Side, OrderType, OrderStatus


def make_row(open_price=100.0, close=100.0, volume=1_000_000):
    return pd.Series({"Open": open_price, "High": 105.0, "Low": 95.0, "Close": close, "Volume": volume})


class TestSimulatedBroker:
    def test_buy_fill(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),  # no slippage for simplicity
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=100,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=150.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 1
        assert fills[0].symbol == "AAPL"
        assert fills[0].quantity == 100
        assert fills[0].price == 150.0
        assert portfolio.cash == 100_000.0 - 15_000.0
        assert portfolio.has_position("AAPL")
        assert portfolio.positions["AAPL"].total_quantity == 100

    def test_sell_fill(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=85_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))

        order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,  # sell all
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=160.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        assert len(fills) == 1
        assert fills[0].quantity == 100
        assert fills[0].price == 160.0
        assert portfolio.cash == 85_000.0 + 16_000.0
        assert not portfolio.has_position("AAPL")

    def test_insufficient_cash_reduces_quantity(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=500.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=100,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 1
        assert fills[0].quantity == 5  # 500 / 100 = 5 shares
        assert portfolio.positions["AAPL"].total_quantity == 5

    def test_slippage_applied(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=100),  # 1% slippage
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=10,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert fills[0].price == 101.0  # 100 + 1% = 101
        assert fills[0].slippage == 1.0

    def test_fees_deducted(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=5.0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=100,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert fills[0].commission == 5.0
        assert portfolio.cash == 100_000.0 - 10_000.0 - 5.0
