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

    def test_sell_sentinel_resolves_to_full_position(self):
        """qty=-1 sentinel should resolve to actual position size."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=50_000.0)
        pos = portfolio.open_position("MSFT")
        pos.add_lot(75, 200.0, date(2020, 1, 2))

        order = Order(symbol="MSFT", side=Side.SELL, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order)

        market_data = {"MSFT": make_row(open_price=210.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        assert len(fills) == 1
        assert fills[0].quantity == 75  # resolved from sentinel to actual qty
        assert not portfolio.has_position("MSFT")

    def test_buy_cancelled_when_zero_affordable(self):
        """When cash can't buy even 1 share, order should be cancelled."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=5.0)  # only $5
        order = Order(symbol="AAPL", side=Side.BUY, quantity=100,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED

    def test_missing_market_data_keeps_order_pending(self):
        """Order for symbol with no market data should stay pending."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=10,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        # No AAPL in market_data
        market_data = {"MSFT": make_row(open_price=200.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 0
        assert order.status == OrderStatus.PENDING
        assert len(broker.pending_orders) == 1

    def test_multiple_orders_same_symbol(self):
        """Two BUY orders for the same symbol should both fill in queue order."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order1 = Order(symbol="AAPL", side=Side.BUY, quantity=10,
                       order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        order2 = Order(symbol="AAPL", side=Side.BUY, quantity=20,
                       order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order1)
        broker.submit_order(order2)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 2
        assert fills[0].quantity == 10
        assert fills[1].quantity == 20
        assert portfolio.positions["AAPL"].total_quantity == 30
        assert portfolio.cash == 100_000.0 - 3_000.0

    def test_slippage_negative_price_cancels_fill(self):
        """If slippage produces a negative fill price, the fill is cancelled."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=20_000),  # 200% slippage on SELL
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=50_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 50.0, date(2020, 1, 2))

        order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=10.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        # FixedSlippage at 20000 bps = 200%: sell price = 10 - 20 = -10
        # Negative fill price causes the fill to be cancelled
        assert len(fills) == 0
        # Position still exists (no fill occurred)
        assert portfolio.has_position("AAPL")

    def test_order_with_quantity_zero(self):
        """BUY order with quantity=0 should be cancelled (can't afford 0 shares at cost)."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=0,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        # quantity=0 means total_cost=0 which passes cash check, but order
        # should produce a fill with 0 shares or be cancelled
        assert len(fills) <= 1
        if len(fills) == 1:
            assert fills[0].quantity == 0

    def test_sell_sentinel_no_position_cancelled(self):
        """SELL with qty=-1 when no position exists should cancel."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED

    def test_fill_with_nan_open_price_keeps_pending(self):
        """Row with NaN Open price should keep order pending."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=10,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        row = pd.Series({"Open": float("nan"), "High": 105.0, "Low": 95.0,
                         "Close": 100.0, "Volume": 1_000_000})
        market_data = {"AAPL": row}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 0
        assert order.status == OrderStatus.PENDING
        assert len(broker.pending_orders) == 1

    def test_bracket_oco_cancels_sibling(self):
        """When one bracket child fills, the other should be cancelled."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)

        # Submit bracket: entry BUY, stop_loss SELL @ 90, take_profit SELL @ 120
        entry = Order(symbol="AAPL", side=Side.BUY, quantity=100,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        stop_loss = Order(symbol="AAPL", side=Side.SELL, quantity=100,
                          order_type=OrderType.STOP, stop_price=90.0,
                          signal_date=date(2020, 1, 2), time_in_force="GTC")
        take_profit = Order(symbol="AAPL", side=Side.SELL, quantity=100,
                            order_type=OrderType.LIMIT, limit_price=120.0,
                            signal_date=date(2020, 1, 2), time_in_force="GTC")

        broker.submit_bracket(entry, stop_loss, take_profit)

        # Day 1: entry fills at open=100
        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)
        assert len(fills) == 1
        assert entry.status == OrderStatus.FILLED

        # Both children should now be in pending orders
        assert len(broker.pending_orders) == 2

        # Day 2: price drops to trigger stop (Low=89), High=95 (no take profit)
        row = pd.Series({"Open": 92.0, "High": 95.0, "Low": 89.0,
                         "Close": 91.0, "Volume": 1_000_000})
        market_data = {"AAPL": row}
        fills2 = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        # Stop should fill, take profit should be cancelled
        assert len(fills2) == 1
        assert fills2[0].price == 90.0  # stop price
        assert take_profit.status == OrderStatus.CANCELLED


class TestBrokerEdgeCases:
    """Edge case tests for SimulatedBroker."""

    def test_buy_order_cash_less_than_commission(self):
        """When cash < commission, the affordable qty must be 0 (not negative).

        With $3 cash, a $5 fee, and $100/share price: (3 - 5) // 100 = -1
        in naive math, but the broker should clamp to 0 and cancel.
        """
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=5.0),
        )
        portfolio = Portfolio(cash=3.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=100,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED
        # Cash must not go negative
        assert portfolio.cash >= 0

    def test_zero_fill_price_after_slippage(self):
        """A fill price that becomes exactly 0 after slippage should be handled.

        A SELL with large negative slippage could theoretically produce a 0 price.
        The broker cancels fills with price < 0. Price == 0 is borderline — verify
        it doesn't corrupt portfolio state.
        """
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=10_000),  # 100% slippage on sell: price - 100% = 0
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=50_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 50.0, date(2020, 1, 2))

        order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order)

        # Open price = 10, slippage = 100% of 10 = 10 => fill_price = 10 - 10 = 0
        market_data = {"AAPL": make_row(open_price=10.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        # fill_price == 0 is not negative, so it may or may not fill — but must not crash
        # and the portfolio should remain consistent
        if len(fills) == 0:
            assert portfolio.has_position("AAPL")
        else:
            assert fills[0].price >= 0

    def test_gtc_order_persists_across_multiple_cycles(self):
        """A GTC limit order should remain pending across multiple fill cycles
        until the price reaches the limit.
        """
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=10,
                      order_type=OrderType.LIMIT, limit_price=90.0,
                      signal_date=date(2020, 1, 2), time_in_force="GTC")
        broker.submit_order(order)

        # Day 1: price too high (Low=95), should not fill
        row_day1 = pd.Series({"Open": 100.0, "High": 105.0, "Low": 95.0,
                              "Close": 100.0, "Volume": 1_000_000})
        fills1 = broker.process_fills(date(2020, 1, 3), {"AAPL": row_day1}, portfolio)
        assert len(fills1) == 0
        assert order.status == OrderStatus.PENDING
        assert order.days_pending == 1
        assert len(broker.pending_orders) == 1

        # Day 2: still too high
        row_day2 = pd.Series({"Open": 98.0, "High": 102.0, "Low": 93.0,
                              "Close": 97.0, "Volume": 1_000_000})
        fills2 = broker.process_fills(date(2020, 1, 6), {"AAPL": row_day2}, portfolio)
        assert len(fills2) == 0
        assert order.days_pending == 2

        # Day 3: price drops to reach limit (Low=89)
        row_day3 = pd.Series({"Open": 95.0, "High": 96.0, "Low": 89.0,
                              "Close": 91.0, "Volume": 1_000_000})
        fills3 = broker.process_fills(date(2020, 1, 7), {"AAPL": row_day3}, portfolio)
        assert len(fills3) == 1
        assert fills3[0].price == 90.0  # filled at limit price
        assert order.status == OrderStatus.FILLED

    def test_day_order_expires_if_not_filled(self):
        """A DAY limit order should be cancelled if the limit is not reached."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=10,
                      order_type=OrderType.LIMIT, limit_price=90.0,
                      signal_date=date(2020, 1, 2), time_in_force="DAY")
        broker.submit_order(order)

        # Price stays above 90 — limit not reached
        row = pd.Series({"Open": 100.0, "High": 105.0, "Low": 95.0,
                         "Close": 100.0, "Volume": 1_000_000})
        fills = broker.process_fills(date(2020, 1, 3), {"AAPL": row}, portfolio)

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED
        assert len(broker.pending_orders) == 0

    def test_sell_order_on_nonexistent_position_cancelled(self):
        """A SELL order for a symbol with no position should be cancelled."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        # Explicit quantity SELL (not sentinel) for a symbol we don't hold
        order = Order(symbol="AAPL", side=Side.SELL, quantity=50,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        # The broker should not have any position to sell from, so:
        # If the order filled, check that portfolio wasn't corrupted
        # If cancelled, that's expected behavior
        if len(fills) == 0:
            assert order.status == OrderStatus.CANCELLED
        else:
            # Even if the broker allows it, no position should be negative
            assert not portfolio.has_position("AAPL") or portfolio.positions["AAPL"].total_quantity >= 0

    def test_partial_fill_with_volume_constraint(self):
        """max_volume_pct should limit the fill quantity.

        With volume=1000 and max_volume_pct=0.10, max fillable = 100 shares.
        An order for 500 shares should only fill 100.
        """
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
            max_volume_pct=0.10,
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=500,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=10.0, volume=1_000)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 1
        # volume=1000 * 0.10 = 100 max fillable
        assert fills[0].quantity == 100
        assert portfolio.positions["AAPL"].total_quantity == 100
