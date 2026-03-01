"""Tests for limit order execution (Feature 3C)."""

from datetime import date

import pandas as pd
import pytest

from backtester.types import SignalAction, Side, OrderType, OrderStatus
from backtester.portfolio.portfolio import Portfolio, PortfolioState
from backtester.portfolio.order import Order
from backtester.execution.broker import SimulatedBroker
from backtester.execution.slippage import FixedSlippage
from backtester.execution.fees import PerTradeFee
from backtester.strategies.base import Signal, Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_row(open_price=100.0, high=105.0, low=95.0, close=100.0, volume=1_000_000):
    """Build a single OHLCV row as a pd.Series."""
    return pd.Series({
        "Open": open_price, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


def make_broker(slippage_bps=0, fee=0.0):
    """Build a SimulatedBroker with configurable slippage/fees."""
    return SimulatedBroker(
        slippage=FixedSlippage(bps=slippage_bps),
        fees=PerTradeFee(fee=fee),
    )


# ---------------------------------------------------------------------------
# 1. Market order regression -- existing behavior unchanged
# ---------------------------------------------------------------------------

class TestMarketOrderRegression:
    """Market orders must continue to fill at Open, unchanged by limit work."""

    def test_market_buy_fills_at_open(self):
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3), {"AAPL": make_row(open_price=150.0)}, portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 150.0
        assert fills[0].quantity == 100
        assert portfolio.cash == 100_000.0 - 15_000.0

    def test_market_sell_fills_at_open(self):
        broker = make_broker()
        portfolio = Portfolio(cash=85_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=-1,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 3),
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6), {"AAPL": make_row(open_price=160.0)}, portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 160.0
        assert not portfolio.has_position("AAPL")

    def test_market_order_has_no_limit_fields(self):
        """A plain market order should have default limit-related fields."""
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=10,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
        )
        assert order.limit_price is None
        assert order.time_in_force == "DAY"
        assert order.expiry_date is None
        assert order.days_pending == 0


# ---------------------------------------------------------------------------
# 2. BUY limit orders
# ---------------------------------------------------------------------------

class TestBuyLimitOrder:
    def test_buy_limit_fills_when_low_at_or_below_limit(self):
        """BUY limit at 97 fills when Low=95 (<= 97). Fill at limit price."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=97.0,
        )
        broker.submit_order(order)

        # Low=95 is below limit=97, so the order fills at 97
        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 97.0
        assert fills[0].quantity == 100
        assert portfolio.cash == pytest.approx(100_000.0 - 97.0 * 100)

    def test_buy_limit_fills_when_low_equals_limit(self):
        """BUY limit at 95 fills when Low=95 (exactly at limit)."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=50,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=95.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 95.0

    def test_buy_limit_does_not_fill_when_low_above_limit(self):
        """BUY limit at 90 does NOT fill when Low=95 (> 90)."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=90.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 0
        # DAY order should be cancelled
        assert order.status == OrderStatus.CANCELLED

    def test_buy_limit_position_created(self):
        """A filled BUY limit order should create a position."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=97.0,
        )
        broker.submit_order(order)

        broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert portfolio.has_position("AAPL")
        assert portfolio.positions["AAPL"].total_quantity == 100


# ---------------------------------------------------------------------------
# 3. SELL limit orders
# ---------------------------------------------------------------------------

class TestSellLimitOrder:
    def test_sell_limit_fills_when_high_at_or_above_limit(self):
        """SELL limit at 103 fills when High=105 (>= 103). Fill at 103."""
        broker = make_broker()
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=103.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 103.0
        assert fills[0].quantity == 100
        assert not portfolio.has_position("AAPL")

    def test_sell_limit_fills_when_high_equals_limit(self):
        """SELL limit at 105 fills when High=105 (exactly at limit)."""
        broker = make_broker()
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=105.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 105.0

    def test_sell_limit_does_not_fill_when_high_below_limit(self):
        """SELL limit at 110 does NOT fill when High=105 (< 110)."""
        broker = make_broker()
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=110.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED


# ---------------------------------------------------------------------------
# 4. DAY order expiry
# ---------------------------------------------------------------------------

class TestDayOrderExpiry:
    def test_day_buy_limit_cancelled_if_not_filled(self):
        """DAY limit order that doesn't fill is cancelled, not kept pending."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=90.0,
            time_in_force="DAY",
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED
        assert len(broker.pending_orders) == 0

    def test_day_sell_limit_cancelled_if_not_filled(self):
        """DAY sell limit order cancelled when high doesn't reach limit."""
        broker = make_broker()
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=110.0,
            time_in_force="DAY",
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED
        assert len(broker.pending_orders) == 0


# ---------------------------------------------------------------------------
# 5. GTC orders -- persistence across days
# ---------------------------------------------------------------------------

class TestGTCOrder:
    def test_gtc_order_persists_when_not_filled(self):
        """GTC limit order stays in pending_orders if limit not reached."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=90.0,
            time_in_force="GTC",
        )
        broker.submit_order(order)

        # Day 1: Low=95, limit=90 not reached
        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 0
        assert order.status == OrderStatus.PENDING
        assert len(broker.pending_orders) == 1

    def test_gtc_order_fills_on_later_day(self):
        """GTC order that doesn't fill day 1 fills on day 2 when price drops."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=90.0,
            time_in_force="GTC",
        )
        broker.submit_order(order)

        # Day 1: Low=95, limit not reached
        broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )
        assert len(broker.pending_orders) == 1

        # Day 2: Low=88, limit 90 IS reached
        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=92.0, high=94.0, low=88.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 90.0
        assert fills[0].quantity == 100
        assert order.status == OrderStatus.FILLED
        assert len(broker.pending_orders) == 0

    def test_gtc_sell_persists_and_fills(self):
        """GTC sell limit that doesn't fill day 1, fills day 2."""
        broker = make_broker()
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=110.0,
            time_in_force="GTC",
        )
        broker.submit_order(order)

        # Day 1: High=105, limit 110 not reached
        broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )
        assert order.status == OrderStatus.PENDING
        assert len(broker.pending_orders) == 1

        # Day 2: High=112, limit 110 reached
        fills = broker.process_fills(
            date(2020, 1, 7),
            {"AAPL": make_row(open_price=108.0, high=112.0, low=106.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 110.0
        assert order.status == OrderStatus.FILLED
        assert not portfolio.has_position("AAPL")


# ---------------------------------------------------------------------------
# 6. GTC with expiry date
# ---------------------------------------------------------------------------

class TestGTCExpiry:
    def test_gtc_order_cancelled_after_expiry(self):
        """GTC order with expiry_date is cancelled when date passes expiry."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=90.0,
            time_in_force="GTC",
            expiry_date=date(2020, 1, 6),
        )
        broker.submit_order(order)

        # Day 1 (Jan 3): not filled, order persists
        broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )
        assert len(broker.pending_orders) == 1

        # Day 2 (Jan 6 = expiry date): still no fill, should be cancelled
        broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )
        assert order.status == OrderStatus.CANCELLED
        assert len(broker.pending_orders) == 0

    def test_gtc_order_fills_before_expiry(self):
        """GTC order fills when price reached before expiry."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=90.0,
            time_in_force="GTC",
            expiry_date=date(2020, 1, 10),
        )
        broker.submit_order(order)

        # Day 1 (Jan 3): Low=88 reaches limit=90
        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=92.0, high=94.0, low=88.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 90.0
        assert order.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# 7. days_pending tracking
# ---------------------------------------------------------------------------

class TestDaysPending:
    def test_days_pending_increments_for_gtc(self):
        """days_pending should increment each day a GTC order doesn't fill."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=85.0,
            time_in_force="GTC",
        )
        broker.submit_order(order)

        assert order.days_pending == 0

        # Day 1: not filled
        broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )
        assert order.days_pending == 1

        # Day 2: not filled
        broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )
        assert order.days_pending == 2

        # Day 3: filled (Low=80 reaches limit=85)
        fills = broker.process_fills(
            date(2020, 1, 7),
            {"AAPL": make_row(open_price=90.0, high=92.0, low=80.0)},
            portfolio,
        )
        assert len(fills) == 1
        assert order.days_pending == 2  # doesn't increment on fill day

    def test_days_pending_stays_zero_for_day_orders(self):
        """DAY orders that don't fill are cancelled, not incremented."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=85.0,
            time_in_force="DAY",
        )
        broker.submit_order(order)

        broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert order.days_pending == 0
        assert order.status == OrderStatus.CANCELLED


# ---------------------------------------------------------------------------
# 8. Slippage on limit orders
# ---------------------------------------------------------------------------

class TestLimitOrderSlippage:
    def test_slippage_applied_to_buy_limit(self):
        """Slippage is applied around the limit price, not the Open."""
        broker = make_broker(slippage_bps=100)  # 1% slippage
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=10,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=97.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        # Slippage on BUY: limit_price + 1% = 97 + 0.97 = 97.97
        assert fills[0].price == pytest.approx(97.97)
        assert fills[0].slippage == pytest.approx(0.97)

    def test_slippage_applied_to_sell_limit(self):
        """Slippage is applied around the limit price for sell limit."""
        broker = make_broker(slippage_bps=100)  # 1% slippage
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(10, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=10,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=103.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        # Slippage on SELL: limit_price - 1% = 103 - 1.03 = 101.97
        assert fills[0].price == pytest.approx(101.97)
        assert fills[0].slippage == pytest.approx(1.03)


# ---------------------------------------------------------------------------
# 9. Fees on limit orders
# ---------------------------------------------------------------------------

class TestLimitOrderFees:
    def test_fees_applied_to_buy_limit(self):
        """Commission should be deducted from cash on limit buy fill."""
        broker = make_broker(fee=5.0)
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=97.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].commission == 5.0
        assert portfolio.cash == pytest.approx(100_000.0 - 97.0 * 100 - 5.0)

    def test_fees_applied_to_sell_limit(self):
        """Commission should be deducted from proceeds on limit sell fill."""
        broker = make_broker(fee=5.0)
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=103.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].commission == 5.0
        assert portfolio.cash == pytest.approx(90_000.0 + 103.0 * 100 - 5.0)


# ---------------------------------------------------------------------------
# 10. Short selling + limit order combo
# ---------------------------------------------------------------------------

class TestShortWithLimitOrder:
    def test_short_entry_limit_order(self):
        """Short entry via limit order (SELL side, reason='short_entry')."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=103.0,
            reason="short_entry",
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 103.0
        assert portfolio.has_position("AAPL")
        assert portfolio.positions["AAPL"].total_quantity == -100
        assert portfolio.positions["AAPL"].is_short is True
        # Cash: received short sale proceeds minus nothing (fee=0)
        assert portfolio.cash == pytest.approx(100_000.0 + 103.0 * 100)

    def test_cover_limit_order(self):
        """Cover short via limit order (BUY side, reason='cover')."""
        broker = make_broker()
        portfolio = Portfolio(cash=200_000.0)
        # Set up a short position
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 110.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=-1,  # sentinel
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=97.0,
            reason="cover",
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].price == 97.0
        assert fills[0].quantity == 100  # sentinel resolved
        assert not portfolio.has_position("AAPL")
        # Cash: pay to buy back 100 shares at 97
        assert portfolio.cash == pytest.approx(200_000.0 - 97.0 * 100)

    def test_short_entry_limit_not_reached(self):
        """Short entry limit at 110 doesn't fill when High=105."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=110.0,
            reason="short_entry",
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED
        assert not portfolio.has_position("AAPL")


# ---------------------------------------------------------------------------
# 11. Signal dataclass -- strategy return type
# ---------------------------------------------------------------------------

class TestSignalDataclass:
    def test_signal_default_values(self):
        """Signal has sensible defaults for optional fields."""
        sig = Signal(action=SignalAction.BUY)
        assert sig.action == SignalAction.BUY
        assert sig.limit_price is None
        assert sig.time_in_force == "DAY"
        assert sig.expiry_date is None

    def test_signal_with_limit_price(self):
        """Signal can carry limit order parameters."""
        sig = Signal(
            action=SignalAction.BUY,
            limit_price=95.0,
            time_in_force="GTC",
            expiry_date=date(2020, 2, 1),
        )
        assert sig.limit_price == 95.0
        assert sig.time_in_force == "GTC"
        assert sig.expiry_date == date(2020, 2, 1)

    def test_signal_is_frozen(self):
        """Signal instances are immutable (frozen dataclass)."""
        sig = Signal(action=SignalAction.SELL)
        with pytest.raises(AttributeError):
            sig.action = SignalAction.BUY  # type: ignore[misc]

    def test_signal_equality_with_action(self):
        """Signal.action should be comparable to SignalAction values."""
        sig = Signal(action=SignalAction.BUY)
        assert sig.action == SignalAction.BUY
        assert sig.action != SignalAction.SELL


# ---------------------------------------------------------------------------
# 12. Backward compatibility -- existing strategies still work
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_strategy_returning_signal_action_works(self):
        """Strategies that return plain SignalAction should still work."""

        class LegacyStrategy(Strategy):
            def compute_indicators(self, df, timeframe_data=None):
                return df.copy()

            def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
                return SignalAction.BUY  # plain enum, not Signal

        strategy = LegacyStrategy()
        row = make_row()
        portfolio_state = PortfolioState(
            cash=100_000, total_equity=100_000, num_positions=0,
            position_symbols=frozenset(),
        )
        result = strategy.generate_signals("AAPL", row, None, portfolio_state)
        assert result == SignalAction.BUY
        assert isinstance(result, SignalAction)

    def test_strategy_returning_signal_object_works(self):
        """Strategies returning a Signal object should be accepted."""

        class LimitStrategy(Strategy):
            def compute_indicators(self, df, timeframe_data=None):
                return df.copy()

            def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
                return Signal(
                    action=SignalAction.BUY,
                    limit_price=95.0,
                    time_in_force="GTC",
                )

        strategy = LimitStrategy()
        row = make_row()
        portfolio_state = PortfolioState(
            cash=100_000, total_equity=100_000, num_positions=0,
            position_symbols=frozenset(),
        )
        result = strategy.generate_signals("AAPL", row, None, portfolio_state)
        assert isinstance(result, Signal)
        assert result.action == SignalAction.BUY
        assert result.limit_price == 95.0


# ---------------------------------------------------------------------------
# 13. Limit order with missing High/Low data
# ---------------------------------------------------------------------------

class TestLimitOrderEdgeCases:
    def test_limit_buy_uses_open_when_low_missing(self):
        """If Low is missing from the row, treat it as Open for limit check."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=100.0,
        )
        broker.submit_order(order)

        # Row with only Open (no High/Low)
        row = pd.Series({"Open": 100.0, "Close": 100.0, "Volume": 1_000_000})
        fills = broker.process_fills(date(2020, 1, 3), {"AAPL": row}, portfolio)

        # Low defaults to Open=100; limit=100 should fill (Low <= limit)
        assert len(fills) == 1
        assert fills[0].price == 100.0

    def test_limit_order_none_limit_price_treated_as_market(self):
        """LIMIT order_type with limit_price=None fills at Open (market)."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=None,  # no actual limit price set
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0)},
            portfolio,
        )

        # Should fill at Open like a market order
        assert len(fills) == 1
        assert fills[0].price == 100.0

    def test_insufficient_cash_reduces_limit_buy_quantity(self):
        """If cash is tight, limit BUY should reduce qty to fit budget."""
        broker = make_broker()
        portfolio = Portfolio(cash=500.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=97.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].quantity == 5  # 500 // 97 = 5 shares
        assert fills[0].price == 97.0

    def test_sell_all_sentinel_with_limit(self):
        """Sell-all sentinel (-1) should resolve correctly for limit orders."""
        broker = make_broker()
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(75, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=-1,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=103.0,
        )
        broker.submit_order(order)

        fills = broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(fills) == 1
        assert fills[0].quantity == 75  # resolved from sentinel
        assert fills[0].price == 103.0
        assert not portfolio.has_position("AAPL")

    def test_no_market_data_keeps_limit_order_pending(self):
        """Limit order with no market data for the symbol stays pending."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=97.0,
            time_in_force="GTC",
        )
        broker.submit_order(order)

        # No AAPL data
        fills = broker.process_fills(
            date(2020, 1, 3), {"MSFT": make_row()}, portfolio,
        )

        assert len(fills) == 0
        assert order.status == OrderStatus.PENDING
        assert len(broker.pending_orders) == 1


# ---------------------------------------------------------------------------
# 14. Activity log entries for limit order fills
# ---------------------------------------------------------------------------

class TestLimitOrderActivityLog:
    def test_buy_limit_creates_activity_log_entry(self):
        """A filled buy limit order should produce an activity log entry."""
        broker = make_broker()
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=97.0,
        )
        broker.submit_order(order)
        broker.process_fills(
            date(2020, 1, 3),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(portfolio.activity_log) == 1
        entry = portfolio.activity_log[0]
        assert entry.symbol == "AAPL"
        assert entry.action == Side.BUY
        assert entry.quantity == 100
        assert entry.price == 97.0

    def test_sell_limit_creates_activity_log_entry(self):
        """A filled sell limit order should produce an activity log entry."""
        broker = make_broker()
        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 95.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 3),
            limit_price=103.0,
        )
        broker.submit_order(order)
        broker.process_fills(
            date(2020, 1, 6),
            {"AAPL": make_row(open_price=100.0, high=105.0, low=95.0)},
            portfolio,
        )

        assert len(portfolio.activity_log) == 1
        entry = portfolio.activity_log[0]
        assert entry.symbol == "AAPL"
        assert entry.action == Side.SELL
        assert entry.quantity == 100
        assert entry.price == 103.0
