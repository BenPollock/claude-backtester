"""Tests for slippage models."""

from datetime import date

from backtester.execution.slippage import FixedSlippage, VolumeSlippage
from backtester.portfolio.order import Order
from backtester.types import Side, OrderType


def make_order(side, quantity=100):
    return Order(
        symbol="TEST", side=side, quantity=quantity,
        order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
    )


class TestVolumeSlippage:
    def test_buy_price_increases(self):
        model = VolumeSlippage(impact_factor=0.1)
        order = make_order(Side.BUY, quantity=1000)
        price = model.compute(order, fill_price=100.0, volume=10_000)
        # slip = 100 * 0.1 * (1000/10000) = 1.0
        assert price == 101.0

    def test_sell_price_decreases(self):
        model = VolumeSlippage(impact_factor=0.1)
        order = make_order(Side.SELL, quantity=1000)
        price = model.compute(order, fill_price=100.0, volume=10_000)
        # slip = 100 * 0.1 * (1000/10000) = 1.0
        assert price == 99.0

    def test_zero_volume_returns_original_price(self):
        model = VolumeSlippage(impact_factor=0.1)
        order = make_order(Side.BUY, quantity=100)
        price = model.compute(order, fill_price=50.0, volume=0)
        assert price == 50.0


class TestFixedSlippage:
    def test_sell_direction(self):
        model = FixedSlippage(bps=100)  # 1% = 100 bps
        order = make_order(Side.SELL)
        price = model.compute(order, fill_price=100.0, volume=1_000_000)
        assert price == 99.0
