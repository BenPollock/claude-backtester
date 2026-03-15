"""Tests for slippage models."""

import math
from datetime import date

from backtester.execution.slippage import FixedSlippage, VolumeSlippage, SqrtImpactSlippage
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

    def test_extreme_bps_over_10000(self):
        """bps > 10000 means > 100% slippage; SELL price goes negative."""
        model = FixedSlippage(bps=15_000)  # 150%
        order = make_order(Side.SELL)
        price = model.compute(order, fill_price=100.0, volume=1_000_000)
        # slip = 100 * 1.5 = 150; sell price = 100 - 150 = -50
        assert price == -50.0

    def test_extreme_bps_buy(self):
        """bps > 10000 on BUY means fill price more than doubles."""
        model = FixedSlippage(bps=15_000)  # 150%
        order = make_order(Side.BUY)
        price = model.compute(order, fill_price=100.0, volume=1_000_000)
        # slip = 100 * 1.5 = 150; buy price = 100 + 150 = 250
        assert price == 250.0


class TestVolumeSlippageEdgeCases:
    def test_negative_volume_returns_original(self):
        """Negative volume should be treated like zero volume (no slippage)."""
        model = VolumeSlippage(impact_factor=0.1)
        order = make_order(Side.BUY, quantity=1000)
        price = model.compute(order, fill_price=100.0, volume=-5000)
        # volume <= 0 returns fill_price unchanged
        assert price == 100.0

    def test_quantity_zero(self):
        """Zero quantity should produce zero slippage."""
        model = VolumeSlippage(impact_factor=0.1)
        order = make_order(Side.BUY, quantity=0)
        price = model.compute(order, fill_price=100.0, volume=10_000)
        # ratio = 0/10000 = 0; slip = 0
        assert price == 100.0


class TestSqrtImpactSlippage:
    def test_basic_buy(self):
        """Verify basic square-root impact computation for BUY."""
        model = SqrtImpactSlippage(sigma=0.02, impact_factor=0.1)
        order = make_order(Side.BUY, quantity=10_000)
        price = model.compute(order, fill_price=100.0, volume=1_000_000)
        # impact = 0.1 * 0.02 * sqrt(10000/1000000) * 100 = 0.1 * 0.02 * 0.1 * 100 = 0.02
        expected = 100.0 + 0.1 * 0.02 * math.sqrt(10_000 / 1_000_000) * 100.0
        assert abs(price - expected) < 1e-10

    def test_basic_sell(self):
        """SELL should decrease the fill price."""
        model = SqrtImpactSlippage(sigma=0.02, impact_factor=0.1)
        order = make_order(Side.SELL, quantity=10_000)
        price = model.compute(order, fill_price=100.0, volume=1_000_000)
        expected = 100.0 - 0.1 * 0.02 * math.sqrt(10_000 / 1_000_000) * 100.0
        assert abs(price - expected) < 1e-10

    def test_zero_volume_no_impact(self):
        """Zero volume should return original price (no impact)."""
        model = SqrtImpactSlippage(sigma=0.02, impact_factor=0.1)
        order = make_order(Side.BUY, quantity=1000)
        price = model.compute(order, fill_price=50.0, volume=0)
        assert price == 50.0
