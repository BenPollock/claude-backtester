"""Tests for extended fee models: PercentageFee, TieredFee, SECFee, TAFFee, CompositeFee."""

import pytest
from datetime import date

from backtester.execution.fees import (
    PercentageFee,
    TieredFee,
    SECFee,
    TAFFee,
    CompositeFee,
    PerTradeFee,
)
from backtester.portfolio.order import Order
from backtester.types import Side, OrderType


def make_order(side: Side = Side.BUY, quantity: int = 100) -> Order:
    """Helper to create a test order."""
    return Order(
        symbol="TEST",
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        signal_date=date(2024, 1, 2),
    )


# ---------- PercentageFee ----------


class TestPercentageFee:
    def test_basic_calculation(self):
        """5 bps on $10,000 notional = $5.00."""
        model = PercentageFee(bps=5)
        order = make_order(Side.BUY, quantity=100)
        fee = model.compute(order, fill_price=100.0, quantity=100)
        # notional = 100 * 100 = 10,000; fee = 10,000 * 5 / 10,000 = 5.0
        assert fee == pytest.approx(5.0)

    def test_zero_bps(self):
        """Zero basis points should produce zero fee."""
        model = PercentageFee(bps=0)
        order = make_order(Side.BUY, quantity=500)
        fee = model.compute(order, fill_price=200.0, quantity=500)
        assert fee == 0.0

    def test_various_notional_values(self):
        """Fee scales linearly with notional value."""
        model = PercentageFee(bps=10)
        order = make_order(Side.BUY, quantity=1)

        fee_small = model.compute(order, fill_price=50.0, quantity=1)
        fee_large = model.compute(order, fill_price=50.0, quantity=1000)

        # fee_small = 50 * 10 / 10,000 = 0.05
        assert fee_small == pytest.approx(0.05)
        # fee_large = 50,000 * 10 / 10,000 = 50.0
        assert fee_large == pytest.approx(50.0)
        # Linear: 1000x quantity -> 1000x fee
        assert fee_large == pytest.approx(fee_small * 1000)

    def test_sell_order(self):
        """PercentageFee applies to both buy and sell orders."""
        model = PercentageFee(bps=5)
        order = make_order(Side.SELL, quantity=200)
        fee = model.compute(order, fill_price=50.0, quantity=200)
        # notional = 200 * 50 = 10,000; fee = 10,000 * 5 / 10,000 = 5.0
        assert fee == pytest.approx(5.0)

    def test_zero_quantity(self):
        """Zero quantity should produce zero fee."""
        model = PercentageFee(bps=10)
        order = make_order(Side.BUY, quantity=0)
        fee = model.compute(order, fill_price=100.0, quantity=0)
        assert fee == 0.0

    def test_zero_price(self):
        """Zero fill price should produce zero fee."""
        model = PercentageFee(bps=10)
        order = make_order(Side.BUY, quantity=100)
        fee = model.compute(order, fill_price=0.0, quantity=100)
        assert fee == 0.0

    def test_default_bps(self):
        """Default bps is 5."""
        model = PercentageFee()
        order = make_order(Side.BUY, quantity=100)
        fee = model.compute(order, fill_price=100.0, quantity=100)
        assert fee == pytest.approx(5.0)


# ---------- TieredFee ----------


class TestTieredFee:
    def test_amount_in_first_tier(self):
        """Notional entirely within the first tier."""
        tiers = [(0, 10), (10_000, 5), (100_000, 2)]
        model = TieredFee(tiers)
        order = make_order(Side.BUY, quantity=50)
        # notional = 50 * 100 = 5,000 (entirely in 0-10k tier at 10 bps)
        fee = model.compute(order, fill_price=100.0, quantity=50)
        assert fee == pytest.approx(5_000 * 10 / 10_000)  # 5.0

    def test_crossing_tier_boundary(self):
        """Notional spans the first two tiers."""
        tiers = [(0, 10), (10_000, 5), (100_000, 2)]
        model = TieredFee(tiers)
        order = make_order(Side.BUY, quantity=200)
        # notional = 200 * 100 = 20,000
        # First 10,000 at 10 bps = 10.0
        # Next 10,000 at 5 bps = 5.0
        # Total = 15.0
        fee = model.compute(order, fill_price=100.0, quantity=200)
        assert fee == pytest.approx(15.0)

    def test_very_large_amount_all_tiers(self):
        """Notional spans all three tiers."""
        tiers = [(0, 10), (10_000, 5), (100_000, 2)]
        model = TieredFee(tiers)
        order = make_order(Side.BUY, quantity=5000)
        # notional = 5000 * 100 = 500,000
        # 0-10k at 10 bps = 10.0
        # 10k-100k at 5 bps = 45.0
        # 100k-500k at 2 bps = 80.0
        # Total = 135.0
        fee = model.compute(order, fill_price=100.0, quantity=5000)
        assert fee == pytest.approx(135.0)

    def test_exact_tier_boundary(self):
        """Notional exactly at a tier boundary."""
        tiers = [(0, 10), (10_000, 5)]
        model = TieredFee(tiers)
        order = make_order(Side.BUY, quantity=100)
        # notional = 100 * 100 = 10,000 (exactly fills first tier)
        fee = model.compute(order, fill_price=100.0, quantity=100)
        assert fee == pytest.approx(10.0)

    def test_single_tier(self):
        """Single tier acts like a flat percentage fee."""
        tiers = [(0, 10)]
        model = TieredFee(tiers)
        order = make_order(Side.BUY, quantity=100)
        fee = model.compute(order, fill_price=100.0, quantity=100)
        assert fee == pytest.approx(10.0)

    def test_unsorted_tiers_are_sorted(self):
        """Tiers passed in wrong order are sorted internally."""
        tiers = [(100_000, 2), (0, 10), (10_000, 5)]
        model = TieredFee(tiers)
        order = make_order(Side.BUY, quantity=200)
        # Same as crossing_tier_boundary test
        fee = model.compute(order, fill_price=100.0, quantity=200)
        assert fee == pytest.approx(15.0)

    def test_empty_tiers_raises(self):
        """Empty tier list should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            TieredFee([])

    def test_zero_notional(self):
        """Zero notional produces zero fee."""
        tiers = [(0, 10), (10_000, 5)]
        model = TieredFee(tiers)
        order = make_order(Side.BUY, quantity=0)
        fee = model.compute(order, fill_price=100.0, quantity=0)
        assert fee == 0.0


# ---------- SECFee ----------


class TestSECFee:
    def test_sell_order_incurs_fee(self):
        """SEC fee applies to sell orders."""
        model = SECFee(rate_per_million=8.0)
        order = make_order(Side.SELL, quantity=1000)
        # notional = 1000 * 100 = 100,000
        # fee = 100,000 * 8 / 1,000,000 = 0.80
        fee = model.compute(order, fill_price=100.0, quantity=1000)
        assert fee == pytest.approx(0.80)

    def test_buy_order_zero_fee(self):
        """SEC fee is zero for buy orders."""
        model = SECFee(rate_per_million=8.0)
        order = make_order(Side.BUY, quantity=1000)
        fee = model.compute(order, fill_price=100.0, quantity=1000)
        assert fee == 0.0

    def test_large_sell(self):
        """SEC fee on $1M notional = $8.00."""
        model = SECFee(rate_per_million=8.0)
        order = make_order(Side.SELL, quantity=10_000)
        fee = model.compute(order, fill_price=100.0, quantity=10_000)
        # notional = 1,000,000; fee = 8.0
        assert fee == pytest.approx(8.0)

    def test_default_rate(self):
        """Default rate is $8 per million."""
        model = SECFee()
        order = make_order(Side.SELL, quantity=10_000)
        fee = model.compute(order, fill_price=100.0, quantity=10_000)
        assert fee == pytest.approx(8.0)

    def test_custom_rate(self):
        """Custom rate per million."""
        model = SECFee(rate_per_million=22.90)
        order = make_order(Side.SELL, quantity=10_000)
        fee = model.compute(order, fill_price=100.0, quantity=10_000)
        # notional = 1,000,000; fee = 22.90
        assert fee == pytest.approx(22.90)

    def test_zero_quantity_sell(self):
        """Zero quantity sell produces zero fee."""
        model = SECFee()
        order = make_order(Side.SELL, quantity=0)
        fee = model.compute(order, fill_price=100.0, quantity=0)
        assert fee == 0.0


# ---------- TAFFee ----------


class TestTAFFee:
    def test_per_share_calculation(self):
        """Basic per-share fee calculation."""
        model = TAFFee(per_share=0.000119, max_per_trade=5.95)
        order = make_order(Side.SELL, quantity=1000)
        fee = model.compute(order, fill_price=100.0, quantity=1000)
        # 1000 * 0.000119 = 0.119
        assert fee == pytest.approx(0.119)

    def test_max_cap_enforcement(self):
        """Fee is capped at max_per_trade."""
        model = TAFFee(per_share=0.000119, max_per_trade=5.95)
        order = make_order(Side.SELL, quantity=100_000)
        fee = model.compute(order, fill_price=100.0, quantity=100_000)
        # 100,000 * 0.000119 = 11.90, capped at 5.95
        assert fee == pytest.approx(5.95)

    def test_buy_order_zero_fee(self):
        """TAF fee is zero for buy orders."""
        model = TAFFee()
        order = make_order(Side.BUY, quantity=1000)
        fee = model.compute(order, fill_price=100.0, quantity=1000)
        assert fee == 0.0

    def test_exactly_at_cap(self):
        """Fee exactly equals the cap."""
        # 50,000 * 0.000119 = 5.95
        model = TAFFee(per_share=0.000119, max_per_trade=5.95)
        order = make_order(Side.SELL, quantity=50_000)
        fee = model.compute(order, fill_price=100.0, quantity=50_000)
        assert fee == pytest.approx(5.95)

    def test_just_under_cap(self):
        """Fee just under the cap is not capped."""
        model = TAFFee(per_share=0.000119, max_per_trade=5.95)
        order = make_order(Side.SELL, quantity=49_000)
        fee = model.compute(order, fill_price=100.0, quantity=49_000)
        expected = 49_000 * 0.000119
        assert fee == pytest.approx(expected)
        assert fee < 5.95

    def test_default_values(self):
        """Default per_share and max_per_trade."""
        model = TAFFee()
        order = make_order(Side.SELL, quantity=1000)
        fee = model.compute(order, fill_price=100.0, quantity=1000)
        assert fee == pytest.approx(1000 * 0.000119)

    def test_zero_quantity_sell(self):
        """Zero quantity produces zero fee."""
        model = TAFFee()
        order = make_order(Side.SELL, quantity=0)
        fee = model.compute(order, fill_price=100.0, quantity=0)
        assert fee == 0.0


# ---------- CompositeFee ----------


class TestCompositeFee:
    def test_combines_multiple_models(self):
        """Sum of a flat fee and a percentage fee."""
        flat = PerTradeFee(fee=1.00)
        pct = PercentageFee(bps=5)
        model = CompositeFee([flat, pct])
        order = make_order(Side.BUY, quantity=100)
        fee = model.compute(order, fill_price=100.0, quantity=100)
        # flat = 1.00, pct = 10,000 * 5 / 10,000 = 5.00, total = 6.00
        assert fee == pytest.approx(6.0)

    def test_all_components_summed(self):
        """Realistic composite: percentage + SEC + TAF."""
        pct = PercentageFee(bps=5)
        sec = SECFee(rate_per_million=8.0)
        taf = TAFFee(per_share=0.000119, max_per_trade=5.95)
        model = CompositeFee([pct, sec, taf])

        order = make_order(Side.SELL, quantity=1000)
        fee = model.compute(order, fill_price=100.0, quantity=1000)

        # pct: 100,000 * 5 / 10,000 = 50.0
        # sec: 100,000 * 8 / 1,000,000 = 0.80
        # taf: 1000 * 0.000119 = 0.119
        expected = 50.0 + 0.80 + 0.119
        assert fee == pytest.approx(expected)

    def test_buy_order_skips_sell_only_fees(self):
        """On a BUY, SEC and TAF contribute zero; only percentage applies."""
        pct = PercentageFee(bps=10)
        sec = SECFee()
        taf = TAFFee()
        model = CompositeFee([pct, sec, taf])

        order = make_order(Side.BUY, quantity=100)
        fee = model.compute(order, fill_price=100.0, quantity=100)

        # pct: 10,000 * 10 / 10,000 = 10.0; sec/taf = 0 for BUY
        assert fee == pytest.approx(10.0)

    def test_empty_composite(self):
        """Composite with no models returns zero fee."""
        model = CompositeFee([])
        order = make_order(Side.BUY, quantity=100)
        fee = model.compute(order, fill_price=100.0, quantity=100)
        assert fee == 0.0

    def test_single_model_composite(self):
        """Composite with one model acts like that model alone."""
        inner = PercentageFee(bps=10)
        model = CompositeFee([inner])
        order = make_order(Side.BUY, quantity=100)
        fee = model.compute(order, fill_price=100.0, quantity=100)
        assert fee == pytest.approx(10.0)

    def test_composite_with_tiered(self):
        """Composite including a TieredFee model."""
        tiered = TieredFee([(0, 10), (10_000, 5)])
        sec = SECFee()
        model = CompositeFee([tiered, sec])

        order = make_order(Side.SELL, quantity=200)
        fee = model.compute(order, fill_price=100.0, quantity=200)

        # tiered: 10,000 at 10bps = 10.0; 10,000 at 5bps = 5.0; total = 15.0
        # sec: 20,000 * 8 / 1,000,000 = 0.16
        assert fee == pytest.approx(15.16)
