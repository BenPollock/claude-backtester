"""Tests for portfolio, position, and order modules."""

from datetime import date

import pytest

from backtester.portfolio.position import Position, Lot
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.order import Order, Trade
from backtester.types import Side, OrderType


class TestPosition:
    def test_empty_position(self):
        pos = Position(symbol="AAPL")
        assert pos.total_quantity == 0
        assert pos.avg_entry_price == 0.0
        assert pos.market_value == 0.0

    def test_add_lot(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        assert pos.total_quantity == 100
        assert pos.avg_entry_price == 150.0

    def test_multiple_lots_avg_price(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.add_lot(100, 200.0, date(2020, 2, 3))
        assert pos.total_quantity == 200
        assert pos.avg_entry_price == 150.0

    def test_market_value(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(110.0)
        assert pos.market_value == 11_000.0

    def test_sell_fifo(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))
        pos.add_lot(50, 120.0, date(2020, 2, 3))

        trades = pos.sell_lots_fifo(70, 130.0, date(2020, 3, 4))

        # Should sell all 50 from first lot, then 20 from second
        assert len(trades) == 2
        assert trades[0].quantity == 50
        assert trades[0].entry_price == 100.0
        assert trades[1].quantity == 20
        assert trades[1].entry_price == 120.0

        # 30 shares remain from second lot
        assert pos.total_quantity == 30
        assert pos.lots[0].entry_price == 120.0

    def test_sell_all(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        trades = pos.sell_lots_fifo(100, 110.0, date(2020, 2, 3))
        assert len(trades) == 1
        assert trades[0].pnl == (110.0 - 100.0) * 100
        assert pos.total_quantity == 0

    def test_sell_by_cost_preserves_fifo_order(self):
        """After sell_lots_by_cost, remaining lots must retain FIFO insertion order."""
        pos = Position(symbol="AAPL")
        # Add 3 lots in FIFO order: cheap, expensive, mid-price
        pos.add_lot(50, 100.0, date(2020, 1, 2))   # lot 0 (cheapest)
        pos.add_lot(50, 200.0, date(2020, 2, 3))   # lot 1 (most expensive)
        pos.add_lot(50, 150.0, date(2020, 3, 4))   # lot 2 (mid)

        # Sell 50 shares, highest cost first — should sell lot 1 ($200)
        trades = pos.sell_lots_by_cost(50, 180.0, date(2020, 4, 5), highest_first=True)

        assert len(trades) == 1
        assert trades[0].entry_price == 200.0

        # Remaining lots must still be in original FIFO order: $100, $150
        assert len(pos.lots) == 2
        assert pos.lots[0].entry_price == 100.0
        assert pos.lots[1].entry_price == 150.0

        # Verify sell_lots_fifo still works correctly on the preserved order
        fifo_trades = pos.sell_lots_fifo(50, 180.0, date(2020, 5, 6))
        assert fifo_trades[0].entry_price == 100.0  # FIFO: cheapest lot was added first

    def test_sell_too_many_raises(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))
        try:
            pos.sell_lots_fifo(100, 110.0, date(2020, 2, 3))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestPortfolio:
    def test_initial_state(self, portfolio):
        assert portfolio.cash == 100_000.0
        assert portfolio.total_equity == 100_000.0
        assert portfolio.num_positions == 0

    def test_open_and_close_position(self, portfolio):
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        portfolio.cash -= 15_000.0

        assert portfolio.num_positions == 1
        assert portfolio.has_position("AAPL")

        pos.update_market_price(150.0)
        assert portfolio.total_equity == 100_000.0  # cash + position = total

    def test_snapshot(self, portfolio):
        pos = portfolio.open_position("AAPL")
        pos.add_lot(10, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)

        snap = portfolio.snapshot()
        assert snap.num_positions == 1
        assert "AAPL" in snap.position_symbols
        assert snap.cash == portfolio.cash

    def test_equity_recording(self, portfolio):
        portfolio.record_equity(date(2020, 1, 2))
        portfolio.record_equity(date(2020, 1, 3))
        assert len(portfolio.equity_history) == 2

    def test_position_weight(self, portfolio):
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)
        portfolio.cash -= 10_000.0

        weight = portfolio.position_weight("AAPL")
        # 10000 / 100000 = 0.10
        assert abs(weight - 0.10) < 0.001

    def test_close_position_with_remaining_qty_no_op(self, portfolio):
        """close_position() should not remove a position that still has shares."""
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))

        portfolio.close_position("AAPL")
        # Position still has qty > 0, so it should remain
        assert portfolio.has_position("AAPL")
        assert portfolio.positions["AAPL"].total_quantity == 100

    def test_position_weight_zero_equity(self):
        """Should return 0.0 when equity is zero or negative."""
        p = Portfolio(cash=0.0)
        pos = p.open_position("AAPL")
        pos.add_lot(10, 100.0, date(2020, 1, 2))
        pos.update_market_price(0.0)
        # cash=0, market_value=0 → equity=0
        assert p.position_weight("AAPL") == 0.0

    def test_max_position_value_empty(self):
        """Empty portfolio should return 0.0 for max_position_value."""
        p = Portfolio(cash=100_000.0)
        assert p.max_position_value() == 0.0

    def test_record_equity_twice_same_date(self):
        """record_equity called twice for same date appends both entries."""
        p = Portfolio(cash=100_000.0)
        p.record_equity(date(2020, 1, 2))
        p.record_equity(date(2020, 1, 2))
        # Both snapshots are stored (no dedup)
        assert len(p.equity_history) == 2
        assert p.equity_history[0][0] == p.equity_history[1][0]

    def test_compute_rebalance_orders_zero_equity(self):
        """Rebalance with zero total equity should return empty list."""
        p = Portfolio(cash=0.0)
        orders = p.compute_rebalance_orders(
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
            prices={"AAPL": 100.0, "MSFT": 200.0},
        )
        assert orders == []

    def test_compute_rebalance_orders_all_zero_weights(self):
        """Rebalance with all-zero target weights should generate sells for existing positions."""
        p = Portfolio(cash=50_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)

        orders = p.compute_rebalance_orders(
            target_weights={"AAPL": 0.0},
            prices={"AAPL": 150.0},
        )
        # Current weight of AAPL = 15000/65000 ~= 0.23, target is 0 => sell
        assert len(orders) == 1
        sym, side, qty = orders[0]
        assert sym == "AAPL"
        assert side == Side.SELL
        assert qty > 0

    def test_position_weight_negative_equity(self):
        """position_weight should return 0.0 when total equity is negative."""
        p = Portfolio(cash=-50_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)
        # equity = -50000 + 10000 = -40000
        assert p.total_equity < 0
        assert p.position_weight("AAPL") == 0.0


class TestPositionEdgeCases:
    def test_sell_lots_by_cost_preserves_fifo_for_subsequent_calls(self):
        """After sell_lots_by_cost, sell_lots_fifo should still use original FIFO order.

        sell_lots_by_cost sorts a copy internally and does not mutate lot order.
        """
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))  # lot A: cheap (first in FIFO)
        pos.add_lot(50, 200.0, date(2020, 2, 3))  # lot B: expensive
        pos.add_lot(50, 150.0, date(2020, 3, 4))  # lot C: mid

        # sell_lots_by_cost sells from most expensive first
        trades1 = pos.sell_lots_by_cost(30, 180.0, date(2020, 4, 1), highest_first=True)
        assert trades1[0].entry_price == 200.0  # sold from lot B

        # Remaining: lot A (50 @100), lot B (20 @200), lot C (50 @150)
        # sell_lots_fifo should still consume from lot A first (original FIFO order)
        trades2 = pos.sell_lots_fifo(30, 180.0, date(2020, 5, 1))
        assert trades2[0].entry_price == 100.0  # FIFO order preserved

    def test_negative_total_equity_short_position(self):
        """Short position with large unrealized loss can produce negative market_value."""
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 100.0, date(2020, 1, 2))  # short 100 at $100
        pos.update_market_price(200.0)  # price doubled = big loss

        assert pos.is_short
        assert pos.market_value == -100 * 200.0  # -20000
        assert pos.unrealized_pnl == (100.0 - 200.0) * 100  # -10000

    def test_accrue_borrow_cost_multiple_days(self):
        """accrue_borrow_cost with days > 1 should scale proportionally."""
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)

        # abs(market_value) = 10000, rate = 0.05 (5% annual)
        cost_1day = pos.accrue_borrow_cost(0.05, days=1)
        expected_1day = 10_000 * 0.05 / 252

        # Reset for multi-day test
        pos2 = Position(symbol="AAPL")
        pos2.add_lot(-100, 100.0, date(2020, 1, 2))
        pos2.update_market_price(100.0)
        cost_5day = pos2.accrue_borrow_cost(0.05, days=5)

        assert cost_1day == pytest.approx(expected_1day)
        assert cost_5day == pytest.approx(expected_1day * 5)
        assert pos2.short_borrow_cost_accrued == pytest.approx(cost_5day)

    def test_accrue_borrow_cost_long_position_zero(self):
        """accrue_borrow_cost on long position should return 0."""
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)
        cost = pos.accrue_borrow_cost(0.05, days=5)
        assert cost == 0.0


class TestPositionZeroEntryPrice:
    """Edge cases for lots with zero entry price."""

    def test_sell_lot_with_zero_entry_price_pnl_pct(self):
        """Selling a lot with entry_price=0 should produce pnl_pct=NaN (not ZeroDivisionError).

        The guard `if lot.entry_price > 0 else _NAN` in sell_lots_fifo
        prevents division by zero and returns NaN for pnl_pct because
        a percentage return from a zero cost basis is undefined.
        """
        import math
        pos = Position(symbol="FREE")
        pos.add_lot(100, 0.0, date(2020, 1, 2))

        trades = pos.sell_lots_fifo(100, 50.0, date(2020, 2, 3))

        assert len(trades) == 1
        # pnl_pct is NaN because percentage return from zero cost basis is undefined
        assert math.isnan(trades[0].pnl_pct)
        # But dollar PnL is still correct: (50 - 0) * 100 = 5000
        assert trades[0].pnl == 5000.0
        assert pos.total_quantity == 0

    def test_lot_commission_after_repeated_partial_sells(self):
        """Verify lot commission doesn't go negative after multiple partial sells.

        When sell_lots_fifo partially sells from a lot, the lot's commission is
        reduced proportionally. Repeated partial sells with rounding could
        theoretically cause the commission to become slightly negative.
        """
        pos = Position(symbol="AAPL")
        # $10 commission on a 3-share lot — tricky for integer division
        pos.add_lot(3, 100.0, date(2020, 1, 2), commission=10.0)

        # Sell 1 share at a time
        trades1 = pos.sell_lots_fifo(1, 110.0, date(2020, 2, 1))
        assert pos.lots[0].commission >= 0

        trades2 = pos.sell_lots_fifo(1, 110.0, date(2020, 3, 1))
        assert pos.lots[0].commission >= 0

        trades3 = pos.sell_lots_fifo(1, 110.0, date(2020, 4, 1))
        assert pos.total_quantity == 0

        # All trades should have non-negative fees
        for t in trades1 + trades2 + trades3:
            assert t.fees_total >= 0


class TestCloseLotsShortFIFO:
    """Edge cases for close_lots_fifo on short positions."""

    def test_short_partial_cover_then_cover_remaining(self):
        """Short 100 shares, cover 50, then cover remaining 50."""
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        assert pos.is_short
        assert pos.total_quantity == -100

        # Cover first 50 at $140 (profit: price dropped)
        trades1 = pos.close_lots_fifo(50, 140.0, date(2020, 2, 3))
        assert len(trades1) == 1
        assert trades1[0].quantity == 50
        assert trades1[0].pnl == (150.0 - 140.0) * 50  # $500 profit
        assert pos.total_quantity == -50

        # Cover remaining 50 at $160 (loss: price rose)
        trades2 = pos.close_lots_fifo(50, 160.0, date(2020, 3, 4))
        assert len(trades2) == 1
        assert trades2[0].quantity == 50
        assert trades2[0].pnl == (150.0 - 160.0) * 50  # -$500 loss
        assert pos.total_quantity == 0
        assert len(pos.lots) == 0


class TestComputeRebalanceEdgeCases:
    """Edge cases for Portfolio.compute_rebalance_orders."""

    def test_rebalance_adds_new_symbol_not_in_positions(self):
        """Rebalancing to a target that includes a symbol not currently held
        should generate a BUY order for it.
        """
        from backtester.types import Side

        p = Portfolio(cash=50_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)
        # equity = 50k cash + 10k position = 60k

        orders = p.compute_rebalance_orders(
            target_weights={"AAPL": 0.20, "MSFT": 0.30},
            prices={"AAPL": 100.0, "MSFT": 200.0},
        )

        # Should have a BUY for MSFT (new symbol, currently 0% -> 30%)
        msft_orders = [(s, side, q) for s, side, q in orders if s == "MSFT"]
        assert len(msft_orders) == 1
        sym, side, qty = msft_orders[0]
        assert side == Side.BUY
        assert qty > 0
        # target = 0.30 * 60k = 18k, at $200/share = 90 shares
        assert qty == 90

    def test_rebalance_with_negative_equity(self):
        """Rebalance with negative total equity should return empty list."""
        p = Portfolio(cash=-100_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)
        # equity = -100k + 10k = -90k

        assert p.total_equity < 0
        orders = p.compute_rebalance_orders(
            target_weights={"AAPL": 0.5},
            prices={"AAPL": 100.0},
        )
        assert orders == []

    def test_position_weight_with_negative_total_equity(self):
        """position_weight should return 0.0 when total equity is negative,
        even when position has positive market value.
        """
        p = Portfolio(cash=-200_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        # equity = -200k + 15k = -185k

        assert p.total_equity < 0
        weight = p.position_weight("AAPL")
        assert weight == 0.0


class TestCloseLotsFifoOnLongPosition:
    """Verify close_lots_fifo raises ValueError when called on a long position.

    close_lots_fifo is designed exclusively for short positions (negative lots).
    A guard at the top of the method rejects long positions with a clear error.
    """

    def test_close_lots_fifo_on_long_position_raises(self):
        """close_lots_fifo on a long position raises ValueError."""
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        assert pos.direction == "long"

        with pytest.raises(ValueError, match="can only be called on short positions"):
            pos.close_lots_fifo(50, 160.0, date(2020, 2, 3))

        # Position is unchanged after the rejected call
        assert pos.total_quantity == 100

    def test_close_lots_fifo_on_flat_position_raises(self):
        """close_lots_fifo on a flat (empty) position raises ValueError."""
        pos = Position(symbol="AAPL")
        assert pos.direction == "flat"

        with pytest.raises(ValueError, match="can only be called on short positions"):
            pos.close_lots_fifo(1, 100.0, date(2020, 1, 2))


class TestMaxPositionValueAllShort:
    """Verify max_position_value behavior with all-short positions."""

    def test_max_position_value_all_short_returns_least_negative(self):
        """With only short positions, max_position_value returns the least
        negative market value (closest to zero).
        """
        p = Portfolio(cash=100_000.0)

        # Short position A: -100 shares at $50 => market_value = -5000
        pos_a = p.open_position("AAPL")
        pos_a.add_lot(-100, 50.0, date(2020, 1, 2))
        pos_a.update_market_price(50.0)

        # Short position B: -200 shares at $30 => market_value = -6000
        pos_b = p.open_position("MSFT")
        pos_b.add_lot(-200, 30.0, date(2020, 1, 2))
        pos_b.update_market_price(30.0)

        # max() of negative values returns the least negative
        assert p.max_position_value() == -5000.0, (
            "max_position_value with all-short positions should return the "
            "least negative value (AAPL at -5000, not MSFT at -6000)"
        )


class TestCommissionRoundingAccumulation:
    """Verify commission allocation across multiple partial sells.

    When a lot has a commission that doesn't divide evenly by the number
    of shares, partial sells allocate commission proportionally. This test
    verifies the total commission across all trades is bounded and does
    not exceed the original entry commission.
    """

    def test_commission_split_across_three_lots_three_sells(self):
        """Create 3 lots each with $10.00 commission, sell 1 share per lot
        across 3 separate FIFO sells. Verify total fees are bounded.
        """
        pos = Position(symbol="AAPL")
        # 3 lots of 3 shares each, $10.00 commission per lot
        pos.add_lot(3, 100.0, date(2020, 1, 2), commission=10.0)
        pos.add_lot(3, 110.0, date(2020, 2, 3), commission=10.0)
        pos.add_lot(3, 120.0, date(2020, 3, 4), commission=10.0)

        total_entry_commission = 30.0  # 3 lots * $10
        all_trades = []

        # Sell 1 share at a time, 9 total sells
        for i in range(9):
            trades = pos.sell_lots_fifo(
                1, 130.0, date(2020, 4, i + 1), exit_commission=0.0
            )
            all_trades.extend(trades)

        assert pos.total_quantity == 0

        # Sum of entry commissions allocated across all trades
        total_fees = sum(t.fees_total for t in all_trades)

        # Fees should be close to the original total entry commission
        # Allow small floating-point tolerance but no more than $0.01
        assert abs(total_fees - total_entry_commission) < 0.01, (
            f"Total fees ({total_fees:.6f}) should be within $0.01 of "
            f"original entry commission ({total_entry_commission:.2f})"
        )

        # Every individual trade should have non-negative fees
        for t in all_trades:
            assert t.fees_total >= 0, f"Trade fees should never be negative: {t.fees_total}"


class TestAllShortPortfolioEquity:
    """Verify total_equity calculation with only short positions."""

    def test_total_equity_with_short_position(self):
        """Cash=$100k and short position worth -$50k => total_equity=$50k.

        Short positions have negative market_value (negative qty * price).
        Total equity = cash + sum(market_values), so short positions
        reduce equity from the cash level.
        """
        p = Portfolio(cash=100_000.0)
        pos = p.open_position("AAPL")
        # Short 500 shares at $100 => market_value = -500 * 100 = -50000
        pos.add_lot(-500, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)

        assert pos.market_value == -50_000.0
        assert p.total_equity == 50_000.0, (
            "total_equity should be cash ($100k) + market_value (-$50k) = $50k"
        )

    def test_total_equity_all_short_multiple_positions(self):
        """Multiple short positions all reduce equity from cash."""
        p = Portfolio(cash=200_000.0)

        pos_a = p.open_position("AAPL")
        pos_a.add_lot(-100, 100.0, date(2020, 1, 2))
        pos_a.update_market_price(100.0)  # market_value = -10000

        pos_b = p.open_position("MSFT")
        pos_b.add_lot(-200, 50.0, date(2020, 1, 2))
        pos_b.update_market_price(50.0)  # market_value = -10000

        assert p.total_equity == 200_000.0 - 10_000.0 - 10_000.0


class TestNegativeAvailableCapital:
    """Verify available_capital behavior when margin exceeds cash."""

    def test_available_capital_goes_negative(self):
        """When margin_used * margin_requirement > cash, available_capital is negative.

        available_capital = cash - margin_used * margin_requirement
        margin_used = sum(abs(market_value)) for short positions
        """
        p = Portfolio(cash=10_000.0)

        # Short 1000 shares at $100 => |market_value| = 100000
        pos = p.open_position("AAPL")
        pos.add_lot(-1000, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)

        assert p.margin_used == 100_000.0
        # available_capital = 10000 - 100000 * 1.5 = -140000
        assert p.available_capital(margin_requirement=1.5) == -140_000.0
        assert p.available_capital(margin_requirement=1.5) < 0, (
            "available_capital should be negative when margin exceeds cash"
        )

    def test_available_capital_with_default_margin(self):
        """Verify the default margin_requirement=1.5 is used."""
        p = Portfolio(cash=50_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(-100, 200.0, date(2020, 1, 2))
        pos.update_market_price(200.0)

        # margin_used = 20000, default requirement = 1.5
        # available = 50000 - 20000 * 1.5 = 20000
        assert p.available_capital() == 20_000.0
