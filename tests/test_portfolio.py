"""Tests for portfolio, position, and order modules."""

import math
from datetime import date

import pytest

from backtester.portfolio.position import Position, Lot, StopState
from backtester.portfolio.portfolio import Portfolio, PortfolioState
from backtester.portfolio.order import Order, Fill, Trade, TradeLogEntry
from backtester.types import Side, OrderType, OrderStatus


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


# ===========================================================================
# Bug fix regression tests
# ===========================================================================


class TestUnrealizedPnlZeroMarketPrice:
    """Regression tests for the bug where unrealized_pnl returned 0.0
    when _market_price was 0, masking real losses.
    """

    def test_long_unrealized_pnl_at_zero_market_price(self):
        """Long position with market_price=0 should show full loss."""
        pos = Position(symbol="GONE")
        pos.add_lot(100, 50.0, date(2020, 1, 2))
        pos.update_market_price(0.0)

        # Loss = (0 - 50) * 100 = -5000
        assert pos.unrealized_pnl == -5000.0

    def test_short_unrealized_pnl_at_zero_market_price(self):
        """Short position with market_price=0 should show full profit."""
        pos = Position(symbol="GONE")
        pos.add_lot(-100, 50.0, date(2020, 1, 2))
        pos.update_market_price(0.0)

        # Profit = (50 - 0) * 100 = 5000
        assert pos.unrealized_pnl == 5000.0

    def test_unrealized_pnl_uninitialized_market_price(self):
        """Newly created position (default _market_price=0) reports correct PnL."""
        pos = Position(symbol="NEW")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        # _market_price is 0.0 by default (uninitialized)
        # PnL = (0 - 100) * 100 = -10000
        assert pos.unrealized_pnl == -10000.0


class TestCloseLotsFifoPnlPctZeroExitPrice:
    """Regression test for close_lots_fifo returning pnl_pct=0.0
    when exit_price is 0. Should return NaN for consistency.
    """

    def test_cover_short_at_zero_price_returns_nan_pnl_pct(self):
        """Covering a short at $0 should produce pnl_pct=NaN (undefined)."""
        pos = Position(symbol="GONE")
        pos.add_lot(-100, 50.0, date(2020, 1, 2))

        trades = pos.close_lots_fifo(100, 0.0, date(2020, 6, 1))

        assert len(trades) == 1
        assert math.isnan(trades[0].pnl_pct)
        # Dollar PnL is correct: (50 - 0) * 100 = 5000
        assert trades[0].pnl == 5000.0


# ===========================================================================
# sell_lots_lifo tests (previously untested)
# ===========================================================================


class TestSellLotsLifo:
    """Unit tests for sell_lots_lifo (LIFO sell ordering)."""

    def test_lifo_sells_from_last_lot_first(self):
        """LIFO should sell from the most recently added lot first."""
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))  # lot 0 (oldest)
        pos.add_lot(50, 120.0, date(2020, 2, 3))  # lot 1 (newest)

        trades = pos.sell_lots_lifo(30, 130.0, date(2020, 3, 4))

        assert len(trades) == 1
        assert trades[0].entry_price == 120.0  # sold from newest lot
        assert trades[0].quantity == 30
        assert pos.total_quantity == 70
        assert pos.lots[1].quantity == 20  # 50 - 30 = 20 remaining

    def test_lifo_spans_multiple_lots(self):
        """Selling more than the last lot should continue to next-to-last."""
        pos = Position(symbol="AAPL")
        pos.add_lot(30, 100.0, date(2020, 1, 2))  # lot 0
        pos.add_lot(20, 120.0, date(2020, 2, 3))  # lot 1 (newest)

        trades = pos.sell_lots_lifo(40, 130.0, date(2020, 3, 4))

        # Should exhaust lot 1 (20), then take 20 from lot 0
        assert len(trades) == 2
        assert trades[0].entry_price == 120.0
        assert trades[0].quantity == 20
        assert trades[1].entry_price == 100.0
        assert trades[1].quantity == 20
        assert pos.total_quantity == 10

    def test_lifo_sell_all(self):
        """Sell all shares via LIFO."""
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))
        pos.add_lot(50, 120.0, date(2020, 2, 3))

        trades = pos.sell_lots_lifo(100, 130.0, date(2020, 3, 4))

        assert len(trades) == 2
        assert pos.total_quantity == 0
        assert len(pos.lots) == 0

    def test_lifo_too_many_raises(self):
        """Selling more than held should raise ValueError."""
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))
        with pytest.raises(ValueError):
            pos.sell_lots_lifo(100, 110.0, date(2020, 2, 3))

    def test_lifo_pnl_calculation(self):
        """Verify PnL is correct for LIFO sells."""
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))

        trades = pos.sell_lots_lifo(100, 150.0, date(2020, 6, 1))
        assert trades[0].pnl == (150.0 - 100.0) * 100  # $5000
        assert trades[0].pnl_pct == pytest.approx(0.5)

    def test_lifo_with_exit_commission(self):
        """LIFO sell with exit commission deducted from PnL."""
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2), commission=5.0)

        trades = pos.sell_lots_lifo(100, 110.0, date(2020, 2, 3), exit_commission=5.0)
        # pnl = (110 - 100) * 100 - (5 entry comm + 5 exit comm) = 990
        assert trades[0].pnl == 990.0
        assert trades[0].fees_total == 10.0


# ===========================================================================
# sell_lots_by_cost tests (lowest_cost_first untested)
# ===========================================================================


class TestSellLotsByCostLowestFirst:
    """Tests for sell_lots_by_cost with highest_first=False."""

    def test_lowest_cost_first(self):
        """Sell cheapest lots first when highest_first=False."""
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 200.0, date(2020, 1, 2))  # expensive
        pos.add_lot(50, 100.0, date(2020, 2, 3))  # cheap
        pos.add_lot(50, 150.0, date(2020, 3, 4))  # mid

        trades = pos.sell_lots_by_cost(50, 180.0, date(2020, 4, 5), highest_first=False)

        assert len(trades) == 1
        assert trades[0].entry_price == 100.0  # cheapest sold first
        assert pos.total_quantity == 100

    def test_lowest_cost_preserves_lot_order(self):
        """After selling lowest cost, remaining lots keep FIFO insertion order."""
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 200.0, date(2020, 1, 2))  # lot A
        pos.add_lot(50, 100.0, date(2020, 2, 3))  # lot B (cheapest)
        pos.add_lot(50, 150.0, date(2020, 3, 4))  # lot C

        pos.sell_lots_by_cost(50, 180.0, date(2020, 4, 5), highest_first=False)

        # Lot B removed; A and C remain in original FIFO order
        assert len(pos.lots) == 2
        assert pos.lots[0].entry_price == 200.0  # lot A
        assert pos.lots[1].entry_price == 150.0  # lot C

    def test_by_cost_too_many_raises(self):
        """Selling more than held raises ValueError."""
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))
        with pytest.raises(ValueError):
            pos.sell_lots_by_cost(100, 110.0, date(2020, 2, 3))


# ===========================================================================
# close_lots_fifo — multi-lot and commission tests
# ===========================================================================


class TestCloseLotsMultiLot:
    """Tests for close_lots_fifo spanning multiple short lots."""

    def test_close_fifo_across_two_short_lots(self):
        """Covering more shares than the first lot should FIFO into the second."""
        pos = Position(symbol="AAPL")
        pos.add_lot(-60, 100.0, date(2020, 1, 2))  # lot 0
        pos.add_lot(-40, 120.0, date(2020, 2, 3))  # lot 1

        trades = pos.close_lots_fifo(80, 110.0, date(2020, 3, 4))

        # Should cover all 60 from lot 0, then 20 from lot 1
        assert len(trades) == 2
        assert trades[0].quantity == 60
        assert trades[0].entry_price == 100.0
        assert trades[0].pnl == (100.0 - 110.0) * 60  # -$600 loss
        assert trades[1].quantity == 20
        assert trades[1].entry_price == 120.0
        assert trades[1].pnl == (120.0 - 110.0) * 20  # $200 profit
        assert pos.total_quantity == -20  # 20 shares remain short

    def test_close_fifo_with_commission_across_lots(self):
        """Commission should be allocated proportionally across covered lots."""
        pos = Position(symbol="AAPL")
        pos.add_lot(-50, 100.0, date(2020, 1, 2), commission=10.0)
        pos.add_lot(-50, 120.0, date(2020, 2, 3), commission=10.0)

        trades = pos.close_lots_fifo(100, 110.0, date(2020, 3, 4), exit_commission=20.0)

        assert len(trades) == 2
        # Exit commission = 20/100 = 0.20 per share
        # Trade 0: entry_comm=10, exit_comm=50*0.20=10, fees=20
        assert trades[0].fees_total == pytest.approx(20.0)
        # Trade 1: entry_comm=10, exit_comm=50*0.20=10, fees=20
        assert trades[1].fees_total == pytest.approx(20.0)
        assert pos.total_quantity == 0

    def test_close_too_many_raises(self):
        """Covering more than short position raises ValueError."""
        pos = Position(symbol="AAPL")
        pos.add_lot(-50, 100.0, date(2020, 1, 2))
        with pytest.raises(ValueError, match="Cannot cover"):
            pos.close_lots_fifo(100, 110.0, date(2020, 2, 3))


# ===========================================================================
# Short position avg_entry_price with multiple lots
# ===========================================================================


class TestShortAvgEntryPrice:
    """Verify avg_entry_price for short positions with multiple lots."""

    def test_short_two_lots_avg(self):
        """avg_entry_price for shorts uses abs(qty) weighted average."""
        pos = Position(symbol="AAPL")
        pos.add_lot(-60, 100.0, date(2020, 1, 2))
        pos.add_lot(-40, 120.0, date(2020, 2, 3))

        # Weighted avg = (100*60 + 120*40) / 100 = 10800/100 = 108
        assert pos.avg_entry_price == pytest.approx(108.0)
        assert pos.total_quantity == -100

    def test_short_avg_after_partial_cover(self):
        """After partially covering, avg_entry_price reflects remaining lots."""
        pos = Position(symbol="AAPL")
        pos.add_lot(-60, 100.0, date(2020, 1, 2))
        pos.add_lot(-40, 120.0, date(2020, 2, 3))

        # Cover first 60 shares (all of lot 0)
        pos.close_lots_fifo(60, 110.0, date(2020, 3, 4))

        # Only lot 1 remains: -40 @ $120
        assert pos.avg_entry_price == pytest.approx(120.0)
        assert pos.total_quantity == -40


# ===========================================================================
# Position.direction property tests
# ===========================================================================


class TestPositionDirection:
    """Tests for the direction property."""

    def test_long_direction(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        assert pos.direction == "long"
        assert not pos.is_short

    def test_short_direction(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 100.0, date(2020, 1, 2))
        assert pos.direction == "short"
        assert pos.is_short

    def test_flat_direction(self):
        pos = Position(symbol="AAPL")
        assert pos.direction == "flat"
        assert not pos.is_short


# ===========================================================================
# StopState tests
# ===========================================================================


class TestStopState:
    """Unit tests for StopState trailing_stop_price computation."""

    def test_trailing_stop_price_with_valid_state(self):
        ss = StopState(trailing_stop_pct=0.05, trailing_high=200.0)
        # trailing_stop_price = 200 * (1 - 0.05) = 190.0
        assert ss.trailing_stop_price == pytest.approx(190.0)

    def test_trailing_stop_price_no_pct(self):
        ss = StopState(trailing_high=200.0)
        assert ss.trailing_stop_price is None

    def test_trailing_stop_price_zero_high(self):
        ss = StopState(trailing_stop_pct=0.05, trailing_high=0.0)
        assert ss.trailing_stop_price is None

    def test_trailing_stop_price_negative_high(self):
        ss = StopState(trailing_stop_pct=0.05, trailing_high=-10.0)
        assert ss.trailing_stop_price is None


# ===========================================================================
# Order / Fill / Trade / TradeLogEntry model tests
# ===========================================================================


class TestOrderModel:
    """Tests for Order dataclass defaults and fields."""

    def test_order_defaults(self):
        o = Order(
            symbol="AAPL", side=Side.BUY, quantity=100,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
        )
        assert o.status == OrderStatus.PENDING
        assert o.time_in_force == "DAY"
        assert o.limit_price is None
        assert o.stop_price is None
        assert o.expiry_date is None
        assert o.days_pending == 0
        assert o.reason == ""
        assert o.parent_id is None
        assert len(o.id) == 12  # uuid hex[:12]

    def test_order_limit_fields(self):
        o = Order(
            symbol="AAPL", side=Side.BUY, quantity=50,
            order_type=OrderType.LIMIT, signal_date=date(2020, 1, 2),
            limit_price=150.0, time_in_force="GTC",
            expiry_date=date(2020, 2, 1),
        )
        assert o.limit_price == 150.0
        assert o.time_in_force == "GTC"
        assert o.expiry_date == date(2020, 2, 1)


class TestFillModel:
    """Tests for Fill frozen dataclass."""

    def test_fill_is_frozen(self):
        f = Fill(
            order_id="abc123", symbol="AAPL", side=Side.BUY,
            quantity=100, price=150.0, commission=5.0,
            fill_date=date(2020, 1, 3), slippage=0.15,
        )
        assert f.symbol == "AAPL"
        assert f.quantity == 100
        with pytest.raises(AttributeError):
            f.price = 200.0  # frozen


class TestTradeLogEntryModel:
    """Tests for TradeLogEntry frozen dataclass."""

    def test_trade_log_entry_fields(self):
        entry = TradeLogEntry(
            date=date(2020, 1, 3), symbol="AAPL", action=Side.SELL,
            quantity=100, price=150.0, value=15_000.0,
            avg_cost_basis=140.0, fees=5.0, slippage=0.15,
        )
        assert entry.action == Side.SELL
        assert entry.avg_cost_basis == 140.0
        assert entry.value == 15_000.0
        with pytest.raises(AttributeError):
            entry.price = 200.0  # frozen


# ===========================================================================
# Portfolio margin / available_capital for long-only
# ===========================================================================


class TestPortfolioMarginLongOnly:
    """Verify margin_used and available_capital for long-only portfolios."""

    def test_margin_used_zero_for_long_only(self):
        """Long positions should contribute 0 to margin_used."""
        p = Portfolio(cash=100_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)

        assert p.margin_used == 0.0

    def test_available_capital_equals_cash_for_long_only(self):
        """For long-only, available_capital should equal cash."""
        p = Portfolio(cash=85_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)

        assert p.available_capital() == 85_000.0


# ===========================================================================
# PortfolioState immutability test
# ===========================================================================


class TestPortfolioStateImmutability:
    """Verify PortfolioState snapshots are frozen."""

    def test_snapshot_is_frozen(self):
        p = Portfolio(cash=100_000.0)
        snap = p.snapshot()
        with pytest.raises(AttributeError):
            snap.cash = 50_000.0

    def test_snapshot_decoupled_from_portfolio(self):
        """Changes to portfolio after snapshot should not affect the snapshot."""
        p = Portfolio(cash=100_000.0)
        snap = p.snapshot()

        p.cash = 50_000.0
        assert snap.cash == 100_000.0
        assert snap.total_equity == 100_000.0


# ===========================================================================
# Holding days calculation
# ===========================================================================


class TestHoldingDays:
    """Verify holding_days is calculated correctly in Trade records."""

    def test_holding_days_long(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))

        trades = pos.sell_lots_fifo(100, 110.0, date(2020, 1, 12))
        assert trades[0].holding_days == 10

    def test_holding_days_short(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 100.0, date(2020, 3, 1))

        trades = pos.close_lots_fifo(100, 90.0, date(2020, 3, 15))
        assert trades[0].holding_days == 14


# ===========================================================================
# Coverage-expanding tests
# ===========================================================================


class TestGetPositionReturnsNone:
    """Verify get_position returns None for unknown symbols."""

    def test_get_position_unknown_symbol(self):
        p = Portfolio(cash=100_000.0)
        assert p.get_position("UNKNOWN") is None

    def test_has_position_unknown_symbol(self):
        p = Portfolio(cash=100_000.0)
        assert not p.has_position("UNKNOWN")


class TestRebalanceMissingPrice:
    """Rebalance when a held position's symbol is missing from prices dict."""

    def test_missing_price_excludes_symbol(self):
        """Position with no price in prices dict should be excluded from current weights."""
        from backtester.types import Side

        p = Portfolio(cash=50_000.0)
        pos_a = p.open_position("AAPL")
        pos_a.add_lot(100, 100.0, date(2020, 1, 2))
        pos_a.update_market_price(100.0)

        pos_b = p.open_position("MSFT")
        pos_b.add_lot(50, 200.0, date(2020, 1, 2))
        pos_b.update_market_price(200.0)

        # Provide price for AAPL only, not MSFT
        orders = p.compute_rebalance_orders(
            target_weights={"AAPL": 0.20},
            prices={"AAPL": 100.0},  # MSFT missing
        )
        # MSFT should not generate any order since it's not in prices
        syms = [s for s, _, _ in orders]
        assert "MSFT" not in syms

    def test_zero_price_excludes_symbol(self):
        """Symbol with price=0 should be excluded from rebalance."""
        from backtester.types import Side

        p = Portfolio(cash=50_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)

        orders = p.compute_rebalance_orders(
            target_weights={"AAPL": 0.50},
            prices={"AAPL": 0.0},  # zero price
        )
        assert orders == []

    def test_negative_target_weight_skipped(self):
        """Negative target weight should be treated like zero (skipped)."""
        from backtester.types import Side

        p = Portfolio(cash=100_000.0)
        orders = p.compute_rebalance_orders(
            target_weights={"AAPL": -0.10, "MSFT": 0.30},
            prices={"AAPL": 100.0, "MSFT": 200.0},
        )
        # Only MSFT should get a BUY; AAPL negative weight skipped
        syms = [s for s, _, _ in orders]
        assert "AAPL" not in syms
        assert "MSFT" in syms


class TestSnapshotMarginUsed:
    """Verify snapshot includes margin_used from short positions."""

    def test_snapshot_includes_margin_used(self):
        p = Portfolio(cash=100_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(-100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)

        snap = p.snapshot()
        assert snap.margin_used == 10_000.0

    def test_snapshot_zero_margin_for_long(self):
        p = Portfolio(cash=100_000.0)
        pos = p.open_position("AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)

        snap = p.snapshot()
        assert snap.margin_used == 0.0


class TestMaxPositionValueMixed:
    """max_position_value with mixed long and short positions."""

    def test_mixed_long_short(self):
        """max_position_value should return the largest market_value (long wins)."""
        p = Portfolio(cash=100_000.0)

        pos_long = p.open_position("AAPL")
        pos_long.add_lot(100, 100.0, date(2020, 1, 2))
        pos_long.update_market_price(100.0)  # market_value = 10000

        pos_short = p.open_position("MSFT")
        pos_short.add_lot(-200, 30.0, date(2020, 1, 2))
        pos_short.update_market_price(30.0)  # market_value = -6000

        assert p.max_position_value() == 10_000.0  # long position is max


class TestOpenPositionIdempotent:
    """open_position should return existing position if already exists."""

    def test_open_existing_returns_same(self):
        p = Portfolio(cash=100_000.0)
        pos1 = p.open_position("AAPL")
        pos1.add_lot(100, 100.0, date(2020, 1, 2))

        pos2 = p.open_position("AAPL")
        assert pos2 is pos1
        assert pos2.total_quantity == 100
