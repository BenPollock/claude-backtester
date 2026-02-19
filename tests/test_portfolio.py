"""Tests for portfolio, position, and order modules."""

from datetime import date

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
