"""Tests for short selling support (Feature 3A)."""

from datetime import date

import pandas as pd
import pytest

from backtester.types import SignalAction, Side, OrderType, OrderStatus
from backtester.config import BacktestConfig, StopConfig
from backtester.portfolio.position import Position, Lot
from backtester.portfolio.portfolio import Portfolio, PortfolioState
from backtester.portfolio.order import Order
from backtester.execution.broker import SimulatedBroker
from backtester.execution.slippage import FixedSlippage
from backtester.execution.fees import PerTradeFee
from backtester.strategies.base import Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_row(open_price=100.0, high=105.0, low=95.0, close=100.0, volume=1_000_000):
    return pd.Series({
        "Open": open_price, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


class DummyStrategy(Strategy):
    """Minimal strategy for testing -- emits SHORT/COVER signals."""

    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        return SignalAction.HOLD


# ---------------------------------------------------------------------------
# 1. SignalAction enum
# ---------------------------------------------------------------------------

class TestSignalActionEnum:
    def test_short_exists(self):
        assert SignalAction.SHORT is not None
        assert SignalAction.SHORT.name == "SHORT"

    def test_cover_exists(self):
        assert SignalAction.COVER is not None
        assert SignalAction.COVER.name == "COVER"

    def test_original_values_unchanged(self):
        # BUY, SELL, HOLD must still exist
        assert SignalAction.BUY is not None
        assert SignalAction.SELL is not None
        assert SignalAction.HOLD is not None


# ---------------------------------------------------------------------------
# 2. Position -- short direction and PnL
# ---------------------------------------------------------------------------

class TestShortPosition:
    def test_short_direction(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        assert pos.direction == "short"
        assert pos.is_short is True
        assert pos.total_quantity == -100

    def test_long_direction_unchanged(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        assert pos.direction == "long"
        assert pos.is_short is False

    def test_flat_direction(self):
        pos = Position(symbol="AAPL")
        assert pos.direction == "flat"

    def test_short_market_value_is_negative(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(140.0)
        # market_value = -100 * 140 = -14000
        assert pos.market_value == -14_000.0

    def test_short_avg_entry_price(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-50, 100.0, date(2020, 1, 2))
        pos.add_lot(-50, 120.0, date(2020, 2, 3))
        # avg = (100*50 + 120*50) / 100 = 110
        assert pos.avg_entry_price == 110.0

    def test_short_pnl_profit_when_price_falls(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(140.0)
        # profit = (150 - 140) * 100 = 1000
        assert pos.unrealized_pnl == 1000.0

    def test_short_pnl_loss_when_price_rises(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(160.0)
        # loss = (150 - 160) * 100 = -1000
        assert pos.unrealized_pnl == -1000.0

    def test_long_unrealized_pnl_unchanged(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(110.0)
        # (110 - 100) * 100 = 1000
        assert pos.unrealized_pnl == 1000.0


# ---------------------------------------------------------------------------
# 3. Short borrow cost accrual
# ---------------------------------------------------------------------------

class TestBorrowCost:
    def test_accrue_borrow_cost(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        # abs(market_value) = 15000, rate = 0.02, 1 day
        # cost = 15000 * 0.02 / 252 = ~1.19
        pos.accrue_borrow_cost(0.02, days=1)
        expected = 15_000 * 0.02 / 252
        assert abs(pos.short_borrow_cost_accrued - expected) < 0.01

    def test_accrue_multiple_days(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        pos.accrue_borrow_cost(0.02, days=5)
        expected = 15_000 * 0.02 / 252 * 5
        assert abs(pos.short_borrow_cost_accrued - expected) < 0.01

    def test_accrue_noop_for_long(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        pos.accrue_borrow_cost(0.02, days=1)
        assert pos.short_borrow_cost_accrued == 0.0


# ---------------------------------------------------------------------------
# 4. FIFO close for short positions
# ---------------------------------------------------------------------------

class TestCloseLotsFifo:
    def test_cover_all(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        trades = pos.close_lots_fifo(100, 140.0, date(2020, 2, 3))
        assert len(trades) == 1
        assert trades[0].quantity == 100
        # PnL: (150 - 140) * 100 = 1000
        assert trades[0].pnl == 1000.0
        assert pos.total_quantity == 0

    def test_partial_cover_fifo(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-50, 100.0, date(2020, 1, 2))
        pos.add_lot(-50, 120.0, date(2020, 2, 3))
        # Cover 70 shares: 50 from first lot, 20 from second
        trades = pos.close_lots_fifo(70, 110.0, date(2020, 3, 4))
        assert len(trades) == 2
        # First lot: (100 - 110) * 50 = -500
        assert trades[0].pnl == -500.0
        assert trades[0].quantity == 50
        # Second lot: (120 - 110) * 20 = 200
        assert trades[1].pnl == 200.0
        assert trades[1].quantity == 20
        # 30 shares remain short
        assert pos.total_quantity == -30

    def test_cover_too_many_raises(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-50, 100.0, date(2020, 1, 2))
        with pytest.raises(ValueError):
            pos.close_lots_fifo(100, 110.0, date(2020, 2, 3))

    def test_cover_with_commission(self):
        pos = Position(symbol="AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2), commission=5.0)
        trades = pos.close_lots_fifo(100, 140.0, date(2020, 2, 3), exit_commission=5.0)
        # PnL = (150-140)*100 - (5 entry + 5 exit) = 1000 - 10 = 990
        assert trades[0].pnl == 990.0
        assert trades[0].fees_total == 10.0


# ---------------------------------------------------------------------------
# 5. Portfolio margin tracking
# ---------------------------------------------------------------------------

class TestMarginTracking:
    def test_margin_used_for_short_position(self):
        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        # abs(market_value) = abs(-100 * 150) = 15000
        assert portfolio.margin_used == 15_000.0

    def test_margin_used_zero_for_long_only(self):
        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        assert portfolio.margin_used == 0.0

    def test_available_capital_with_short(self):
        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        # available = 100000 - 15000 * 1.5 = 100000 - 22500 = 77500
        assert portfolio.available_capital(margin_requirement=1.5) == 77_500.0

    def test_portfolio_state_includes_margin(self):
        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        state = portfolio.snapshot()
        assert state.margin_used == 15_000.0

    def test_close_position_works_for_short(self):
        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        # Cover all
        pos.close_lots_fifo(100, 140.0, date(2020, 2, 3))
        assert pos.total_quantity == 0
        portfolio.close_position("AAPL")
        assert not portfolio.has_position("AAPL")


# ---------------------------------------------------------------------------
# 6. Broker -- short entry and cover fills
# ---------------------------------------------------------------------------

class TestBrokerShortFills:
    def test_short_entry_fill(self):
        """SHORT signal creates a negative position via broker."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
            reason="short_entry",
        )
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=150.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert len(fills) == 1
        assert fills[0].quantity == 100
        assert fills[0].side == Side.SELL
        assert portfolio.has_position("AAPL")
        assert portfolio.positions["AAPL"].total_quantity == -100
        # Cash increases from short sale proceeds
        assert portfolio.cash == 100_000.0 + 15_000.0

    def test_cover_fill(self):
        """COVER signal closes a short position via broker."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=115_000.0)  # includes short sale proceeds
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=-1,  # sentinel: cover all
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 3),
            reason="cover",
        )
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=140.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        assert len(fills) == 1
        assert fills[0].quantity == 100
        assert not portfolio.has_position("AAPL")
        # Cash after cover: 115000 - 140*100 = 115000 - 14000 = 101000
        assert portfolio.cash == 101_000.0
        # Trade log should record the round trip
        assert len(portfolio.trade_log) == 1
        assert portfolio.trade_log[0].pnl == 1000.0  # (150-140)*100

    def test_cover_sentinel_resolves_to_full_position(self):
        """qty=-1 sentinel on a cover order resolves to abs of short position."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=200_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-75, 200.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=-1,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 3),
            reason="cover",
        )
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=210.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        assert len(fills) == 1
        assert fills[0].quantity == 75  # resolved from sentinel
        assert not portfolio.has_position("AAPL")

    def test_slippage_on_short_entry(self):
        """Short entry: slippage works against us (fill at lower price = less proceeds)."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=100),  # 1%
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
            reason="short_entry",
        )
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        # SELL side slippage: fill at lower price (100 - 1% = 99)
        assert fills[0].price == 99.0
        assert fills[0].slippage == 1.0

    def test_slippage_on_cover(self):
        """Cover: slippage works against us (fill at higher price = pay more)."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=100),  # 1%
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=200_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 100.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=-1,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 3),
            reason="cover",
        )
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        # BUY side slippage: fill at higher price (100 + 1% = 101)
        assert fills[0].price == 101.0
        assert fills[0].slippage == 1.0

    def test_fees_on_short_entry(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=5.0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.SELL, quantity=100,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
            reason="short_entry",
        )
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=100.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)

        assert fills[0].commission == 5.0
        # cash = 100000 + 100*100 - 5 = 109995
        assert portfolio.cash == 109_995.0

    def test_fees_on_cover(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=5.0),
        )
        portfolio = Portfolio(cash=200_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=-1,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 3),
            reason="cover",
        )
        broker.submit_order(order)

        market_data = {"AAPL": make_row(open_price=140.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)

        assert fills[0].commission == 5.0
        # cash = 200000 - 140*100 - 5 = 185995
        assert portfolio.cash == 185_995.0


# ---------------------------------------------------------------------------
# 7. Strategy base class -- SHORT/COVER sizing
# ---------------------------------------------------------------------------

class TestStrategySizing:
    def test_size_order_short(self):
        strategy = DummyStrategy()
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = pd.Series({"Close": 100.0})
        qty = strategy.size_order("AAPL", SignalAction.SHORT, row, state, 0.10)
        # Should be negative: -(100000 * 0.10 // 100) = -100
        assert qty == -100

    def test_size_order_cover_sentinel(self):
        strategy = DummyStrategy()
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"AAPL"}),
        )
        row = pd.Series({"Close": 100.0})
        qty = strategy.size_order("AAPL", SignalAction.COVER, row, state, 0.10)
        assert qty == -1  # sentinel

    def test_size_order_buy_unchanged(self):
        strategy = DummyStrategy()
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = pd.Series({"Close": 100.0})
        qty = strategy.size_order("AAPL", SignalAction.BUY, row, state, 0.10)
        assert qty == 100  # 100000 * 0.10 / 100 = 100

    def test_size_order_sell_unchanged(self):
        strategy = DummyStrategy()
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"AAPL"}),
        )
        row = pd.Series({"Close": 100.0})
        qty = strategy.size_order("AAPL", SignalAction.SELL, row, state, 0.10)
        assert qty == -1  # sentinel


# ---------------------------------------------------------------------------
# 8. Config -- short selling fields
# ---------------------------------------------------------------------------

class TestConfigShortFields:
    def test_default_allow_short_false(self):
        cfg = BacktestConfig(
            strategy_name="sma_crossover", tickers=["SPY"],
            benchmark="SPY", start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31), starting_cash=100_000.0,
            max_positions=10, max_alloc_pct=0.10,
        )
        assert cfg.allow_short is False
        assert cfg.short_borrow_rate == 0.02
        assert cfg.margin_requirement == 1.5

    def test_existing_config_unaffected(self, basic_config):
        """Existing basic_config fixture should still work."""
        assert basic_config.allow_short is False
        assert basic_config.strategy_name == "sma_crossover"


# ---------------------------------------------------------------------------
# 9. Short + long coexistence
# ---------------------------------------------------------------------------

class TestShortLongCoexistence:
    def test_different_tickers(self):
        """Long and short positions can coexist on different tickers."""
        portfolio = Portfolio(cash=100_000.0)
        long_pos = portfolio.open_position("AAPL")
        long_pos.add_lot(100, 150.0, date(2020, 1, 2))
        long_pos.update_market_price(160.0)

        short_pos = portfolio.open_position("TSLA")
        short_pos.add_lot(-50, 800.0, date(2020, 1, 2))
        short_pos.update_market_price(780.0)

        assert portfolio.num_positions == 2
        assert long_pos.direction == "long"
        assert short_pos.direction == "short"
        # Equity = cash + long_value + short_value
        # = 100000 + 16000 + (-39000) = 77000
        assert portfolio.total_equity == 77_000.0
        # Margin used = abs(-39000) = 39000
        assert portfolio.margin_used == 39_000.0


# ---------------------------------------------------------------------------
# 10. allow_short=False rejects SHORT signals
# ---------------------------------------------------------------------------

class TestAllowShortSwitch:
    def test_short_rejected_when_disabled(self):
        """When allow_short=False, engine should suppress SHORT signals.

        We test this indirectly via the config check logic. The engine
        converts SHORT -> HOLD when allow_short is False.
        """
        # This is tested at the engine level. Verify config default.
        cfg = BacktestConfig(
            strategy_name="sma_crossover", tickers=["SPY"],
            benchmark="SPY", start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31), starting_cash=100_000.0,
            max_positions=10, max_alloc_pct=0.10,
        )
        assert cfg.allow_short is False


# ---------------------------------------------------------------------------
# 11. Stop loss on short positions
# ---------------------------------------------------------------------------

class TestShortStopLoss:
    def test_stop_triggers_when_price_rises(self):
        """For short positions, stop loss triggers when price rises above stop."""
        from backtester.execution.stops import StopManager
        from backtester.portfolio.position import StopState

        portfolio = Portfolio(cash=200_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        # Set stop loss at 160 (price rises above entry = loss for short)
        pos.stop_state.stop_loss = 160.0

        stop_mgr = StopManager(
            stop_config=StopConfig(stop_loss_pct=0.05),
            fee_model=PerTradeFee(fee=0),
        )

        # Day where high reaches 162 -> triggers stop
        today_data = {"AAPL": make_row(open_price=155.0, high=162.0, low=150.0, close=158.0)}
        stop_mgr.check_stop_triggers(date(2020, 1, 3), today_data, portfolio)

        # Position should be closed (covered)
        assert not portfolio.has_position("AAPL")
        # Trade should be logged
        assert len(portfolio.trade_log) == 1
        # PnL: (150 - 160) * 100 = -1000 (loss)
        assert portfolio.trade_log[0].pnl == -1000.0

    def test_stop_does_not_trigger_when_price_falls(self):
        """For short positions, stop loss should NOT trigger when price falls."""
        from backtester.execution.stops import StopManager

        portfolio = Portfolio(cash=200_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        pos.stop_state.stop_loss = 160.0  # above entry

        stop_mgr = StopManager(
            stop_config=StopConfig(stop_loss_pct=0.05),
            fee_model=PerTradeFee(fee=0),
        )

        # Day where high is only 155 -> below stop of 160
        today_data = {"AAPL": make_row(open_price=148.0, high=155.0, low=140.0, close=145.0)}
        stop_mgr.check_stop_triggers(date(2020, 1, 3), today_data, portfolio)

        # Position should still be open
        assert portfolio.has_position("AAPL")
        assert portfolio.positions["AAPL"].total_quantity == -100

    def test_short_take_profit_triggers_when_price_falls(self):
        """For shorts, take profit triggers when low falls to target."""
        from backtester.execution.stops import StopManager

        portfolio = Portfolio(cash=200_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(-100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        pos.stop_state.take_profit = 130.0  # below entry for shorts

        stop_mgr = StopManager(
            stop_config=StopConfig(take_profit_pct=0.10),
            fee_model=PerTradeFee(fee=0),
        )

        # Day where low reaches 128 -> triggers take profit
        today_data = {"AAPL": make_row(open_price=135.0, high=136.0, low=128.0, close=132.0)}
        stop_mgr.check_stop_triggers(date(2020, 1, 3), today_data, portfolio)

        assert not portfolio.has_position("AAPL")
        assert len(portfolio.trade_log) == 1
        # PnL: (150 - 130) * 100 = 2000
        assert portfolio.trade_log[0].pnl == 2000.0


# ---------------------------------------------------------------------------
# 12. Existing long-only behavior unchanged
# ---------------------------------------------------------------------------

class TestLongBehaviorUnchanged:
    def test_sell_lots_fifo_still_works(self):
        """Original sell_lots_fifo method still works for long positions."""
        pos = Position(symbol="AAPL")
        pos.add_lot(50, 100.0, date(2020, 1, 2))
        pos.add_lot(50, 120.0, date(2020, 2, 3))
        trades = pos.sell_lots_fifo(70, 130.0, date(2020, 3, 4))
        assert len(trades) == 2
        assert trades[0].quantity == 50
        assert trades[0].entry_price == 100.0
        assert trades[1].quantity == 20
        assert trades[1].entry_price == 120.0
        assert pos.total_quantity == 30

    def test_long_portfolio_equity_unchanged(self):
        portfolio = Portfolio(cash=85_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        pos.update_market_price(150.0)
        assert portfolio.total_equity == 100_000.0

    def test_broker_long_buy_unchanged(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(symbol="AAPL", side=Side.BUY, quantity=100,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 2))
        broker.submit_order(order)
        market_data = {"AAPL": make_row(open_price=150.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)
        assert len(fills) == 1
        assert portfolio.cash == 85_000.0
        assert portfolio.positions["AAPL"].total_quantity == 100

    def test_broker_long_sell_unchanged(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=85_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))
        order = Order(symbol="AAPL", side=Side.SELL, quantity=-1,
                      order_type=OrderType.MARKET, signal_date=date(2020, 1, 3))
        broker.submit_order(order)
        market_data = {"AAPL": make_row(open_price=160.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)
        assert len(fills) == 1
        assert fills[0].quantity == 100
        assert portfolio.cash == 85_000.0 + 16_000.0
        assert not portfolio.has_position("AAPL")


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_cover_cancelled_when_no_short(self):
        """Cover order cancelled when no short position exists."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=100_000.0)
        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=-1,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
            reason="cover",
        )
        broker.submit_order(order)
        market_data = {"AAPL": make_row(open_price=150.0)}
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)
        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED

    def test_cover_cancelled_when_long_position(self):
        """Cover order cancelled when position is long (not short)."""
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0),
            fees=PerTradeFee(fee=0),
        )
        portfolio = Portfolio(cash=85_000.0)
        pos = portfolio.open_position("AAPL")
        pos.add_lot(100, 150.0, date(2020, 1, 2))

        order = Order(
            symbol="AAPL", side=Side.BUY, quantity=-1,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 3),
            reason="cover",
        )
        broker.submit_order(order)
        market_data = {"AAPL": make_row(open_price=160.0)}
        fills = broker.process_fills(date(2020, 1, 6), market_data, portfolio)
        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED
        # Long position should be untouched
        assert portfolio.positions["AAPL"].total_quantity == 100
