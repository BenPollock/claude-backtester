"""Tests for stop-loss, take-profit, and trailing stop logic."""

from datetime import date

import pandas as pd
import pytest

from backtester.config import BacktestConfig, StopConfig
from backtester.engine import BacktestEngine
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.position import Position, StopState
from backtester.portfolio.order import Fill
from backtester.types import Side


def _make_engine(stop_config):
    """Create a minimal BacktestEngine with the given StopConfig."""
    config = BacktestConfig(
        strategy_name="sma_crossover",
        tickers=["TEST"],
        benchmark="TEST",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 12, 31),
        starting_cash=100_000.0,
        max_positions=10,
        max_alloc_pct=0.10,
        fee_per_trade=0.0,
        strategy_params={"sma_fast": 20, "sma_slow": 50},
        stop_config=stop_config,
    )
    return BacktestEngine(config)


def _make_fill(symbol="TEST", price=100.0, side=Side.BUY):
    return Fill(
        order_id="abc", symbol=symbol, side=side, quantity=100,
        price=price, commission=0.0, fill_date=date(2020, 6, 1), slippage=0.0,
    )


class TestSetStopsForFills:
    def test_pct_stop_loss(self):
        engine = _make_engine(StopConfig(stop_loss_pct=0.05))
        portfolio = Portfolio(cash=100_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        portfolio.positions["TEST"] = pos

        fills = [_make_fill(price=100.0)]
        engine._stop_mgr.set_stops_for_fills(fills, {}, portfolio)

        assert pos.stop_state.stop_loss == pytest.approx(95.0)

    def test_pct_take_profit(self):
        engine = _make_engine(StopConfig(take_profit_pct=0.20))
        portfolio = Portfolio(cash=100_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        portfolio.positions["TEST"] = pos

        fills = [_make_fill(price=100.0)]
        engine._stop_mgr.set_stops_for_fills(fills, {}, portfolio)

        assert pos.stop_state.take_profit == pytest.approx(120.0)

    def test_atr_stop_loss(self):
        engine = _make_engine(StopConfig(stop_loss_atr=2.0))
        portfolio = Portfolio(cash=100_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        portfolio.positions["TEST"] = pos

        today_data = {"TEST": pd.Series({"ATR": 3.0})}
        fills = [_make_fill(price=100.0)]
        engine._stop_mgr.set_stops_for_fills(fills, today_data, portfolio)

        # stop = 100 - 2.0 * 3.0 = 94.0
        assert pos.stop_state.stop_loss == pytest.approx(94.0)

    def test_tighter_of_pct_and_atr(self):
        # pct stop = 100*(1-0.05) = 95, ATR stop = 100 - 2*3 = 94
        # tighter = max(95, 94) = 95
        engine = _make_engine(StopConfig(stop_loss_pct=0.05, stop_loss_atr=2.0))
        portfolio = Portfolio(cash=100_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        portfolio.positions["TEST"] = pos

        today_data = {"TEST": pd.Series({"ATR": 3.0})}
        fills = [_make_fill(price=100.0)]
        engine._stop_mgr.set_stops_for_fills(fills, today_data, portfolio)

        assert pos.stop_state.stop_loss == pytest.approx(95.0)

    def test_trailing_stop_initialization(self):
        engine = _make_engine(StopConfig(trailing_stop_pct=0.08))
        portfolio = Portfolio(cash=100_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        portfolio.positions["TEST"] = pos

        fills = [_make_fill(price=100.0)]
        engine._stop_mgr.set_stops_for_fills(fills, {}, portfolio)

        assert pos.stop_state.trailing_stop_pct == 0.08
        assert pos.stop_state.trailing_high == 100.0

    def test_sell_fill_ignored(self):
        engine = _make_engine(StopConfig(stop_loss_pct=0.05))
        portfolio = Portfolio(cash=100_000.0)
        fills = [_make_fill(side=Side.SELL)]
        engine._stop_mgr.set_stops_for_fills(fills, {}, portfolio)
        # No crash, no stops set


class TestCheckStopTriggers:
    def _setup(self, stop_state):
        """Create portfolio with a position that has the given stop state."""
        portfolio = Portfolio(cash=0.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        pos.stop_state = stop_state
        portfolio.positions["TEST"] = pos
        return portfolio

    def test_stop_loss_triggers(self):
        engine = _make_engine(StopConfig(stop_loss_pct=0.05))
        ss = StopState(stop_loss=95.0)
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 94.0, "High": 101.0, "Close": 95.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        # Position should be closed
        assert "TEST" not in portfolio.positions
        assert len(portfolio.trade_log) == 1
        assert portfolio.cash > 0

    def test_take_profit_triggers(self):
        engine = _make_engine(StopConfig(take_profit_pct=0.20))
        ss = StopState(take_profit=120.0)
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 99.0, "High": 121.0, "Close": 119.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        assert "TEST" not in portfolio.positions
        assert len(portfolio.trade_log) == 1

    def test_trailing_stop_triggers(self):
        engine = _make_engine(StopConfig(trailing_stop_pct=0.10))
        ss = StopState(trailing_stop_pct=0.10, trailing_high=110.0)
        # trailing stop price = 110 * 0.90 = 99.0
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 98.0, "High": 105.0, "Close": 99.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        assert "TEST" not in portfolio.positions

    def test_no_trigger_within_bounds(self):
        engine = _make_engine(StopConfig(stop_loss_pct=0.05, take_profit_pct=0.20))
        ss = StopState(stop_loss=95.0, take_profit=120.0)
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 96.0, "High": 110.0, "Close": 105.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        # Position should still be open
        assert "TEST" in portfolio.positions
        assert portfolio.positions["TEST"].total_quantity == 100

    def test_cash_updated_on_stop(self):
        engine = _make_engine(StopConfig(stop_loss_pct=0.05))
        ss = StopState(stop_loss=95.0)
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 94.0, "High": 101.0, "Close": 95.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        # Sold 100 shares at stop price 95.0, fee=0
        assert portfolio.cash == pytest.approx(9500.0)

    def test_short_position_stop_loss_triggers_on_high(self):
        """Short stop_loss should trigger when High >= stop level (price rises against us)."""
        engine = _make_engine(StopConfig(stop_loss_pct=0.10))
        portfolio = Portfolio(cash=20_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(-100, 100.0, date(2020, 6, 1))  # short at 100
        pos.update_market_price(100.0)
        # Short stop_loss is above entry: 100 * 1.10 = 110
        pos.stop_state = StopState(stop_loss=110.0)
        portfolio.positions["TEST"] = pos

        # High reaches 111, which is >= 110 stop
        today_data = {"TEST": pd.Series({"Low": 105.0, "High": 111.0, "Close": 108.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        # Short should be covered
        assert "TEST" not in portfolio.positions
        assert len(portfolio.trade_log) == 1
        # Covered at 110 (stop price), paid 110*100 = 11000
        assert portfolio.cash == pytest.approx(20_000.0 - 110.0 * 100)

    def test_short_position_take_profit_triggers_on_low(self):
        """Short take_profit should trigger when Low <= target level (price falls in our favor)."""
        engine = _make_engine(StopConfig(take_profit_pct=0.20))
        portfolio = Portfolio(cash=20_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(-100, 100.0, date(2020, 6, 1))
        pos.update_market_price(100.0)
        # Short take_profit is below entry: 100 * 0.80 = 80
        pos.stop_state = StopState(take_profit=80.0)
        portfolio.positions["TEST"] = pos

        today_data = {"TEST": pd.Series({"Low": 79.0, "High": 95.0, "Close": 82.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        assert "TEST" not in portfolio.positions
        assert len(portfolio.trade_log) == 1

    def test_stop_and_take_profit_both_trigger_same_day(self):
        """When both stop_loss and take_profit could trigger on the same bar,
        stop_loss should take priority (it's checked first in the code)."""
        engine = _make_engine(StopConfig(stop_loss_pct=0.05, take_profit_pct=0.20))
        ss = StopState(stop_loss=95.0, take_profit=120.0)
        portfolio = self._setup(ss)

        # Wide bar: Low=94 triggers stop, High=121 triggers take_profit
        today_data = {"TEST": pd.Series({"Low": 94.0, "High": 121.0, "Close": 105.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        # Position should be closed
        assert "TEST" not in portfolio.positions
        assert len(portfolio.trade_log) == 1
        # stop_loss was checked first, so exit at stop price 95.0
        assert portfolio.trade_log[0].exit_price == 95.0

    def test_gap_through_stop(self):
        """Price gaps past the stop level; should still trigger at the stop price."""
        engine = _make_engine(StopConfig(stop_loss_pct=0.05))
        ss = StopState(stop_loss=95.0)
        portfolio = self._setup(ss)

        # Gap down: open at 90, low at 88 -- both well below stop of 95
        today_data = {"TEST": pd.Series({"Low": 88.0, "High": 92.0, "Close": 89.0})}
        engine._stop_mgr.check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        assert "TEST" not in portfolio.positions
        # Fills at the stop price (95.0), not the actual low
        assert portfolio.trade_log[0].exit_price == 95.0
        assert portfolio.cash == pytest.approx(9500.0)

    def test_trailing_stop_high_water_mark_updates(self):
        """Trailing stop high-water mark should ratchet upward with new highs."""
        engine = _make_engine(StopConfig(trailing_stop_pct=0.10))
        portfolio = Portfolio(cash=0.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        pos.stop_state = StopState(trailing_stop_pct=0.10, trailing_high=100.0)
        portfolio.positions["TEST"] = pos

        # Day 1: High=105 => trailing_high updates to 105
        today_data = {"TEST": pd.Series({"High": 105.0, "Low": 99.0, "Close": 103.0})}
        engine._stop_mgr.update_trailing_highs(portfolio, today_data)
        assert pos.stop_state.trailing_high == 105.0
        # trailing stop price = 105 * 0.90 = 94.5
        assert pos.stop_state.trailing_stop_price == pytest.approx(94.5)

        # Day 2: High=103 (lower) => trailing_high stays at 105
        today_data = {"TEST": pd.Series({"High": 103.0, "Low": 98.0, "Close": 101.0})}
        engine._stop_mgr.update_trailing_highs(portfolio, today_data)
        assert pos.stop_state.trailing_high == 105.0

        # Day 3: High=112 => trailing_high updates to 112
        today_data = {"TEST": pd.Series({"High": 112.0, "Low": 104.0, "Close": 110.0})}
        engine._stop_mgr.update_trailing_highs(portfolio, today_data)
        assert pos.stop_state.trailing_high == 112.0
        assert pos.stop_state.trailing_stop_price == pytest.approx(112.0 * 0.90)
