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
        engine._set_stops_for_fills(fills, {}, portfolio)

        assert pos.stop_state.stop_loss == pytest.approx(95.0)

    def test_pct_take_profit(self):
        engine = _make_engine(StopConfig(take_profit_pct=0.20))
        portfolio = Portfolio(cash=100_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        portfolio.positions["TEST"] = pos

        fills = [_make_fill(price=100.0)]
        engine._set_stops_for_fills(fills, {}, portfolio)

        assert pos.stop_state.take_profit == pytest.approx(120.0)

    def test_atr_stop_loss(self):
        engine = _make_engine(StopConfig(stop_loss_atr=2.0))
        portfolio = Portfolio(cash=100_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        portfolio.positions["TEST"] = pos

        today_data = {"TEST": pd.Series({"ATR": 3.0})}
        fills = [_make_fill(price=100.0)]
        engine._set_stops_for_fills(fills, today_data, portfolio)

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
        engine._set_stops_for_fills(fills, today_data, portfolio)

        assert pos.stop_state.stop_loss == pytest.approx(95.0)

    def test_trailing_stop_initialization(self):
        engine = _make_engine(StopConfig(trailing_stop_pct=0.08))
        portfolio = Portfolio(cash=100_000.0)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 6, 1))
        portfolio.positions["TEST"] = pos

        fills = [_make_fill(price=100.0)]
        engine._set_stops_for_fills(fills, {}, portfolio)

        assert pos.stop_state.trailing_stop_pct == 0.08
        assert pos.stop_state.trailing_high == 100.0

    def test_sell_fill_ignored(self):
        engine = _make_engine(StopConfig(stop_loss_pct=0.05))
        portfolio = Portfolio(cash=100_000.0)
        fills = [_make_fill(side=Side.SELL)]
        engine._set_stops_for_fills(fills, {}, portfolio)
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
        engine._check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        # Position should be closed
        assert "TEST" not in portfolio.positions
        assert len(portfolio.trade_log) == 1
        assert portfolio.cash > 0

    def test_take_profit_triggers(self):
        engine = _make_engine(StopConfig(take_profit_pct=0.20))
        ss = StopState(take_profit=120.0)
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 99.0, "High": 121.0, "Close": 119.0})}
        engine._check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        assert "TEST" not in portfolio.positions
        assert len(portfolio.trade_log) == 1

    def test_trailing_stop_triggers(self):
        engine = _make_engine(StopConfig(trailing_stop_pct=0.10))
        ss = StopState(trailing_stop_pct=0.10, trailing_high=110.0)
        # trailing stop price = 110 * 0.90 = 99.0
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 98.0, "High": 105.0, "Close": 99.0})}
        engine._check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        assert "TEST" not in portfolio.positions

    def test_no_trigger_within_bounds(self):
        engine = _make_engine(StopConfig(stop_loss_pct=0.05, take_profit_pct=0.20))
        ss = StopState(stop_loss=95.0, take_profit=120.0)
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 96.0, "High": 110.0, "Close": 105.0})}
        engine._check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        # Position should still be open
        assert "TEST" in portfolio.positions
        assert portfolio.positions["TEST"].total_quantity == 100

    def test_cash_updated_on_stop(self):
        engine = _make_engine(StopConfig(stop_loss_pct=0.05))
        ss = StopState(stop_loss=95.0)
        portfolio = self._setup(ss)

        today_data = {"TEST": pd.Series({"Low": 94.0, "High": 101.0, "Close": 95.0})}
        engine._check_stop_triggers(date(2020, 7, 1), today_data, portfolio)

        # Sold 100 shares at stop price 95.0, fee=0
        assert portfolio.cash == pytest.approx(9500.0)
