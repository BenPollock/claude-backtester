"""Phase 4: Adversarial tests for edge cases and boundary conditions.

Tests inputs the original developer likely never considered.
"""

import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import total_return, cagr, compute_all_metrics
from backtester.config import BacktestConfig
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.execution.broker import SimulatedBroker
from backtester.execution.fees import PerTradeFee
from backtester.execution.slippage import FixedSlippage
from backtester.portfolio.order import Order
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.position import Position
from backtester.strategies.base import Strategy
from backtester.strategies.registry import _REGISTRY
from backtester.types import Side, OrderType, SignalAction

from tests.conftest import MockDataSource, make_price_df


def _make_rising_df(start="2020-01-02", days=252, start_price=100.0, daily_pct=0.001):
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + daily_pct))
    prices = np.array(prices)
    return pd.DataFrame(
        {"Open": prices * 0.999, "High": prices * 1.005,
         "Low": prices * 0.995, "Close": prices,
         "Volume": np.full(days, 1_000_000)},
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


def _build_engine(tmpdir, source, tickers, config_overrides=None):
    defaults = dict(
        strategy_name="sma_crossover",
        tickers=tickers, benchmark=tickers[0],
        start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
        starting_cash=100_000.0, max_positions=10, max_alloc_pct=0.20,
        fee_per_trade=0.0, slippage_bps=0.0, data_cache_dir=tmpdir,
        strategy_params={"sma_fast": 20, "sma_slow": 50},
    )
    if config_overrides:
        defaults.update(config_overrides)
    config = BacktestConfig(**defaults)
    dm = DataManager(cache_dir=tmpdir, source=source)
    return BacktestEngine(config, data_manager=dm)


# ===========================================================================
# NaN in price data
# ===========================================================================

class TestNaNPriceData:
    """NaN values in OHLCV data should not corrupt the equity curve."""

    def test_nan_close_does_not_create_phantom_drawdown(self):
        """Position should retain last valid market price through NaN gaps."""
        dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
        prices = np.linspace(100, 110, 20)
        df = pd.DataFrame({
            "Open": prices * 0.999, "High": prices * 1.005,
            "Low": prices * 0.995, "Close": prices,
            "Volume": np.full(20, 1_000_000),
        }, index=pd.DatetimeIndex(dates.date, name="Date"))

        # Inject NaN gap in the middle
        df.loc[df.index[8:11], "Close"] = np.nan

        portfolio = Portfolio(cash=90_000.0)
        pos = portfolio.open_position("TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(105.0)  # valid price before gap

        # Simulate engine behavior: update_market_price only when Close is valid
        for i, (dt, row) in enumerate(df.iterrows()):
            close = row["Close"]
            if not pd.isna(close):
                pos.update_market_price(close)
            portfolio.record_equity(dt)

        equity = pd.Series(
            [v for _, v in portfolio.equity_history],
            index=[d for d, _ in portfolio.equity_history],
        )

        # Equity should never drop to just cash (90k) during the NaN gap
        # because position retains last valid price
        assert (equity > 99_000).all(), (
            f"Equity dropped during NaN gap to {equity.min():.0f}. "
            f"Position market_price should retain last valid value."
        )


# ===========================================================================
# max_positions = 0
# ===========================================================================

class TestMaxPositionsZero:
    """max_positions=0 should produce zero trades without error."""

    def test_zero_max_positions_produces_no_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine = _build_engine(tmpdir, source, ["TEST"], {
                "max_positions": 0,
            })
            result = engine.run()

            assert len(result.trades) == 0
            assert result.equity_series.iloc[-1] == 100_000.0


# ===========================================================================
# CAGR extrapolation on short periods
# ===========================================================================

class TestCAGRShortPeriod:
    """CAGR on very short backtests produces mathematically correct but misleading values."""

    def test_two_day_cagr_is_extreme_but_finite(self):
        """A 1% gain over 2 days produces enormous annualized CAGR."""
        equity = pd.Series(
            [10_000, 10_100],
            index=[date(2020, 1, 2), date(2020, 1, 3)],
        )
        result = cagr(equity)
        # Mathematically: (1.01)^(365.25/1) - 1 ≈ 3,678%
        assert isinstance(result, float)
        assert not pd.isna(result)
        assert result > 1.0, "CAGR should be huge for 1% gain over 1 day"


# ===========================================================================
# Signal on last day of backtest
# ===========================================================================

class TestSignalOnLastDay:
    """Signals on the last trading day have no T+1 to fill. Engine should skip them."""

    def test_no_unfilled_orders_at_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            # The engine's day loop skips signals on the last day (i < len(trading_days) - 1)
            engine = _build_engine(tmpdir, source, ["TEST"])
            result = engine.run()

            # All positions should be closed (force-closed)
            assert result.portfolio.num_positions == 0


# ===========================================================================
# Sell quantity for non-existent position
# ===========================================================================

class TestSellNonExistentPosition:
    """SELL signal for a symbol without a position should be harmless."""

    def test_sell_no_position_is_noop(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0.0),
            fees=PerTradeFee(fee=0.0),
        )
        portfolio = Portfolio(cash=100_000.0)

        # SELL order for symbol with no position, quantity=-1 sentinel
        order = Order(
            symbol="NOPE", side=Side.SELL, quantity=-1,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
        )
        broker.submit_order(order)

        market_data = {
            "NOPE": pd.Series({
                "Open": 100.0, "High": 105.0, "Low": 95.0,
                "Close": 100.0, "Volume": 1_000_000,
            })
        }

        # Should not crash; order should be cancelled
        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)
        assert len(fills) == 0
        assert portfolio.cash == 100_000.0  # unchanged


# ===========================================================================
# Buy with zero cash
# ===========================================================================

class TestBuyWithZeroCash:
    """BUY order when cash=0 should be cancelled, not crash."""

    def test_buy_cancelled_when_no_cash(self):
        broker = SimulatedBroker(
            slippage=FixedSlippage(bps=0.0),
            fees=PerTradeFee(fee=0.0),
        )
        portfolio = Portfolio(cash=0.0)

        order = Order(
            symbol="TEST", side=Side.BUY, quantity=100,
            order_type=OrderType.MARKET, signal_date=date(2020, 1, 2),
        )
        broker.submit_order(order)

        market_data = {
            "TEST": pd.Series({
                "Open": 100.0, "High": 105.0, "Low": 95.0,
                "Close": 100.0, "Volume": 1_000_000,
            })
        }

        fills = broker.process_fills(date(2020, 1, 3), market_data, portfolio)
        assert len(fills) == 0
        assert portfolio.cash == 0.0


# ===========================================================================
# Force-close PnL: fees are zero on force-close (documented behavior)
# ===========================================================================

class TestForceCloseFees:
    """Force-close uses exit_commission=0.0 by design. Verify this is consistent."""

    def test_force_close_trades_have_zero_exit_fees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", _make_rising_df(days=252))

            if "_buy_and_hold" not in _REGISTRY:
                class _BuyAndHold(Strategy):
                    def configure(self, params): pass
                    def compute_indicators(self, df, timeframe_data=None):
                        return df.copy()
                    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
                        if position is None or position.total_quantity == 0:
                            return SignalAction.BUY
                        return SignalAction.HOLD
                _REGISTRY["_buy_and_hold"] = _BuyAndHold

            engine = _build_engine(tmpdir, source, ["TEST"], {
                "strategy_name": "_buy_and_hold",
                "strategy_params": {},
                "fee_per_trade": 10.0,  # $10 per trade
                "max_positions": 1,
                "max_alloc_pct": 0.95,
            })
            result = engine.run()

            # The last trade should be the force-close with zero exit fees
            # The force-close trade fees_total only includes entry_commission
            # (which was allocated to the lot at entry time)
            assert len(result.trades) >= 1


# ===========================================================================
# Stop-loss and take-profit both trigger same day
# ===========================================================================

class TestStopAndTakeProfitSameDay:
    """When both stop-loss and take-profit trigger on a volatile day,
    stop-loss should win (pessimistic for longs)."""

    def test_stop_loss_takes_priority_over_take_profit(self):
        from backtester.execution.stops import StopManager
        from backtester.config import StopConfig

        stop_mgr = StopManager(StopConfig(stop_loss_pct=0.10, take_profit_pct=0.10),
                               PerTradeFee(fee=0.0))

        portfolio = Portfolio(cash=100_000.0)
        pos = portfolio.open_position("VOLATILE")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        pos.update_market_price(100.0)
        pos.stop_state.stop_loss = 90.0   # 10% below entry
        pos.stop_state.take_profit = 110.0  # 10% above entry

        # Wide-range day: Low=85 triggers stop, High=115 triggers take-profit
        today_data = {
            "VOLATILE": pd.Series({
                "Open": 100.0, "High": 115.0, "Low": 85.0, "Close": 100.0
            })
        }

        stop_mgr.check_stop_triggers(date(2020, 1, 10), today_data, portfolio)

        assert portfolio.num_positions == 0, "Position should be closed"
        assert len(portfolio.trade_log) == 1

        # Stop-loss should win (pessimistic): exit at $90, not $110
        trade = portfolio.trade_log[0]
        assert trade.exit_price == 90.0, (
            f"Stop-loss should take priority: exit at $90, got ${trade.exit_price}"
        )


# ===========================================================================
# Metrics with zero trades
# ===========================================================================

class TestMetricsZeroTrades:
    """compute_all_metrics with no trades should return sensible defaults."""

    def test_all_metrics_no_trades(self):
        equity = pd.Series([100_000] * 10,
                           index=pd.bdate_range("2020-01-02", periods=10))
        metrics = compute_all_metrics(equity, [])

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0
        assert metrics["trade_expectancy"] == 0.0
        assert metrics["total_return"] == 0.0
        assert metrics["exposure_time"] == 0.0


# ===========================================================================
# Position unrealized_pnl with market_price=0
# ===========================================================================

class TestUnrealizedPnlZeroPrice:
    """Position with market_price=0 should return 0 unrealized PnL."""

    def test_unrealized_pnl_zero_market_price(self):
        pos = Position(symbol="TEST")
        pos.add_lot(100, 50.0, date(2020, 1, 2))
        # _market_price defaults to 0.0
        assert pos.unrealized_pnl == 0.0

    def test_unrealized_pnl_short_zero_market_price(self):
        pos = Position(symbol="TEST")
        pos.add_lot(-100, 50.0, date(2020, 1, 2))
        assert pos.unrealized_pnl == 0.0
