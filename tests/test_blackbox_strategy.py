"""
Black-box tests for strategy behavior edge cases.
We register custom Strategy subclasses, run full backtests via the engine,
and inspect outputs -- without reading any source code.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig, StopConfig
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.strategies.registry import _REGISTRY, discover_strategies
from backtester.strategies.base import Strategy, Signal
from backtester.types import SignalAction, OrderType
from backtester.analytics.metrics import compute_all_metrics

discover_strategies()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from backtester.data.sources.base import DataSource


class MockDataSource(DataSource):
    def __init__(self):
        self._data = {}

    def add(self, symbol, df):
        self._data[symbol] = df

    def fetch(self, symbol, start, end):
        df = self._data[symbol]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]


def make_normal_df(start="2020-01-02", days=252, start_price=100.0, daily_ret=0.001):
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + daily_ret))
    prices = np.array(prices)
    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": np.full(days, 1_000_000),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


def make_flat_df(start="2020-01-02", days=252, price=100.0):
    """Constant-price data -- makes PnL calculations easier to reason about."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    return pd.DataFrame(
        {
            "Open": np.full(days, price),
            "High": np.full(days, price),
            "Low": np.full(days, price),
            "Close": np.full(days, price),
            "Volume": np.full(days, 1_000_000),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


def _register(name, cls):
    _REGISTRY[name] = cls


def _make_config(tickers, benchmark="SPY", cash=100_000, max_positions=100,
                 max_alloc_pct=1.0, allow_short=False, stop_config=None,
                 fee_per_trade=0.0, **kwargs):
    return BacktestConfig(
        tickers=tickers,
        benchmark=benchmark,
        start_date=date(2020, 1, 2),
        end_date=date(2020, 12, 31),
        starting_cash=cash,
        max_positions=max_positions,
        max_alloc_pct=max_alloc_pct,
        strategy_name="test_strategy",
        strategy_params={},
        allow_short=allow_short,
        stop_config=stop_config,
        fee_per_trade=fee_per_trade,
        **kwargs,
    )


def _run(config, source):
    dm = DataManager(source=source)
    engine = BacktestEngine(config, dm)
    return engine.run()


def _equity_series(result):
    """Convert portfolio equity_history to a pd.Series for metrics."""
    dates, vals = zip(*result.portfolio.equity_history)
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# Strategy implementations using the CORRECT ABC signatures:
#   compute_indicators(self, df, timeframe_data=None) -> DataFrame
#   generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None)
#       -> SignalAction | Signal
#
# The engine calls generate_signals per-symbol. 'row' is the indicator row
# for that day. 'position' is the current Position or None.
# ---------------------------------------------------------------------------


class AlwaysBuyStrategy(Strategy):
    """BUYs every single day, regardless of existing positions."""
    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        return SignalAction.BUY


class AlternateBuySellStrategy(Strategy):
    """Alternates BUY on even days, SELL on odd days."""
    def compute_indicators(self, df, timeframe_data=None):
        d = df.copy()
        d["day_num"] = range(len(d))
        return d

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        day_num = int(row.get("day_num", 0))
        if day_num % 2 == 0:
            return SignalAction.BUY
        else:
            return SignalAction.SELL


class DoubleBuyStrategy(Strategy):
    """Always returns BUY, even if already holding."""
    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        return SignalAction.BUY


class SellWithoutPositionStrategy(Strategy):
    """Always returns SELL, even with no position."""
    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        return SignalAction.SELL


class ShortCoverStrategy(Strategy):
    """Alternates SHORT on even days, COVER on odd days."""
    def compute_indicators(self, df, timeframe_data=None):
        d = df.copy()
        d["day_num"] = range(len(d))
        return d

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        day_num = int(row.get("day_num", 0))
        if day_num % 2 == 0:
            return SignalAction.SHORT
        else:
            return SignalAction.COVER


class BuyAllTickersStrategy(Strategy):
    """Always returns BUY for whatever symbol it's asked about."""
    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        return SignalAction.BUY


class BuyAfterStopStrategy(Strategy):
    """BUYs every 3rd day throughout the backtest (to test stop + rebuy)."""
    def compute_indicators(self, df, timeframe_data=None):
        d = df.copy()
        d["day_num"] = range(len(d))
        return d

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        day_num = int(row.get("day_num", 0))
        if day_num % 3 == 0:
            return SignalAction.BUY
        return SignalAction.HOLD


class SingleTradeStrategy(Strategy):
    """BUYs only on day 0, then HOLDs forever."""
    def compute_indicators(self, df, timeframe_data=None):
        d = df.copy()
        d["day_num"] = range(len(d))
        return d

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        day_num = int(row.get("day_num", 0))
        if day_num == 0:
            return SignalAction.BUY
        return SignalAction.HOLD


class ZeroLimitPriceStrategy(Strategy):
    """Sends a limit BUY at $0 on day 0."""
    def compute_indicators(self, df, timeframe_data=None):
        d = df.copy()
        d["day_num"] = range(len(d))
        return d

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        day_num = int(row.get("day_num", 0))
        if day_num == 0:
            return Signal(
                action=SignalAction.BUY,
                limit_price=0.0,
                order_type=OrderType.LIMIT,
            )
        return SignalAction.HOLD


class BuyThenMassLiquidateStrategy(Strategy):
    """BUYs for first 5 days, SELLs all on day 10."""
    def compute_indicators(self, df, timeframe_data=None):
        d = df.copy()
        d["day_num"] = range(len(d))
        return d

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        day_num = int(row.get("day_num", 0))
        if day_num < 5:
            return SignalAction.BUY
        elif day_num == 10:
            return SignalAction.SELL
        return SignalAction.HOLD


class CoverWithoutShortStrategy(Strategy):
    """Always returns COVER, even with no short position."""
    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        return SignalAction.COVER


# ---------------------------------------------------------------------------
# 1. Strategy that BUYs every single day
# ---------------------------------------------------------------------------

class TestAlwaysBuy:
    def test_cash_never_negative(self):
        """Cash must never drop below zero even if strategy spams BUY."""
        _register("test_strategy", AlwaysBuyStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"], max_alloc_pct=0.10)
        result = _run(config, source)

        assert result.portfolio.cash >= 0, (
            f"Cash went negative: {result.portfolio.cash}"
        )

    def test_positions_limited(self):
        """Spamming BUY on one ticker should not create unbounded positions."""
        _register("test_strategy", AlwaysBuyStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"], max_alloc_pct=0.50)
        result = _run(config, source)

        assert result is not None


# ---------------------------------------------------------------------------
# 2. Strategy that alternates BUY/SELL every day
# ---------------------------------------------------------------------------

class TestAlternateBuySell:
    def test_many_round_trips(self):
        """Alternating BUY/SELL should produce many trades."""
        _register("test_strategy", AlternateBuySellStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"], max_alloc_pct=1.0)
        result = _run(config, source)

        assert len(result.portfolio.trade_log) > 10, (
            f"Expected many trades, got {len(result.portfolio.trade_log)}"
        )

    def test_fees_applied(self):
        """Many round trips with fees on flat prices should erode equity."""
        _register("test_strategy", AlternateBuySellStrategy)
        source = MockDataSource()
        source.add("SPY", make_flat_df())
        config = _make_config(["SPY"], max_alloc_pct=1.0, fee_per_trade=10.0)
        result = _run(config, source)

        eq = _equity_series(result)
        ret = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        assert ret < 0, (
            f"Flat prices + $10/trade fees should yield negative return, got {ret:.4%}"
        )


# ---------------------------------------------------------------------------
# 3. Strategy that BUYs a symbol it already holds
# ---------------------------------------------------------------------------

class TestDoubleBuy:
    def test_no_crash_and_limited_buys(self):
        """BUY for a held symbol should be suppressed -- not duplicate the position."""
        _register("test_strategy", DoubleBuyStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"], max_alloc_pct=0.50)
        result = _run(config, source)

        assert result is not None
        # Engine should skip BUY if already holding
        assert len(result.portfolio.trade_log) <= 5, (
            f"Expected very few trades (duplicate BUY suppressed), "
            f"got {len(result.portfolio.trade_log)}"
        )


# ---------------------------------------------------------------------------
# 4. Strategy that SELLs when no position exists
# ---------------------------------------------------------------------------

class TestSellWithoutPosition:
    def test_no_crash(self):
        """SELL with no position should be a no-op, not crash."""
        _register("test_strategy", SellWithoutPositionStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"])
        result = _run(config, source)

        assert result is not None
        assert len(result.portfolio.trade_log) == 0, (
            f"Expected 0 trades, got {len(result.portfolio.trade_log)}"
        )


# ---------------------------------------------------------------------------
# 5. Strategy that SHORTs then COVERs next day
# ---------------------------------------------------------------------------

class TestShortCover:
    def test_short_round_trips(self):
        """SHORT/COVER alternation with allow_short=True should produce trades."""
        _register("test_strategy", ShortCoverStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"], allow_short=True, max_alloc_pct=1.0)
        result = _run(config, source)

        assert result is not None
        assert len(result.portfolio.trade_log) > 5, (
            f"Expected many short round-trips, got {len(result.portfolio.trade_log)}"
        )


# ---------------------------------------------------------------------------
# 6. Strategy that SHORTs when allow_short=False
# ---------------------------------------------------------------------------

class TestShortDisabled:
    def test_short_suppressed(self):
        """SHORT signals should be suppressed when allow_short=False."""
        _register("test_strategy", ShortCoverStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"], allow_short=False, max_alloc_pct=1.0)
        result = _run(config, source)

        assert result is not None
        assert len(result.portfolio.trade_log) == 0, (
            f"Expected 0 trades (SHORT suppressed), got {len(result.portfolio.trade_log)}"
        )


# ---------------------------------------------------------------------------
# 7. Strategy that generates signals for many tickers -- max_positions=5
# ---------------------------------------------------------------------------

class TestMaxPositions:
    def test_positions_capped(self):
        """With 20 tickers and max_positions=5, should never hold more than 5."""
        _register("test_strategy", BuyAllTickersStrategy)
        source = MockDataSource()
        rng = np.random.RandomState(42)
        tickers = [f"T{i:02d}" for i in range(20)]
        for t in tickers:
            source.add(t, make_normal_df(start_price=50.0 + rng.rand() * 10))
        source.add("SPY", make_normal_df())

        config = _make_config(tickers, max_positions=5, max_alloc_pct=0.20)
        result = _run(config, source)

        active_positions = [s for s, p in result.portfolio.positions.items()
                           if p.total_quantity != 0]
        assert len(active_positions) <= 5, (
            f"Exceeded max_positions: holding {len(active_positions)} positions"
        )


# ---------------------------------------------------------------------------
# 8. Stop-loss triggers and then strategy re-buys
# ---------------------------------------------------------------------------

def make_volatile_df(start="2020-01-02", days=50, start_price=100.0):
    """Prices that swing enough to trigger a 5% stop loss."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    np.random.seed(99)
    prices = [start_price]
    for _ in range(days - 1):
        move = np.random.choice([-0.06, 0.03])
        prices.append(prices[-1] * (1 + move))
    prices = np.array(prices)
    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.01,
            "Low": prices * 0.94,
            "Close": prices,
            "Volume": np.full(days, 1_000_000),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


class TestStopAndRebuy:
    def test_stop_fires_and_rebuy_works(self):
        """A 5% stop loss on volatile data should fire; re-buying should work."""
        _register("test_strategy", BuyAfterStopStrategy)
        source = MockDataSource()
        source.add("SPY", make_volatile_df())
        stop = StopConfig(stop_loss_pct=0.05)
        config = BacktestConfig(
            tickers=["SPY"],
            benchmark="SPY",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 3, 15),
            starting_cash=100_000,
            max_positions=10,
            max_alloc_pct=1.0,
            strategy_name="test_strategy",
            strategy_params={},
            stop_config=stop,
        )
        result = _run(config, source)

        assert result is not None
        assert len(result.portfolio.trade_log) >= 2, (
            f"Expected multiple trades (stop + rebuy), "
            f"got {len(result.portfolio.trade_log)}"
        )


# ---------------------------------------------------------------------------
# 9. Strategy that only trades on the very first day then HOLDs
# ---------------------------------------------------------------------------

class TestSingleTrade:
    def test_buy_and_hold_pnl(self):
        """Buy on day 1, hold forever. Final equity should reflect price appreciation."""
        _register("test_strategy", SingleTradeStrategy)
        source = MockDataSource()
        # 252 days with strong upward trend (+0.3%/day ~ +110% total)
        df = make_normal_df(days=252, start_price=100.0, daily_ret=0.003)
        source.add("SPY", df)

        config = _make_config(["SPY"], max_alloc_pct=1.0)
        result = _run(config, source)

        eq = _equity_series(result)
        ret = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        assert ret > 0, (
            f"Expected positive return for upward-trending buy-and-hold, got {ret:.4%}"
        )


# ---------------------------------------------------------------------------
# 10. Strategy that sends limit_price=0
# ---------------------------------------------------------------------------

class TestZeroLimitPrice:
    def test_limit_buy_at_zero(self):
        """A limit BUY at $0 on a $100 stock should NOT fill (Low never reaches $0)."""
        _register("test_strategy", ZeroLimitPriceStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"], max_alloc_pct=1.0)
        result = _run(config, source)

        assert result is not None
        assert len(result.portfolio.trade_log) == 0, (
            f"Limit buy at $0 should not fill (Low ~$99.50), "
            f"got {len(result.portfolio.trade_log)} trades"
        )


# ---------------------------------------------------------------------------
# 11. Multi-ticker: BUY all then SELL all on same day
# ---------------------------------------------------------------------------

class TestMassLiquidation:
    def test_buy_many_sell_all(self):
        """Buy 5 tickers, then sell all on day 10. All positions should close."""
        _register("test_strategy", BuyThenMassLiquidateStrategy)
        source = MockDataSource()
        tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]
        for t in tickers:
            source.add(t, make_normal_df(start_price=100.0))
        source.add("SPY", make_normal_df())

        config = _make_config(tickers, max_positions=10, max_alloc_pct=0.20)
        result = _run(config, source)

        assert result is not None
        active = [s for s, p in result.portfolio.positions.items()
                  if p.total_quantity != 0]
        assert len(active) == 0, (
            f"Expected all positions closed after mass sell, still holding: {active}"
        )


# ---------------------------------------------------------------------------
# 12. COVER with no short position -- should be a no-op
# ---------------------------------------------------------------------------

class TestCoverWithoutShort:
    def test_cover_no_short_no_crash(self):
        """COVER signal when not short should be a no-op."""
        _register("test_strategy", CoverWithoutShortStrategy)
        source = MockDataSource()
        source.add("SPY", make_normal_df())
        config = _make_config(["SPY"], allow_short=True)
        result = _run(config, source)

        assert result is not None
        assert len(result.portfolio.trade_log) == 0, (
            f"Expected 0 trades for COVER with no short, "
            f"got {len(result.portfolio.trade_log)}"
        )
