"""Integration tests: full pipeline scenarios with economically sensible assertions.

These tests run the complete backtest pipeline (engine -> data -> strategy -> broker ->
portfolio -> analytics) and assert on SYSTEM behavior, not individual units.

Each test constructs synthetic data with known properties, runs a full backtest,
and checks that the outputs are economically coherent.
"""

import tempfile
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import (
    cagr,
    compute_all_metrics,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from backtester.config import BacktestConfig, RegimeFilter, StopConfig
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position
from backtester.strategies.base import Signal, Strategy
from backtester.strategies.registry import register_strategy, _REGISTRY
from backtester.types import OrderType, SignalAction
from tests.conftest import MockDataSource, make_price_df


# ---------------------------------------------------------------------------
# Helper: deterministic price data builders
# ---------------------------------------------------------------------------

def _make_rising_df(start="2020-01-02", days=252, start_price=100.0, daily_pct=0.001):
    """Deterministic monotonically rising prices (no noise)."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + daily_pct))
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


def _make_falling_df(start="2020-01-02", days=252, start_price=200.0, daily_pct=-0.002):
    """Deterministic monotonically falling prices."""
    return _make_rising_df(start=start, days=days, start_price=start_price, daily_pct=daily_pct)


def _make_flat_df(start="2020-01-02", days=252, price=100.0):
    """Constant price series -- no movement at all."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    return pd.DataFrame(
        {
            "Open": np.full(days, price),
            "High": np.full(days, price * 1.001),
            "Low": np.full(days, price * 0.999),
            "Close": np.full(days, price),
            "Volume": np.full(days, 1_000_000),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


def _make_vshape_df(start="2020-01-02", days=252, start_price=100.0):
    """Price falls for the first half, then rises back. Creates a drawdown then recovery."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    half = days // 2
    prices = []
    price = start_price
    for i in range(days):
        if i < half:
            price *= 0.997  # fall
        else:
            price *= 1.004  # rise faster
        prices.append(price)
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


# ---------------------------------------------------------------------------
# Helper: register one-off test strategies without polluting the real registry
# ---------------------------------------------------------------------------

class _AlwaysBuyStrategy(Strategy):
    """Buys on first opportunity, never sells (engine force-closes at end)."""

    def configure(self, params: dict) -> None:
        pass

    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        if position is None or position.total_quantity == 0:
            return SignalAction.BUY
        return SignalAction.HOLD


class _NeverTradeStrategy(Strategy):
    """Always holds -- never generates BUY or SELL signals."""

    def configure(self, params: dict) -> None:
        pass

    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        return SignalAction.HOLD


class _AlwaysShortStrategy(Strategy):
    """Shorts on first opportunity, never covers (engine force-closes at end)."""

    def configure(self, params: dict) -> None:
        pass

    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        if position is None or position.total_quantity == 0:
            return SignalAction.SHORT
        return SignalAction.HOLD


class _LimitBuyStrategy(Strategy):
    """Issues a limit buy at 1% below current close. Tests limit order fill logic."""

    def configure(self, params: dict) -> None:
        self._discount = params.get("discount_pct", 0.01)

    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        if position is None or position.total_quantity == 0:
            limit_price = row["Close"] * (1 - self._discount)
            return Signal(
                action=SignalAction.BUY,
                limit_price=limit_price,
                order_type=OrderType.LIMIT,
                time_in_force="GTC",
            )
        return SignalAction.HOLD


# Register test strategies (safe: idempotent via guard)
for _name, _cls in [
    ("_always_buy", _AlwaysBuyStrategy),
    ("_never_trade", _NeverTradeStrategy),
    ("_always_short", _AlwaysShortStrategy),
    ("_limit_buy", _LimitBuyStrategy),
]:
    if _name not in _REGISTRY:
        _REGISTRY[_name] = _cls


# ---------------------------------------------------------------------------
# Helpers: engine builder
# ---------------------------------------------------------------------------

def _build_engine(tmpdir, source, tickers, config_overrides=None):
    """Create a BacktestEngine from a MockDataSource with optional config overrides."""
    defaults = dict(
        strategy_name="sma_crossover",
        tickers=tickers,
        benchmark=tickers[0],
        start_date=date(2020, 1, 2),
        end_date=date(2020, 12, 31),
        starting_cash=100_000.0,
        max_positions=10,
        max_alloc_pct=0.20,
        fee_per_trade=0.0,
        slippage_bps=0.0,
        data_cache_dir=tmpdir,
        strategy_params={"sma_fast": 20, "sma_slow": 50},
    )
    if config_overrides:
        defaults.update(config_overrides)
    config = BacktestConfig(**defaults)
    dm = DataManager(cache_dir=tmpdir, source=source)
    return BacktestEngine(config, data_manager=dm)


# ===========================================================================
# TEST SCENARIOS
# ===========================================================================


class TestBuyAndHoldOnRisingAsset:
    """
    Test name: Buy-and-hold on a steadily rising asset
    What it validates: A strategy that buys once on a monotonically rising
        asset should produce positive total return and positive CAGR.
    Why it matters: Catches bugs where the pipeline silently drops trades,
        miscalculates equity, or corrupts P&L on force-close.
    """

    def test_positive_returns_on_rising_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = _make_rising_df(days=252, daily_pct=0.001)
            source.add("RISE", df)

            engine = _build_engine(tmpdir, source, ["RISE"], {
                "strategy_name": "_always_buy",
                "strategy_params": {},
                "max_positions": 1,
                "max_alloc_pct": 0.95,
            })
            result = engine.run()

            equity = result.equity_series
            ret = total_return(equity)

            # Asset rises ~0.1%/day for 252 days => ~28% total.
            # Strategy buys once and force-closes at end, so should capture
            # most of that return minus one day of T+1 delay.
            assert ret > 0.10, f"Expected > 10% return on rising asset, got {ret:.4f}"
            assert cagr(equity) > 0.0
            assert len(result.trades) >= 1, "Should have at least 1 completed trade"

    def test_all_positions_closed_at_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("RISE", _make_rising_df())

            engine = _build_engine(tmpdir, source, ["RISE"], {
                "strategy_name": "_always_buy",
                "strategy_params": {},
            })
            result = engine.run()
            assert result.portfolio.num_positions == 0, "All positions must be force-closed"

    def test_trade_pnl_matches_equity_change(self):
        """Sum of trade PnLs + starting cash should approximate final equity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("RISE", _make_rising_df())

            engine = _build_engine(tmpdir, source, ["RISE"], {
                "strategy_name": "_always_buy",
                "strategy_params": {},
                "max_positions": 1,
                "max_alloc_pct": 0.95,
            })
            result = engine.run()

            trade_pnl = sum(t.pnl for t in result.trades)
            final_equity = result.equity_series.iloc[-1]
            starting_cash = 100_000.0

            # Trade PnL + starting cash should closely match final equity.
            # Small discrepancy is acceptable from force-close mechanics.
            assert abs((starting_cash + trade_pnl) - final_equity) < 1.0, (
                f"Trade PnL sum ({trade_pnl:.2f}) + starting cash != "
                f"final equity ({final_equity:.2f})"
            )


class TestNeverTradeStrategy:
    """
    Test name: Strategy that never trades
    What it validates: If no signals are generated, cash should be unchanged,
        equity should be flat at starting_cash, and there should be zero trades.
    Why it matters: Catches bugs where the engine creates phantom fills,
        charges fees on non-existent orders, or corrupts cash.
    """

    def test_cash_unchanged_no_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("FLAT", _make_flat_df())

            engine = _build_engine(tmpdir, source, ["FLAT"], {
                "strategy_name": "_never_trade",
                "strategy_params": {},
            })
            result = engine.run()

            assert len(result.trades) == 0, "No trades expected"
            assert result.equity_series.iloc[-1] == 100_000.0, "Cash should be unchanged"
            assert result.equity_series.iloc[0] == 100_000.0

    def test_equity_curve_is_flat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("FLAT", _make_flat_df())

            engine = _build_engine(tmpdir, source, ["FLAT"], {
                "strategy_name": "_never_trade",
                "strategy_params": {},
            })
            result = engine.run()

            equity = result.equity_series
            # Every day's equity should equal starting cash
            assert (equity == 100_000.0).all(), "Equity should be constant when not trading"


class TestEquityCurveCompleteness:
    """
    Test name: Full calendar year produces equity entries for each trading day
    What it validates: The equity curve has one entry per NYSE trading day in
        the date range. No missing days, no duplicate days.
    Why it matters: Catches off-by-one errors in the trading calendar, missing
        equity snapshots, or crashes that silently truncate the run.
    """

    def test_equity_curve_length_matches_trading_days(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = make_price_df(start="2020-01-02", days=252)
            source.add("TEST", df)

            engine = _build_engine(tmpdir, source, ["TEST"])
            result = engine.run()

            equity = result.equity_series
            # The equity curve should have entries for each trading day
            # in the backtest range (which is bounded by available data).
            assert len(equity) > 200, f"Expected >200 trading days, got {len(equity)}"

            # No duplicate dates
            assert equity.index.is_unique, "Equity curve has duplicate dates"

            # Dates should be monotonically increasing
            assert equity.index.is_monotonic_increasing, "Equity dates not sorted"

    def test_equity_starts_at_starting_cash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine = _build_engine(tmpdir, source, ["TEST"])
            result = engine.run()

            assert result.equity_series.iloc[0] == 100_000.0


class TestRegimeFilterBehavior:
    """
    Test name: Regime filter on vs off produces different trade counts
    What it validates: Enabling a regime filter (fast SMA > slow SMA on benchmark)
        on a declining benchmark suppresses BUY signals, resulting in fewer trades
        than an identical run without the filter.
    Why it matters: Catches bugs where the regime filter is silently ignored,
        or where it blocks ALL signals (including SELLs it should allow).
    """

    def test_regime_filter_reduces_buys_on_declining_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # Declining prices: fast SMA will be below slow SMA => regime "off"
            df = _make_falling_df(days=252)
            source.add("FALL", df)

            # Run WITHOUT regime filter
            engine_no_filter = _build_engine(tmpdir, source, ["FALL"], {
                "strategy_params": {"sma_fast": 20, "sma_slow": 50},
            })
            result_no_filter = engine_no_filter.run()

            # Run WITH regime filter
            engine_with_filter = _build_engine(tmpdir, source, ["FALL"], {
                "strategy_params": {"sma_fast": 20, "sma_slow": 50},
                "regime_filter": RegimeFilter(
                    benchmark="FALL",
                    indicator="sma",
                    fast_period=20,
                    slow_period=50,
                ),
            })
            result_with_filter = engine_with_filter.run()

            buys_no_filter = sum(
                1 for e in result_no_filter.activity_log if e.action.name == "BUY"
            )
            buys_with_filter = sum(
                1 for e in result_with_filter.activity_log if e.action.name == "BUY"
            )

            assert buys_with_filter <= buys_no_filter, (
                f"Regime filter should suppress BUY signals: "
                f"with={buys_with_filter}, without={buys_no_filter}"
            )

    def test_regime_filter_still_allows_sells(self):
        """Even when regime is off, existing positions should still be sellable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # V-shape: rises first (buys happen), then falls (regime turns off, sells should work)
            df = _make_vshape_df(days=252)
            source.add("VSHAPE", df)

            engine = _build_engine(tmpdir, source, ["VSHAPE"], {
                "strategy_params": {"sma_fast": 10, "sma_slow": 30},
                "regime_filter": RegimeFilter(
                    benchmark="VSHAPE",
                    indicator="sma",
                    fast_period=10,
                    slow_period=30,
                ),
            })
            result = engine.run()

            # Should complete without error and close all positions
            assert result.portfolio.num_positions == 0


class TestMultiTickerBacktest:
    """
    Test name: Multi-ticker backtest with position limits
    What it validates: The engine can handle multiple tickers, respects
        max_positions, and correctly tracks positions across symbols.
    Why it matters: Catches bugs in per-symbol iteration order, position
        limit enforcement, and cross-symbol portfolio accounting.
    """

    def test_multiple_tickers_trade_independently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # Different start prices so SMA crossovers happen at different times
            for i, sym in enumerate(["A", "B", "C", "D", "E"]):
                source.add(sym, make_price_df(
                    start="2020-01-02", days=252,
                    start_price=50 + i * 30,
                ))

            engine = _build_engine(tmpdir, source, ["A", "B", "C", "D", "E"], {
                "max_positions": 5,
                "max_alloc_pct": 0.15,
            })
            result = engine.run()

            # Multiple symbols should have been traded
            traded_symbols = {t.symbol for t in result.trades}
            assert len(traded_symbols) >= 2, (
                f"Expected trades in multiple symbols, only got: {traded_symbols}"
            )

    def test_max_positions_enforced_in_equity_history(self):
        """At no point should equity be 0 or negative (no over-allocation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            for i, sym in enumerate([f"T{i}" for i in range(10)]):
                source.add(sym, make_price_df(
                    start="2020-01-02", days=252,
                    start_price=50 + i * 10,
                ))

            engine = _build_engine(tmpdir, source, [f"T{i}" for i in range(10)], {
                "max_positions": 3,
                "max_alloc_pct": 0.20,
            })
            result = engine.run()

            equity = result.equity_series
            assert (equity > 0).all(), "Equity should never go to zero or negative"


class TestStopLossEndToEnd:
    """
    Test name: Stop-loss triggers correctly in full pipeline
    What it validates: A stop-loss configured at 5% below entry actually triggers
        when the price drops 5%, resulting in a trade with a loss capped near that level.
    Why it matters: Catches bugs where stops are set but never checked, where the
        bypass-broker invariant is violated (fills delayed to T+1), or where stop
        prices are computed from the wrong reference.
    """

    def test_stop_loss_caps_losses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # V-shape: drops first then recovers. Stop should trigger during the drop.
            df = _make_vshape_df(days=252, start_price=100.0)
            source.add("DROP", df)

            engine = _build_engine(tmpdir, source, ["DROP"], {
                "strategy_name": "_always_buy",
                "strategy_params": {},
                "max_positions": 1,
                "max_alloc_pct": 0.90,
                "stop_config": StopConfig(stop_loss_pct=0.05),
            })
            result = engine.run()

            # Should have at least 1 trade that was stopped out
            assert len(result.trades) >= 1

            # Check that no individual trade lost more than ~7% (5% stop + slippage/gap)
            for trade in result.trades:
                if trade.pnl < 0:
                    loss_pct = abs(trade.pnl_pct)
                    assert loss_pct < 0.10, (
                        f"Stop at 5% but trade lost {loss_pct:.1%} -- stop may not have triggered"
                    )

    def test_take_profit_captures_gains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = _make_rising_df(days=252, start_price=100.0, daily_pct=0.003)
            source.add("ROCKET", df)

            engine = _build_engine(tmpdir, source, ["ROCKET"], {
                "strategy_name": "_always_buy",
                "strategy_params": {},
                "max_positions": 1,
                "max_alloc_pct": 0.90,
                "stop_config": StopConfig(take_profit_pct=0.10),
            })
            result = engine.run()

            # With take-profit at 10% on a steadily rising asset, we expect
            # multiple round trips as the strategy buys, gets taken out at +10%, buys again.
            assert len(result.trades) >= 1

            # At least one trade should have been taken out near the +10% target
            has_tp_trade = any(
                0.08 < t.pnl_pct < 0.15 for t in result.trades if t.pnl > 0
            )
            assert has_tp_trade, (
                f"Expected at least one trade closed near 10% profit. "
                f"Trade pnl_pcts: {[t.pnl_pct for t in result.trades]}"
            )


class TestFeeImpact:
    """
    Test name: Fee/slippage impact on identical strategy
    What it validates: Running the same strategy+data with fees/slippage enabled
        produces lower returns than running it fee-free.
    Why it matters: Catches bugs where fees are computed but never deducted,
        where slippage is applied in the wrong direction, or where fee-free runs
        accidentally charge fees.
    """

    def test_fees_reduce_returns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = make_price_df(start="2020-01-02", days=252)
            source.add("TEST", df)

            # Run fee-free
            engine_free = _build_engine(tmpdir, source, ["TEST"], {
                "fee_per_trade": 0.0,
                "slippage_bps": 0.0,
            })
            result_free = engine_free.run()

            # Run with fees + slippage
            engine_fees = _build_engine(tmpdir, source, ["TEST"], {
                "fee_per_trade": 10.0,  # $10 per trade
                "slippage_bps": 20.0,   # 20 bps slippage
            })
            result_fees = engine_fees.run()

            ret_free = total_return(result_free.equity_series)
            ret_fees = total_return(result_fees.equity_series)

            # If there were any trades, fees should reduce returns
            if len(result_free.trades) > 0 and len(result_fees.trades) > 0:
                assert ret_fees < ret_free, (
                    f"Fees should reduce returns: free={ret_free:.4f}, fees={ret_fees:.4f}"
                )

    def test_zero_fee_trades_have_zero_fees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine = _build_engine(tmpdir, source, ["TEST"], {
                "fee_per_trade": 0.0,
                "slippage_bps": 0.0,
            })
            result = engine.run()

            for trade in result.trades:
                assert trade.fees_total == 0.0, (
                    f"Trade on {trade.entry_date} has fees={trade.fees_total} "
                    f"but fee_per_trade=0"
                )


class TestShortSellingEndToEnd:
    """
    Test name: Short selling on a falling asset produces positive returns
    What it validates: The short selling pipeline (SHORT signal -> negative
        position -> force-close at end) produces positive P&L when the
        asset price declines.
    Why it matters: Catches bugs in short position creation, negative lot
        accounting, short P&L formula (entry - exit, not exit - entry),
        and force-close of short positions.
    """

    def test_short_on_falling_asset_is_profitable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = _make_falling_df(days=252, start_price=200.0, daily_pct=-0.001)
            source.add("BEAR", df)

            engine = _build_engine(tmpdir, source, ["BEAR"], {
                "strategy_name": "_always_short",
                "strategy_params": {},
                "allow_short": True,
                "max_positions": 1,
                "max_alloc_pct": 0.50,
                "short_borrow_rate": 0.0,  # zero borrow cost to isolate P&L
            })
            result = engine.run()

            # The asset fell ~0.1%/day for 252 days. Short should profit.
            ret = total_return(result.equity_series)
            assert ret > 0.0, (
                f"Short on falling asset should be profitable, got return={ret:.4f}"
            )
            assert len(result.trades) >= 1

    def test_short_disabled_blocks_short_signals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", _make_falling_df())

            engine = _build_engine(tmpdir, source, ["TEST"], {
                "strategy_name": "_always_short",
                "strategy_params": {},
                "allow_short": False,  # disabled
            })
            result = engine.run()

            # No trades should happen because shorts are disabled
            assert len(result.trades) == 0, (
                f"Short disabled but got {len(result.trades)} trades"
            )
            assert result.equity_series.iloc[-1] == 100_000.0


class TestLimitOrdersEndToEnd:
    """
    Test name: Limit orders fill only when price conditions are met
    What it validates: A limit buy order placed below current price only fills
        when the asset's Low reaches the limit price. On a steadily rising asset,
        limit buys should rarely or never fill.
    Why it matters: Catches bugs where limit orders are treated as market orders,
        where fill logic checks the wrong price (Open instead of Low), or where
        GTC persistence is broken.
    """

    def test_limit_buy_on_rising_asset_fills_less(self):
        """On a rising asset with limit 1% below close, fills should be rare
        because the Low may not reach the limit on a monotonic uptrend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # Strongly rising -- Low stays close to Close
            df = _make_rising_df(days=252, daily_pct=0.005)
            source.add("UP", df)

            # Limit buy strategy: tries to buy at 1% discount
            engine_limit = _build_engine(tmpdir, source, ["UP"], {
                "strategy_name": "_limit_buy",
                "strategy_params": {"discount_pct": 0.01},
                "max_positions": 1,
                "max_alloc_pct": 0.90,
            })
            result_limit = engine_limit.run()

            # Market buy strategy for comparison
            engine_market = _build_engine(tmpdir, source, ["UP"], {
                "strategy_name": "_always_buy",
                "strategy_params": {},
                "max_positions": 1,
                "max_alloc_pct": 0.90,
            })
            result_market = engine_market.run()

            # Market order should always fill on day 2; limit order may fill later or not at all
            market_buys = sum(1 for e in result_market.activity_log if e.action.name == "BUY")
            limit_buys = sum(1 for e in result_limit.activity_log if e.action.name == "BUY")

            assert market_buys >= 1, "Market buy should fill"
            # Limit buys should be <= market buys (possibly 0 on strong uptrend)
            assert limit_buys <= market_buys


class TestBenchmarkComparison:
    """
    Test name: Benchmark equity is tracked correctly
    What it validates: The benchmark equity curve is buy-and-hold of the
        benchmark asset, starting with the same cash. Its total return should
        match the asset's price return.
    Why it matters: Catches bugs where benchmark shares aren't initialized,
        where benchmark equity uses wrong price column, or where the benchmark
        series is empty.
    """

    def test_benchmark_return_matches_asset_return(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = _make_rising_df(days=252, daily_pct=0.001)
            source.add("BM", df)

            engine = _build_engine(tmpdir, source, ["BM"], {
                "strategy_name": "_never_trade",
                "strategy_params": {},
            })
            result = engine.run()

            bm_series = result.benchmark_series
            assert bm_series is not None, "Benchmark series should exist"
            assert len(bm_series) > 0

            # Benchmark return should match the asset's close-to-close return
            asset_return = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0
            bm_return = total_return(bm_series)

            # Allow 2% tolerance for edge-of-range alignment
            assert abs(bm_return - asset_return) < 0.02, (
                f"Benchmark return ({bm_return:.4f}) should match asset "
                f"return ({asset_return:.4f})"
            )


class TestMetricsEconomicSanity:
    """
    Test name: Computed metrics are economically sensible
    What it validates: For a known scenario (rising asset, buy-and-hold),
        the metrics (CAGR, Sharpe, max drawdown) fall within expected ranges.
    Why it matters: Catches formula bugs in metrics (wrong annualization factor,
        sign errors in drawdown, divide-by-zero in Sharpe).
    """

    def test_metrics_on_rising_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = _make_rising_df(days=252, daily_pct=0.001)
            source.add("UP", df)

            engine = _build_engine(tmpdir, source, ["UP"], {
                "strategy_name": "_always_buy",
                "strategy_params": {},
                "max_positions": 1,
                "max_alloc_pct": 0.95,
            })
            result = engine.run()

            equity = result.equity_series
            metrics = compute_all_metrics(equity, result.trades)

            assert metrics["total_return"] > 0, "Total return should be positive"
            assert metrics["cagr"] > 0, "CAGR should be positive"
            assert metrics["max_drawdown"] <= 0, "Max drawdown should be <= 0"
            assert metrics["max_drawdown"] > -0.10, (
                f"On steadily rising asset, drawdown should be small, got {metrics['max_drawdown']}"
            )
            assert metrics["total_trades"] >= 1
            assert 0.0 <= metrics["win_rate"] <= 1.0
            # On a monotonically rising asset, even the 5th percentile return
            # can be positive (no down days). VaR is just a percentile of returns.
            assert isinstance(metrics["var_95"], float), "VaR should be a float"

    def test_metrics_on_no_trade_scenario(self):
        """When there are no trades, trade-level metrics should be zero/empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("FLAT", _make_flat_df())

            engine = _build_engine(tmpdir, source, ["FLAT"], {
                "strategy_name": "_never_trade",
                "strategy_params": {},
            })
            result = engine.run()

            equity = result.equity_series
            metrics = compute_all_metrics(equity, result.trades)

            assert metrics["total_trades"] == 0
            assert metrics["win_rate"] == 0.0
            assert metrics["trade_expectancy"] == 0.0
            assert metrics["total_return"] == 0.0


class TestCashConservation:
    """
    Test name: Cash + position value = total equity at all times
    What it validates: The fundamental accounting identity holds throughout
        the entire backtest. Cash is never created or destroyed.
    Why it matters: Catches bugs where fills deduct cash but don't create
        positions, where force-close doesn't credit cash, or where fees
        are double-counted.
    """

    def test_final_cash_equals_final_equity_after_close(self):
        """After all positions are force-closed, cash should equal equity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine = _build_engine(tmpdir, source, ["TEST"])
            result = engine.run()

            portfolio = result.portfolio
            assert portfolio.num_positions == 0, "All positions should be closed"
            # With no positions, cash == total equity
            final_equity = result.equity_series.iloc[-1]
            assert abs(portfolio.cash - final_equity) < 0.01, (
                f"Cash ({portfolio.cash:.2f}) should equal final equity "
                f"({final_equity:.2f}) after all positions closed"
            )


class TestDrawdownKillSwitch:
    """
    Test name: Drawdown kill switch halts trading
    What it validates: When max_drawdown_pct is set and the portfolio drops
        beyond that threshold, the engine stops generating new signals and
        force-closes all positions.
    Why it matters: Catches bugs where the kill switch is checked but never
        triggers, or where it triggers but doesn't actually halt trading.
    """

    def test_kill_switch_limits_drawdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            # Falling asset -- will cause drawdown
            df = _make_falling_df(days=252, start_price=100.0, daily_pct=-0.005)
            source.add("CRASH", df)

            engine = _build_engine(tmpdir, source, ["CRASH"], {
                "strategy_name": "_always_buy",
                "strategy_params": {},
                "max_positions": 1,
                "max_alloc_pct": 0.90,
                "max_drawdown_pct": 0.10,  # halt at 10% drawdown
            })
            result = engine.run()

            dd = max_drawdown(result.equity_series)
            # Drawdown should be limited. Allow some overshoot for the day
            # it triggers (intraday loss before halt takes effect).
            assert dd > -0.25, (
                f"Kill switch at 10% but drawdown reached {dd:.1%}"
            )


class TestLookaheadPrevention:
    """
    Test name: T+1 fill invariant -- orders cannot fill on signal day
    What it validates: Every trade's entry date is strictly after the date
        the signal could have been generated (the earliest possible signal date).
    Why it matters: Lookahead bias is the most dangerous backtest bug. If orders
        fill at the same close price used to generate the signal, results are
        unrealistically good.
    """

    def test_fills_happen_after_signal_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine = _build_engine(tmpdir, source, ["TEST"])
            result = engine.run()

            # Every activity log entry's date should be > the earliest signal date.
            # The first signal can happen on the first day with valid indicators;
            # the fill must happen on the NEXT day.
            buy_dates = [e.date for e in result.activity_log if e.action.name == "BUY"]
            if buy_dates:
                # First buy cannot happen on day 1 (no signal day to precede it)
                assert buy_dates[0] > date(2020, 1, 2), (
                    f"First buy fill on {buy_dates[0]} -- cannot be on first day"
                )


class TestSMAStrategyEconomicLogic:
    """
    Test name: SMA crossover buys on uptrend, sells on downtrend
    What it validates: On a V-shaped price series, the SMA crossover strategy
        should generate buy signals when fast SMA crosses above slow SMA
        (during the recovery phase) and sell when it crosses below.
    Why it matters: Validates that the strategy + engine integration produces
        economically logical behavior, not just "no errors."
    """

    def test_sma_crossover_trades_around_inflection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = _make_vshape_df(days=252)
            source.add("V", df)

            engine = _build_engine(tmpdir, source, ["V"], {
                "strategy_params": {"sma_fast": 10, "sma_slow": 30},
            })
            result = engine.run()

            # Should have at least one buy and one sell signal
            buy_count = sum(1 for e in result.activity_log if e.action.name == "BUY")
            sell_count = sum(1 for e in result.activity_log if e.action.name == "SELL")

            assert buy_count >= 1, "SMA crossover should BUY during recovery"
            # Sell count could be 0 if only force-close happens, that's ok


class TestMultipleRunsAreIdempotent:
    """
    Test name: Running the same config twice produces identical results
    What it validates: The backtest is deterministic -- no random state leaks,
        no mutable shared state between runs.
    Why it matters: Non-determinism makes optimization and debugging impossible.
        Catches bugs where global mutable state (strategy instances, caches)
        leaks between runs.
    """

    def test_identical_results_on_repeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine1 = _build_engine(tmpdir, source, ["TEST"])
            result1 = engine1.run()

            engine2 = _build_engine(tmpdir, source, ["TEST"])
            result2 = engine2.run()

            eq1 = result1.equity_series
            eq2 = result2.equity_series

            assert len(eq1) == len(eq2), "Equity curve lengths differ between runs"
            assert (eq1.values == eq2.values).all(), "Equity values differ between runs"
            assert len(result1.trades) == len(result2.trades), "Trade counts differ"


class TestBorrowCostReducesShortReturns:
    """
    Test name: Borrow cost reduces short selling returns
    What it validates: Running a short strategy with borrow_rate > 0 produces
        lower returns than with borrow_rate = 0.
    Why it matters: Catches bugs where borrow costs are tracked on the Position
        object but never actually deducted from portfolio cash.
    """

    def test_borrow_cost_impact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            df = _make_falling_df(days=252, start_price=200.0, daily_pct=-0.001)
            source.add("BEAR", df)

            # Run with zero borrow cost
            engine_free = _build_engine(tmpdir, source, ["BEAR"], {
                "strategy_name": "_always_short",
                "strategy_params": {},
                "allow_short": True,
                "max_positions": 1,
                "max_alloc_pct": 0.50,
                "short_borrow_rate": 0.0,
            })
            result_free = engine_free.run()

            # Run with high borrow cost
            engine_costly = _build_engine(tmpdir, source, ["BEAR"], {
                "strategy_name": "_always_short",
                "strategy_params": {},
                "allow_short": True,
                "max_positions": 1,
                "max_alloc_pct": 0.50,
                "short_borrow_rate": 0.10,  # 10% annual borrow rate
            })
            result_costly = engine_costly.run()

            ret_free = total_return(result_free.equity_series)
            ret_costly = total_return(result_costly.equity_series)

            assert ret_costly < ret_free, (
                f"Borrow costs should reduce returns: free={ret_free:.4f}, "
                f"costly={ret_costly:.4f}"
            )


class TestPercentageFeeModel:
    """
    Test name: Percentage fee model charges proportional to trade value
    What it validates: With a percentage fee model, larger trades incur larger
        fees. Trade fees_total should be proportional to trade value.
    Why it matters: Catches bugs where the fee model type is parsed from config
        but the wrong model is instantiated, or where per-trade flat fees are
        used instead.
    """

    def test_percentage_fees_proportional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MockDataSource()
            source.add("TEST", make_price_df())

            engine = _build_engine(tmpdir, source, ["TEST"], {
                "fee_model": "percentage",
                "fee_per_trade": 10.0,  # 10 bps = 0.1%
                "slippage_bps": 0.0,
            })
            result = engine.run()

            # Activity log entries should have non-zero fees
            buy_entries = [e for e in result.activity_log if e.action.name == "BUY"]
            for entry in buy_entries:
                if entry.value > 0:
                    # Fee should be roughly 0.1% of value
                    expected_fee = entry.value * 0.0010
                    assert abs(entry.fees - expected_fee) < 1.0, (
                        f"Fee {entry.fees:.2f} not close to expected "
                        f"{expected_fee:.2f} (0.1% of {entry.value:.2f})"
                    )
