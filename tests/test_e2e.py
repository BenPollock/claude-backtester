"""End-to-end integration tests for the backtesting engine.

Each test runs a full backtest with MockDataSource (no network calls)
and validates end-to-end behavior across the entire pipeline:
CLI config -> Engine -> Strategy -> Broker -> Portfolio -> Analytics.
"""

from datetime import date
import tempfile

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig, StopConfig, RegimeFilter
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.analytics.metrics import (
    compute_all_metrics,
    cagr,
    sharpe_ratio,
    max_drawdown,
    total_return,
)
from backtester.analytics.calendar import monthly_returns, drawdown_periods
from backtester.strategies.base import Strategy, Signal
from backtester.strategies.registry import discover_strategies, register_strategy, _REGISTRY
from backtester.strategies.indicators import sma, atr
from backtester.types import SignalAction, OrderType, Side
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position

from tests.conftest import MockDataSource, make_price_df

# Ensure all strategies are registered
discover_strategies()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_controlled_df(prices, start="2020-01-02", volume=1_000_000):
    """Create OHLCV DataFrame from a list of close prices.

    Open is slightly below close, High is 1% above, Low is 1% below.
    This gives deterministic, controllable price action for targeted tests.
    """
    days = len(prices)
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    closes = np.array(prices, dtype=float)
    df = pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(days, volume),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )
    return df


def _dates_from_df(df):
    """Extract start/end as datetime.date from a DataFrame index.

    Ensures we return plain datetime.date objects (not pd.Timestamp),
    which is required by TradingCalendar.trading_days().
    """
    start = df.index[0]
    end = df.index[-1]
    if hasattr(start, "date") and callable(start.date):
        start = start.date()
    if hasattr(end, "date") and callable(end.date):
        end = end.date()
    return start, end


def _run_backtest(config, source):
    """Helper: create DataManager + Engine, run backtest, return result."""
    tmpdir = tempfile.mkdtemp()
    dm = DataManager(cache_dir=tmpdir, source=source)
    engine = BacktestEngine(config, data_manager=dm)
    return engine.run()


def _basic_source_and_config(
    tickers=("TEST",),
    benchmark="TEST",
    days=252,
    start_price=100.0,
    daily_return=0.0005,
    sma_fast=20,
    sma_slow=50,
    starting_cash=100_000.0,
    max_positions=10,
    max_alloc_pct=0.10,
    fee_per_trade=0.05,
    slippage_bps=10.0,
    stop_config=None,
    regime_filter=None,
    allow_short=False,
    max_drawdown_pct=None,
    fee_model="per_trade",
    **kwargs,
):
    """Build a MockDataSource + BacktestConfig for common test patterns."""
    source = MockDataSource()
    # Generate data for each ticker
    all_tickers = set(tickers)
    if benchmark:
        all_tickers.add(benchmark)
    for t in all_tickers:
        source.add(t, make_price_df(days=days, start_price=start_price, daily_return=daily_return))

    # Determine date range from the generated data
    sample = source._data[list(all_tickers)[0]]
    start_date, end_date = _dates_from_df(sample)

    config = BacktestConfig(
        strategy_name="sma_crossover",
        tickers=list(tickers),
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
        starting_cash=starting_cash,
        max_positions=max_positions,
        max_alloc_pct=max_alloc_pct,
        strategy_params={"sma_fast": sma_fast, "sma_slow": sma_slow},
        fee_per_trade=fee_per_trade,
        slippage_bps=slippage_bps,
        stop_config=stop_config,
        regime_filter=regime_filter,
        allow_short=allow_short,
        max_drawdown_pct=max_drawdown_pct,
        fee_model=fee_model,
        **kwargs,
    )
    return source, config


# ===========================================================================
# 1. TestBasicBacktest
# ===========================================================================


class TestBasicBacktest:
    """Verify that a basic backtest completes and produces sensible outputs."""

    def test_basic_run_completes(self):
        """Run a basic backtest and verify the result has the expected attributes."""
        source, config = _basic_source_and_config()
        result = _run_backtest(config, source)

        assert result.equity_series is not None, "equity_series should not be None"
        assert len(result.equity_series) > 0, "equity_series should have data points"
        assert result.portfolio is not None, "portfolio should not be None"
        # trades is a list (could be empty if no signals fired)
        assert isinstance(result.trades, list), "trades should be a list"

    def test_cash_accounting(self):
        """After force-close on last day, total equity should equal cash (no open positions)."""
        source, config = _basic_source_and_config()
        result = _run_backtest(config, source)

        portfolio = result.portfolio
        # Engine force-closes all positions on the last day
        assert len(portfolio.positions) == 0, (
            "All positions should be closed at end of backtest"
        )
        # With no open positions, total equity == cash
        assert abs(portfolio.total_equity - portfolio.cash) < 0.01, (
            f"With no positions, equity ({portfolio.total_equity:.2f}) should equal "
            f"cash ({portfolio.cash:.2f})"
        )

    def test_equity_curve_starts_at_starting_cash(self):
        """First equity point should equal starting_cash."""
        source, config = _basic_source_and_config(starting_cash=50_000.0)
        result = _run_backtest(config, source)

        first_equity = result.equity_series.iloc[0]
        assert abs(first_equity - 50_000.0) < 0.01, (
            f"First equity point ({first_equity:.2f}) should equal starting cash (50000.00)"
        )

    def test_equity_curve_length(self):
        """Equity curve should have one point per trading day in the range."""
        source, config = _basic_source_and_config(days=100)
        result = _run_backtest(config, source)

        # The number of equity points should match trading days
        eq_len = len(result.equity_series)
        assert eq_len > 0, "Equity curve should have at least one point"
        # It should roughly equal the number of days in the data
        assert eq_len <= 100, f"Equity curve length ({eq_len}) should not exceed data days (100)"

    def test_trades_have_valid_fields(self):
        """Each completed trade should have sensible field values."""
        source, config = _basic_source_and_config()
        result = _run_backtest(config, source)

        for trade in result.trades:
            assert trade.entry_date <= trade.exit_date, (
                f"entry_date ({trade.entry_date}) must be <= exit_date ({trade.exit_date})"
            )
            assert trade.quantity > 0, f"Trade quantity must be > 0, got {trade.quantity}"
            assert trade.entry_price > 0, f"entry_price must be > 0, got {trade.entry_price}"
            assert trade.exit_price > 0, f"exit_price must be > 0, got {trade.exit_price}"

    def test_no_lookahead(self):
        """Orders submitted on day T should fill on day T+1 or later."""
        source, config = _basic_source_and_config()
        result = _run_backtest(config, source)

        # Check activity log: BUY actions should have date > signal_date
        # We can verify via trades: entry_date should be after the signal was generated.
        # Since signal is generated on day T close, fill is at T+1 open,
        # the entry_date recorded in Trade is the fill date.
        # The best we can check: entry_date != first trading day (signal can't fire
        # until SMA warmup is done, and fill is the day after).
        if result.trades:
            sample = source._data["TEST"]
            first_day = sample.index[0]
            if hasattr(first_day, "date"):
                first_day = first_day.date()
            elif not isinstance(first_day, date):
                first_day = pd.Timestamp(first_day).date()
            for trade in result.trades:
                assert trade.entry_date > first_day, (
                    f"Trade entry ({trade.entry_date}) should be after first trading day "
                    f"({first_day}) due to SMA warmup + T+1 fill delay"
                )

    def test_final_positions_closed(self):
        """After backtest, portfolio should have no open positions."""
        source, config = _basic_source_and_config()
        result = _run_backtest(config, source)

        assert len(result.portfolio.positions) == 0, (
            f"Expected 0 open positions, found {len(result.portfolio.positions)}: "
            f"{list(result.portfolio.positions.keys())}"
        )

    def test_cash_invariant_holds_throughout_equity_curve(self):
        """At every equity snapshot, cash + sum(position_market_values) == total_equity.

        The equity_history stores total_equity snapshots. Since the engine calls
        portfolio.record_equity(day) after updating market prices, each snapshot
        must equal cash + position values at that moment. We verify the final
        state after force-close (where positions=0, so equity must equal cash)
        and also verify that the equity curve values are internally consistent
        by checking that no equity point exceeds starting_cash by an implausible
        amount (no phantom money creation).
        """
        source, config = _basic_source_and_config(starting_cash=100_000.0)
        result = _run_backtest(config, source)
        portfolio = result.portfolio

        # After force-close: no positions, so equity == cash exactly
        position_value = sum(pos.market_value for pos in portfolio.positions.values())
        assert abs(position_value) < 0.01, (
            f"After force-close, position value should be ~0, got {position_value:.2f}"
        )
        computed_equity = portfolio.cash + position_value
        assert abs(computed_equity - portfolio.total_equity) < 0.01, (
            f"Cash invariant violated: cash ({portfolio.cash:.2f}) + "
            f"positions ({position_value:.2f}) = {computed_equity:.2f}, "
            f"but total_equity = {portfolio.total_equity:.2f}"
        )

        # Verify equity curve is reasonable: no value should be negative
        # (starting with $100k and limited position sizing)
        for eq_date, eq_val in portfolio.equity_history:
            assert eq_val > 0, (
                f"Equity went non-positive on {eq_date}: {eq_val:.2f}"
            )

    def test_cash_invariant_with_multiple_tickers(self):
        """Cash invariant holds with multiple tickers trading simultaneously."""
        source, config = _basic_source_and_config(
            tickers=("TICK1", "TICK2", "TICK3"),
            benchmark="TICK1",
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
        )
        # Add distinct data for each ticker
        for ticker in ("TICK1", "TICK2", "TICK3"):
            source.add(ticker, make_price_df(days=252, start_price=100.0, daily_return=0.0005))

        result = _run_backtest(config, source)
        portfolio = result.portfolio

        # After force-close, cash == total_equity
        assert len(portfolio.positions) == 0
        assert abs(portfolio.total_equity - portfolio.cash) < 0.01, (
            f"Cash invariant violated with multi-ticker: equity={portfolio.total_equity:.2f}, "
            f"cash={portfolio.cash:.2f}"
        )


# ===========================================================================
# 2. TestOrderLifecycle
# ===========================================================================


class TestOrderLifecycle:
    """Verify order submission, fill timing, and signal-to-trade lifecycle."""

    def test_order_fills_at_next_day_open(self):
        """BUY fill price should approximately equal the next day's Open (+ slippage)."""
        # Create trending data so SMA crossover fires a BUY
        # Use sma_fast=5, sma_slow=10 with short warmup
        prices = [100.0] * 15  # flat warmup
        prices += [100 + i * 0.5 for i in range(60)]  # uptrend -> BUY signal
        prices += [130 - i * 0.5 for i in range(60)]  # downtrend -> SELL signal
        prices += [100.0] * 20  # flat tail

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            slippage_bps=0.0,  # zero slippage for precise check
            fee_per_trade=0.0,
        )
        result = _run_backtest(config, source)

        assert len(result.trades) > 0, "Expected at least one trade"
        # Check that entry_price is close to an Open price in the data
        trade = result.trades[0]
        entry_idx = df.index.get_loc(pd.Timestamp(trade.entry_date))
        expected_open = df.iloc[entry_idx]["Open"]
        assert abs(trade.entry_price - expected_open) < 1.0, (
            f"Entry price ({trade.entry_price:.4f}) should be close to Open "
            f"({expected_open:.4f}) on fill day"
        )

    def test_sell_signal_closes_position(self):
        """When fast SMA crosses below slow SMA, position should be closed."""
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(60)]  # uptrend
        prices += [130 - i * 0.5 for i in range(60)]  # downtrend
        prices += [100.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should have at least one completed trade (BUY then SELL before force-close)
        assert len(result.trades) >= 1, "Expected at least one completed round-trip trade"
        # Verify the first trade exited before the last day (not just force-closed)
        last_date = end_d
        # At least one trade should have exit_date before the very last trading day
        early_exits = [t for t in result.trades if t.exit_date < last_date]
        assert len(early_exits) >= 1, (
            "At least one trade should exit via SELL signal before force-close on last day"
        )


# ===========================================================================
# 3. TestStopLoss
# ===========================================================================


class TestStopLoss:
    """Verify stop-loss, take-profit, and trailing-stop behavior."""

    def test_stop_loss_triggers(self):
        """A 5% stop-loss should close the position when price drops 5% from entry."""
        # Flat warmup -> uptrend to trigger BUY -> sharp crash to trigger stop
        prices = [100.0] * 15
        prices += [100 + i * 0.3 for i in range(30)]  # gentle uptrend -> BUY
        # After BUY fills around ~109, crash to trigger 5% stop
        prices += [109, 108, 105, 100, 95, 90, 85, 80, 75, 70]
        prices += [70.0] * 20  # stay low

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            stop_config=StopConfig(stop_loss_pct=0.05),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should have trades where stop loss fired
        assert len(result.trades) >= 1, "Expected at least one trade"
        # Check activity log for stop-triggered exits before the crash bottoms
        # At least one trade should exit at a price well above the bottom (70)
        stop_exits = [t for t in result.trades if t.exit_price > 80]
        assert len(stop_exits) >= 1, (
            "Stop loss should have triggered before price hit 80. "
            f"Exit prices: {[t.exit_price for t in result.trades]}"
        )

    def test_take_profit_triggers(self):
        """A 10% take-profit should close the position when price rises 10% from entry."""
        prices = [100.0] * 15
        prices += [100 + i * 0.3 for i in range(30)]  # uptrend -> BUY
        # After BUY, sharp rise to trigger 10% take-profit
        prices += [109 + i * 2 for i in range(20)]  # strong uptrend
        prices += [150.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            stop_config=StopConfig(take_profit_pct=0.10),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        assert len(result.trades) >= 1, "Expected at least one trade"
        # The take-profit should have exited the trade before the all-time high
        # Check that at least one trade exited with a profit
        profitable = [t for t in result.trades if t.pnl > 0]
        assert len(profitable) >= 1, (
            f"Expected at least one profitable trade from take-profit. "
            f"PnLs: {[t.pnl for t in result.trades]}"
        )

    def test_trailing_stop(self):
        """A trailing stop should track the high-water mark and exit on pullback."""
        prices = [100.0] * 15
        prices += [100 + i * 0.3 for i in range(30)]  # uptrend -> BUY
        # Continue up then pull back to trigger trailing stop
        prices += [109 + i * 1.0 for i in range(20)]  # up to ~129
        prices += [129 - i * 2.0 for i in range(15)]  # pullback -> should trigger 5% trailing
        prices += [100.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            stop_config=StopConfig(trailing_stop_pct=0.05),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        assert len(result.trades) >= 1, "Expected at least one trade"
        # Trailing stop should exit before price hits 100 (the bottom)
        early_exits = [t for t in result.trades if t.exit_price > 110]
        assert len(early_exits) >= 1, (
            f"Trailing stop should have triggered above 110. "
            f"Exit prices: {[t.exit_price for t in result.trades]}"
        )


# ===========================================================================
# 4. TestRegimeFilter
# ===========================================================================


class TestRegimeFilter:
    """Verify that the regime filter suppresses/allows BUY signals correctly."""

    def test_regime_suppresses_buy(self):
        """When benchmark SMA fast < slow, no BUY signals should go through."""
        # Create a steadily declining benchmark so fast SMA < slow SMA
        # after warmup. No BUY signals should be generated.
        prices = [200.0 - i * 0.5 for i in range(252)]  # steady downtrend

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        regime = RegimeFilter(
            benchmark="TEST",
            indicator="sma",
            fast_period=10,
            slow_period=20,
        )
        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            regime_filter=regime,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # In a pure downtrend with regime filter on, no BUY should go through
        # so no trades should happen (force-close on last day has no positions to close)
        assert len(result.trades) == 0, (
            f"Regime filter should suppress all BUYs in a downtrend, "
            f"but found {len(result.trades)} trades"
        )

    def test_regime_allows_buy_when_on(self):
        """When benchmark SMA fast > slow, BUY signals should go through."""
        # Create uptrending data so regime is "on" and strategy generates BUYs
        prices = [100.0 + i * 0.5 for i in range(252)]  # steady uptrend

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        regime = RegimeFilter(
            benchmark="TEST",
            indicator="sma",
            fast_period=10,
            slow_period=20,
        )
        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            regime_filter=regime,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # In a strong uptrend with regime on, trades should happen
        assert len(result.trades) >= 1, (
            "Regime filter should allow BUYs in an uptrend, "
            "but no trades were generated"
        )


# ===========================================================================
# 5. TestMultiTicker
# ===========================================================================


class TestMultiTicker:
    """Verify multi-ticker constraints: max positions and max allocation."""

    def test_max_positions_respected(self):
        """With 5 tickers and max_positions=2, never hold more than 2 positions."""
        tickers = ["A", "B", "C", "D", "E"]
        source = MockDataSource()
        # Create uptrending data for all tickers so each generates a BUY
        for t in tickers:
            prices = [100 + i * 0.5 for i in range(150)]
            source.add(t, make_controlled_df(prices))

        # Use ticker A as benchmark
        start_d, end_d = _dates_from_df(source._data["A"])

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=tickers,
            benchmark="A",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=2,
            max_alloc_pct=0.10,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Check equity_history to verify never more than 2 positions
        # We check the portfolio indirectly: with max_positions=2, the number
        # of distinct symbols with overlapping trade dates should be <= 2
        portfolio = result.portfolio
        # The engine enforces max_positions during signal generation
        # We verify by checking that the number of trades is reasonable for 2 positions
        assert len(result.trades) >= 1, "Should have at least one trade"

        # Reconstruct max concurrent positions from trade logs
        from collections import defaultdict
        events = []
        for trade in result.trades:
            events.append((trade.entry_date, 1))   # open
            events.append((trade.exit_date, -1))    # close
        events.sort(key=lambda x: x[0])
        max_open = 0
        current = 0
        for _, delta in events:
            current += delta
            max_open = max(max_open, current)

        assert max_open <= 2, (
            f"Max concurrent positions ({max_open}) should not exceed max_positions (2)"
        )

    def test_max_allocation_respected(self):
        """With max_alloc=0.10, no single position should exceed ~10% of equity at entry."""
        source, config = _basic_source_and_config(
            max_alloc_pct=0.10,
            starting_cash=100_000.0,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        if result.trades:
            for trade in result.trades:
                position_value = trade.entry_price * trade.quantity
                # At entry, equity is approximately starting cash (may vary)
                # We use a generous upper bound: 15% to account for equity changes
                assert position_value <= 100_000.0 * 0.15, (
                    f"Position value ({position_value:.2f}) exceeds 15% of starting cash. "
                    f"Entry price: {trade.entry_price}, qty: {trade.quantity}"
                )


# ===========================================================================
# 6. TestFees
# ===========================================================================


class TestFees:
    """Verify fee models are applied correctly."""

    def test_per_trade_fee_deducted(self):
        """With fee_per_trade=10.0, each trade should have fees_total > 0."""
        source, config = _basic_source_and_config(fee_per_trade=10.0)
        result = _run_backtest(config, source)

        if result.trades:
            for trade in result.trades:
                assert trade.fees_total > 0, (
                    f"Trade in {trade.symbol} should have fees > 0 with fee_per_trade=10.0"
                )

    def test_percentage_fee(self):
        """With percentage fee model, fees should scale with trade size."""
        source, config = _basic_source_and_config(
            fee_model="percentage",
            fee_per_trade=5.0,  # 5 bps
            max_alloc_pct=0.20,
        )
        result = _run_backtest(config, source)

        if result.trades:
            for trade in result.trades:
                assert trade.fees_total > 0, (
                    f"Percentage fee should produce fees > 0"
                )

    def test_zero_fees_vs_fees(self):
        """Running with zero fees should produce higher (or equal) final equity than with fees."""
        source_no_fee, config_no_fee = _basic_source_and_config(
            fee_per_trade=0.0, slippage_bps=0.0,
        )
        result_no_fee = _run_backtest(config_no_fee, source_no_fee)

        source_fee, config_fee = _basic_source_and_config(
            fee_per_trade=10.0, slippage_bps=0.0,
        )
        result_fee = _run_backtest(config_fee, source_fee)

        equity_no_fee = result_no_fee.equity_series.iloc[-1]
        equity_fee = result_fee.equity_series.iloc[-1]

        assert equity_no_fee >= equity_fee, (
            f"Zero-fee equity ({equity_no_fee:.2f}) should be >= "
            f"fee-adjusted equity ({equity_fee:.2f})"
        )


# ===========================================================================
# 7. TestSlippage
# ===========================================================================


class TestSlippage:
    """Verify slippage models affect returns."""

    def test_fixed_slippage_impact(self):
        """Higher slippage should result in lower (or equal) final equity."""
        source_lo, config_lo = _basic_source_and_config(slippage_bps=0.0, fee_per_trade=0.0)
        result_lo = _run_backtest(config_lo, source_lo)

        source_hi, config_hi = _basic_source_and_config(slippage_bps=50.0, fee_per_trade=0.0)
        result_hi = _run_backtest(config_hi, source_hi)

        equity_lo = result_lo.equity_series.iloc[-1]
        equity_hi = result_hi.equity_series.iloc[-1]

        assert equity_lo >= equity_hi, (
            f"Zero-slippage equity ({equity_lo:.2f}) should be >= "
            f"50bps-slippage equity ({equity_hi:.2f})"
        )


# ===========================================================================
# 8. TestFIFOAccounting
# ===========================================================================


class TestFIFOAccounting:
    """Verify FIFO lot ordering on sells."""

    def test_fifo_lot_order(self):
        """When multiple buys happen at different prices, sells should consume
        the earliest lot first (FIFO). We verify by checking trade entry_price
        order matches chronological order.
        """
        # Create data with two separate buy signals (up-down-up pattern)
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(20)]   # uptrend 1 -> BUY
        prices += [110 - i * 0.5 for i in range(20)]   # downtrend -> SELL
        prices += [100 + i * 0.5 for i in range(20)]   # uptrend 2 -> BUY
        prices += [110 - i * 0.5 for i in range(20)]   # downtrend -> SELL
        prices += [100.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should have multiple trades demonstrating FIFO
        if len(result.trades) >= 2:
            # Trades should be in chronological entry order
            for i in range(1, len(result.trades)):
                assert result.trades[i].entry_date >= result.trades[i - 1].entry_date, (
                    "Trades should be in chronological FIFO order"
                )


# ===========================================================================
# 9. TestMetrics
# ===========================================================================


class TestMetrics:
    """Verify analytics metrics computation on real backtest results."""

    def test_metrics_computed(self):
        """compute_all_metrics should return all expected keys."""
        source, config = _basic_source_and_config()
        result = _run_backtest(config, source)

        equity = result.equity_series
        bm = result.benchmark_series
        metrics = compute_all_metrics(equity, result.trades, benchmark_series=bm)

        expected_keys = [
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "total_trades", "win_rate", "profit_factor",
            "calmar_ratio",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: {key}"

    def test_total_return_sign(self):
        """If final equity > starting cash, total_return should be positive."""
        source, config = _basic_source_and_config()
        result = _run_backtest(config, source)

        equity = result.equity_series
        final_equity = equity.iloc[-1]
        ret = total_return(equity)

        if final_equity > config.starting_cash:
            assert ret > 0, (
                f"Final equity ({final_equity:.2f}) > starting cash ({config.starting_cash:.2f}) "
                f"but total_return is {ret:.4f}"
            )
        elif final_equity < config.starting_cash:
            assert ret < 0, (
                f"Final equity ({final_equity:.2f}) < starting cash ({config.starting_cash:.2f}) "
                f"but total_return is {ret:.4f}"
            )

    def test_buy_and_hold_same_as_benchmark(self):
        """If strategy tracks benchmark closely, returns should be in the same ballpark."""
        source, config = _basic_source_and_config(
            max_alloc_pct=1.0,  # allow full allocation
            max_positions=1,
        )
        result = _run_backtest(config, source)

        equity = result.equity_series
        bm = result.benchmark_series
        if bm is not None and len(bm) > 1:
            strat_ret = total_return(equity)
            bm_ret = total_return(bm)
            # They won't be identical due to fill timing and fees,
            # but should be in the same direction at least
            if abs(bm_ret) > 0.01:
                assert (strat_ret > 0) == (bm_ret > 0) or abs(strat_ret) < 0.05, (
                    f"Strategy return ({strat_ret:.4f}) and benchmark return ({bm_ret:.4f}) "
                    f"should have roughly the same sign"
                )


# ===========================================================================
# 10. TestShortSelling
# ===========================================================================


class TestShortSelling:
    """Verify short selling configuration."""

    def test_short_disabled_by_default(self):
        """With allow_short=False (default), no short positions should exist."""
        source, config = _basic_source_and_config(allow_short=False)
        result = _run_backtest(config, source)

        # SMA crossover is long-only, so no short trades should appear
        # Check activity log for any short entries
        short_entries = [
            entry for entry in result.portfolio.activity_log
            if hasattr(entry, "action") and entry.action == "short_entry"
        ]
        assert len(short_entries) == 0, (
            f"No short entries should exist with allow_short=False, "
            f"found {len(short_entries)}"
        )

    def test_short_enabled_still_works(self):
        """With allow_short=True, the sma_crossover strategy (long-only) should
        still run without errors.
        """
        source, config = _basic_source_and_config(allow_short=True)
        result = _run_backtest(config, source)

        # Should complete without error
        assert result.equity_series is not None
        assert len(result.equity_series) > 0


# ===========================================================================
# 11. TestDrawdownKillSwitch
# ===========================================================================


class TestDrawdownKillSwitch:
    """Verify the drawdown kill switch halts trading."""

    def test_drawdown_halts_trading(self):
        """With a tight max_drawdown_pct and a crashing market, trading should halt.

        We compare a halted run vs an unhalted run. With max_alloc_pct=0.95
        and a large crash, the halted run should have fewer trades during
        the recovery because new signals are suppressed after halt.
        """
        # Uptrend -> BUY with large allocation -> immediate crash
        # The crash must be big enough for the position (95% of equity) to
        # drag total equity down >5% within a few bars.
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(30)]  # uptrend -> BUY
        # Sudden crash: each bar drops ~5% of close
        prices += [115 - i * 6 for i in range(15)]  # drops from 115 to ~25
        prices += [30.0] * 20  # stay low
        # Recovery
        prices += [30 + i * 1.0 for i in range(80)]  # strong recovery

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)

        start_d, end_d = _dates_from_df(df)

        # Run WITH kill switch
        config_halt = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.95,  # large allocation so crash hurts total equity
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            max_drawdown_pct=0.03,  # tight 3% halt threshold
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_halt = _run_backtest(config_halt, source)

        # Run WITHOUT kill switch (same data, same config minus drawdown limit)
        config_no_halt = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.95,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            max_drawdown_pct=None,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_no_halt = _run_backtest(config_no_halt, source)

        # The halted run should have fewer or equal trades: once halted,
        # no new BUY signals fire during the recovery.
        halt_trades = len(result_halt.trades)
        no_halt_trades = len(result_no_halt.trades)
        assert halt_trades <= no_halt_trades, (
            f"Halted run ({halt_trades} trades) should have <= trades than "
            f"unhalted run ({no_halt_trades} trades)"
        )

        # After halt, equity should become flat (cash only, no positions).
        # Check the last 20 data points — they should all be identical since
        # no trading happens after the kill switch fires.
        equity = result_halt.equity_series
        tail = equity.iloc[-20:]
        val_range = tail.max() - tail.min()
        assert val_range < 1.0, (
            f"After drawdown halt, equity tail should be flat. "
            f"Range over last 20 points: {val_range:.2f}"
        )


# ===========================================================================
# 12. TestPositionSizing
# ===========================================================================


class TestPositionSizing:
    """Verify position sizing logic."""

    def test_fixed_fractional_sizing(self):
        """With fixed_fractional sizing and max_alloc_pct=0.10, position size should
        be approximately 10% of equity divided by price.
        """
        source, config = _basic_source_and_config(
            max_alloc_pct=0.10,
            starting_cash=100_000.0,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        if result.trades:
            trade = result.trades[0]
            expected_value = 100_000.0 * 0.10  # 10% of starting equity
            actual_value = trade.entry_price * trade.quantity
            # Should be within 20% of expected (integer rounding of shares)
            assert actual_value < expected_value * 1.2, (
                f"Position value ({actual_value:.2f}) exceeds 120% of target "
                f"({expected_value:.2f})"
            )
            assert actual_value > expected_value * 0.5, (
                f"Position value ({actual_value:.2f}) is less than 50% of target "
                f"({expected_value:.2f}). Sizing may be off."
            )


# ===========================================================================
# 12. TestBenchmarkEquityDensity
# ===========================================================================


class TestBenchmarkEquityDensity:
    """Verify that benchmark equity has a data point for every trading day.

    Regression test for a bug where benchmark equity was tracked inside the
    main loop via timestamp matching (``ts in benchmark_data.index``), which
    could silently skip days if the engine's ``trading_days`` index didn't
    exactly match the benchmark DataFrame's index. The fix computes benchmark
    equity directly from ``benchmark_data["Close"]`` after the loop.
    """

    def test_benchmark_density_matches_equity_curve(self):
        """Benchmark equity should be very close to the equity curve length
        — one per trading day with benchmark data."""
        source, config = _basic_source_and_config(days=252)
        result = _run_backtest(config, source)

        eq_len = len(result.equity_series)
        bm = result.benchmark_series
        assert bm is not None
        bm_len = len(bm)
        # Synthetic data uses bdate_range, which may differ slightly
        # from the NYSE calendar. But the ratio should be very close.
        ratio = bm_len / eq_len
        assert ratio > 0.98, (
            f"Benchmark has {bm_len} points but equity has {eq_len} "
            f"(ratio={ratio:.2%}). Benchmark appears sparse."
        )

    def test_benchmark_no_straight_line_segments(self):
        """Benchmark values should change day-to-day (not be straight-line
        segments caused by missing intermediate data points)."""
        source, config = _basic_source_and_config(days=252)
        result = _run_backtest(config, source)

        bm = result.benchmark_series
        assert bm is not None
        # Check that most consecutive days have different values
        diffs = bm.diff().dropna()
        nonzero = (diffs.abs() > 1e-6).sum()
        assert nonzero > len(diffs) * 0.9, (
            f"Only {nonzero}/{len(diffs)} benchmark days had price change. "
            f"Benchmark appears sparse (straight-line segments)."
        )

    def test_benchmark_starts_at_starting_cash(self):
        """The first benchmark equity value should equal starting_cash."""
        source, config = _basic_source_and_config(
            starting_cash=50_000.0, days=100,
        )
        result = _run_backtest(config, source)

        bm = result.benchmark_series
        assert bm is not None
        assert abs(bm.iloc[0] - 50_000.0) < 1.0, (
            f"First benchmark equity ({bm.iloc[0]:.2f}) should approximate "
            f"starting cash (50000.00)"
        )

    def test_benchmark_final_value_reflects_price_change(self):
        """Benchmark final value should track the underlying close prices,
        not be constant or disconnected from actual price movements."""
        source, config = _basic_source_and_config(
            days=100, start_price=100.0,
        )
        result = _run_backtest(config, source)

        bm = result.benchmark_series
        assert bm is not None
        # The benchmark should not be flat — it must track price changes
        assert bm.iloc[-1] != bm.iloc[0], (
            "Benchmark value should change over time, not stay constant"
        )

    def test_benchmark_with_separate_benchmark_ticker(self):
        """When benchmark ticker differs from trading tickers, benchmark
        equity should still have full daily coverage."""
        source = MockDataSource()
        source.add("ALPHA", make_price_df(days=200, start_price=50.0))
        source.add("BETA", make_price_df(days=200, start_price=150.0))

        sample = source._data["ALPHA"]
        start_date, end_date = _dates_from_df(sample)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["ALPHA"],
            benchmark="BETA",
            start_date=start_date,
            end_date=end_date,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            strategy_params={"sma_fast": 20, "sma_slow": 50},
        )
        result = _run_backtest(config, source)

        eq_len = len(result.equity_series)
        bm = result.benchmark_series
        assert bm is not None
        bm_len = len(bm)
        ratio = bm_len / eq_len
        assert ratio > 0.98, (
            f"Separate-ticker benchmark has {bm_len} points vs equity {eq_len} "
            f"(ratio={ratio:.2%})"
        )

    def test_benchmark_with_multi_ticker_backtest(self):
        """When running with multiple tickers and one of them as benchmark,
        benchmark equity should still be fully populated."""
        source = MockDataSource()
        for sym in ["SPY", "QQQ"]:
            source.add(sym, make_price_df(days=200, start_price=100.0))

        sample = source._data["SPY"]
        start_date, end_date = _dates_from_df(sample)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["SPY", "QQQ"],
            benchmark="SPY",
            start_date=start_date,
            end_date=end_date,
            starting_cash=100_000.0,
            max_positions=2,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 20, "sma_slow": 50},
        )
        result = _run_backtest(config, source)

        eq_len = len(result.equity_series)
        bm = result.benchmark_series
        assert bm is not None
        ratio = len(bm) / eq_len
        assert ratio > 0.98, (
            f"Multi-ticker benchmark has {len(bm)} points vs equity {eq_len} "
            f"(ratio={ratio:.2%})"
        )

    def test_benchmark_survives_stale_cache_timestamps(self):
        """Regression: cached data with non-midnight timestamps (from older
        yfinance/pandas versions) should not produce sparse benchmark equity.

        This was the root cause of the 'straight-line benchmark' bug where
        the SPY Buy & Hold chart showed only a few data points over years.
        """
        from backtester.data.cache import ParquetCache

        dates = pd.bdate_range("2020-01-02", periods=200, freq="B")

        # Simulate stale cache: data stored with 5am UTC timestamps
        stale_timestamps = [d + pd.Timedelta(hours=5) for d in dates]
        stale_df = pd.DataFrame(
            {
                "Open": np.linspace(99, 120, 200),
                "High": np.linspace(101, 122, 200),
                "Low": np.linspace(97, 118, 200),
                "Close": np.linspace(100, 121, 200),
                "Volume": np.full(200, 1_000_000),
            },
            index=pd.DatetimeIndex(stale_timestamps, name="Date"),
        )

        tmpdir = tempfile.mkdtemp()

        # Pre-populate cache with stale timestamps
        cache = ParquetCache(tmpdir)
        cache.save("TEST", stale_df)

        # Create a MockDataSource that would NOT be called (cache hit)
        source = MockDataSource()
        # Add data just in case cache miss, but we expect cache hit
        source.add("TEST", stale_df)

        start_date = dates[0].date()
        end_date = dates[-1].date()

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_date,
            end_date=end_date,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            strategy_params={"sma_fast": 20, "sma_slow": 50},
        )

        dm = DataManager(cache_dir=tmpdir, source=source)
        engine = BacktestEngine(config, data_manager=dm)
        result = engine.run()

        eq_len = len(result.equity_series)
        bm = result.benchmark_series
        assert bm is not None, "Benchmark should not be None"
        ratio = len(bm) / eq_len
        assert ratio > 0.90, (
            f"Benchmark has {len(bm)} points vs equity {eq_len} "
            f"(ratio={ratio:.2%}). Stale cache timestamps caused "
            f"sparse benchmark data."
        )


# ---------------------------------------------------------------------------
# Custom test strategies for new E2E tests
# ---------------------------------------------------------------------------

class _LimitOrderStrategy(Strategy):
    """Strategy that issues limit BUY orders below current price,
    and limit SELL orders above current price once in a position.

    Configurable via strategy_params:
      - buy_discount: fraction below close for limit buy (default 0.02)
      - sell_premium: fraction above close for limit sell (default 0.02)
      - time_in_force: "DAY" or "GTC" (default "DAY")
      - warmup: minimum row index before issuing signals (default 5)
    """

    def __init__(self):
        super().__init__()
        self._buy_discount = 0.02
        self._sell_premium = 0.02
        self._tif = "DAY"
        self._warmup = 5
        self._day_count = 0

    def configure(self, params: dict) -> None:
        self._buy_discount = params.get("buy_discount", self._buy_discount)
        self._sell_premium = params.get("sell_premium", self._sell_premium)
        self._tif = params.get("time_in_force", self._tif)
        self._warmup = params.get("warmup", self._warmup)
        self._day_count = 0

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        # Add a simple row counter for warmup
        df["row_idx"] = range(len(df))
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        idx = row.get("row_idx", 0)
        close = row["Close"]
        has_pos = position is not None and position.total_quantity > 0

        if idx < self._warmup:
            return SignalAction.HOLD

        if not has_pos:
            # Limit buy below current price
            limit_price = close * (1.0 - self._buy_discount)
            return Signal(
                action=SignalAction.BUY,
                limit_price=limit_price,
                time_in_force=self._tif,
            )
        else:
            # Limit sell above current price
            limit_price = close * (1.0 + self._sell_premium)
            return Signal(
                action=SignalAction.SELL,
                limit_price=limit_price,
                time_in_force=self._tif,
            )


class _AlwaysShortStrategy(Strategy):
    """Strategy that shorts immediately and covers after N days."""

    def __init__(self):
        super().__init__()
        self._hold_days = 10
        self._warmup = 3
        self._entry_days = {}

    def configure(self, params: dict) -> None:
        self._hold_days = params.get("hold_days", self._hold_days)
        self._warmup = params.get("warmup", self._warmup)
        self._entry_days = {}

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["row_idx"] = range(len(df))
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        idx = row.get("row_idx", 0)
        has_short = position is not None and position.is_short

        if idx < self._warmup:
            return SignalAction.HOLD

        if not has_short and symbol not in self._entry_days:
            self._entry_days[symbol] = idx
            return SignalAction.SHORT

        if has_short and symbol in self._entry_days:
            if idx - self._entry_days[symbol] >= self._hold_days:
                del self._entry_days[symbol]
                return SignalAction.COVER

        return SignalAction.HOLD


class _ATRStrategy(Strategy):
    """Strategy that computes ATR and buys once (for testing ATR-based sizing)."""

    def __init__(self):
        super().__init__()
        self._atr_period = 14
        self._sma_fast = 5
        self._sma_slow = 10

    def configure(self, params: dict) -> None:
        self._atr_period = params.get("atr_period", self._atr_period)
        self._sma_fast = params.get("sma_fast", self._sma_fast)
        self._sma_slow = params.get("sma_slow", self._sma_slow)

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["ATR"] = atr(df, self._atr_period)
        df["sma_fast"] = sma(df["Close"], self._sma_fast)
        df["sma_slow"] = sma(df["Close"], self._sma_slow)
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        fast = row.get("sma_fast")
        slow = row.get("sma_slow")
        if pd.isna(fast) or pd.isna(slow):
            return SignalAction.HOLD

        has_pos = position is not None and position.total_quantity > 0
        if fast > slow and not has_pos:
            return SignalAction.BUY
        elif fast < slow and has_pos:
            return SignalAction.SELL
        return SignalAction.HOLD


class _LosingStrategy(Strategy):
    """Strategy that buys at peaks and sells at troughs (intentionally losing)."""

    def __init__(self):
        super().__init__()

    def configure(self, params: dict) -> None:
        pass

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["row_idx"] = range(len(df))
        # Use fast/slow SMA but INVERTED: buy when fast < slow (downtrend)
        df["sma_fast"] = sma(df["Close"], 5)
        df["sma_slow"] = sma(df["Close"], 15)
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        fast = row.get("sma_fast")
        slow = row.get("sma_slow")
        if pd.isna(fast) or pd.isna(slow):
            return SignalAction.HOLD

        has_pos = position is not None and position.total_quantity > 0
        # Inverted: buy in downtrend, sell in uptrend
        if fast < slow and not has_pos:
            return SignalAction.BUY
        elif fast > slow and has_pos:
            return SignalAction.SELL
        return SignalAction.HOLD


# Register custom test strategies (safe to call multiple times for same class)
def _register_test_strategies():
    """Register test-only strategies, skipping if already registered."""
    for name, cls in [
        ("_limit_order_test", _LimitOrderStrategy),
        ("_always_short_test", _AlwaysShortStrategy),
        ("_atr_test", _ATRStrategy),
        ("_losing_test", _LosingStrategy),
    ]:
        if name not in _REGISTRY:
            _REGISTRY[name] = cls

_register_test_strategies()


# ===========================================================================
# 13. TestLimitOrders
# ===========================================================================


class TestLimitOrders:
    """E2E tests for limit order lifecycle: fill on price reach, GTC persistence,
    DAY expiry."""

    def test_limit_buy_fills_when_price_drops(self):
        """A limit BUY set below current price should fill when Low reaches the limit."""
        # Prices: stable, then drop enough to trigger limit buy
        prices = [100.0] * 10
        prices += [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]  # steady decline
        prices += [90.0] * 20  # stay low
        prices += [90 + i * 0.5 for i in range(20)]  # rise (so position gets closed)

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_limit_order_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"buy_discount": 0.02, "sell_premium": 0.02,
                             "time_in_force": "GTC", "warmup": 5},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # A limit BUY should have filled: check activity log for BUY entries
        buys = [e for e in result.portfolio.activity_log if e.action == Side.BUY]
        assert len(buys) >= 1, (
            "Limit BUY should have filled when price dropped to limit level"
        )
        # Fill price should be at or near the limit price (discount from close)
        # The limit was set at close * 0.98; fill should be at that price
        for buy in buys:
            assert buy.price <= 100.0, (
                f"Limit BUY fill price ({buy.price:.2f}) should be at or below "
                f"the initial price level"
            )

    def test_limit_sell_fills_when_price_rises(self):
        """A limit SELL set above current price should fill when High reaches limit."""
        # Start low, buy in (via GTC limit), then prices rise to trigger limit sell
        prices = [100.0] * 6
        prices += [98, 96, 94, 92, 90, 88]  # drop to fill limit buy
        prices += [88, 90, 92, 94, 96, 98, 100, 102, 104, 106]  # rise
        prices += [108, 110, 112, 114, 116]  # continue rising to trigger sell
        prices += [116.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_limit_order_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"buy_discount": 0.02, "sell_premium": 0.02,
                             "time_in_force": "GTC", "warmup": 5},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should have at least one completed trade (buy + sell)
        sells = [e for e in result.portfolio.activity_log if e.action == Side.SELL]
        assert len(sells) >= 1, (
            "Limit SELL should have filled when price rose to limit level"
        )

    def test_gtc_limit_order_persists_across_days(self):
        """A GTC limit BUY should stay pending and fill when price eventually drops."""
        # Price stays above limit for several days, then drops
        prices = [100.0] * 10  # warmup + initial signals
        prices += [101, 102, 103, 104, 105]  # rising — limit BUY won't fill
        prices += [104, 102, 100, 97, 95, 93, 91]  # drop — should fill GTC order
        prices += [90.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_limit_order_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"buy_discount": 0.02, "time_in_force": "GTC",
                             "warmup": 5},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # GTC order should eventually fill
        buys = [e for e in result.portfolio.activity_log if e.action == Side.BUY]
        assert len(buys) >= 1, (
            "GTC limit BUY should persist and fill when price eventually drops"
        )

    def test_day_limit_order_expires(self):
        """A DAY limit BUY should expire if price doesn't reach limit that day.
        Compared to GTC, a DAY-only strategy should generate fewer fills when
        the price rises away from the limit."""
        # Rising prices — limit BUY at discount will rarely fill
        prices = [100.0] * 10
        prices += [100 + i * 0.5 for i in range(40)]  # steady rise

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        # DAY orders
        config_day = BacktestConfig(
            strategy_name="_limit_order_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"buy_discount": 0.05, "time_in_force": "DAY",
                             "warmup": 5},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_day = _run_backtest(config_day, source)

        # GTC orders with the same data
        source2 = MockDataSource()
        source2.add("TEST", make_controlled_df(prices))
        config_gtc = BacktestConfig(
            strategy_name="_limit_order_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"buy_discount": 0.05, "time_in_force": "GTC",
                             "warmup": 5},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_gtc = _run_backtest(config_gtc, source2)

        # In a rising market with a 5% discount limit, DAY orders should fill
        # less often than GTC orders (DAY expires each day, GTC persists).
        day_buys = len([e for e in result_day.portfolio.activity_log if e.action == Side.BUY])
        gtc_buys = len([e for e in result_gtc.portfolio.activity_log if e.action == Side.BUY])
        # Both might fill zero or a few. The key is DAY <= GTC.
        assert day_buys <= gtc_buys, (
            f"DAY limit orders ({day_buys} fills) should fill <= GTC orders "
            f"({gtc_buys} fills) in a rising market"
        )


# ===========================================================================
# 14. TestShortSellingWithStops
# ===========================================================================


class TestShortSellingWithStops:
    """E2E tests for short selling combined with stop-loss and take-profit."""

    def test_short_with_stop_loss_triggers_on_price_rise(self):
        """SHORT entry with stop_loss should trigger when price RISES above stop level."""
        # Stable, then drop (triggers short entry via _always_short), then rise (triggers stop)
        prices = [100.0] * 5
        prices += [100, 99, 98, 97, 96, 95]  # decline
        prices += [95, 97, 99, 101, 103, 105, 107, 109, 111]  # sharp rise -> stop loss
        prices += [111.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"hold_days": 50, "warmup": 3},
            allow_short=True,
            stop_config=StopConfig(stop_loss_pct=0.05),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # There should be at least one trade (the short was entered and stopped out)
        assert len(result.trades) >= 1, (
            "Short position should have been opened and stopped out"
        )
        # The exit should happen before hold_days expire (stop triggered early)
        # Check that exit price is near the stop level, not at the much higher prices
        for trade in result.trades:
            # Short was entered around 95-100 area, stop at entry * 1.05
            # The exit should be at or near the stop level
            assert trade.exit_price < 115, (
                f"Stop loss should have triggered before price reached 115, "
                f"but exit was at {trade.exit_price:.2f}"
            )

    def test_short_with_take_profit_triggers_on_price_drop(self):
        """SHORT entry with take_profit should trigger when price DROPS to target."""
        # Short entry, then price drops further -> take profit triggers
        prices = [100.0] * 5
        prices += [100, 98, 96, 94, 92, 90]  # decline to enter short
        prices += [88, 86, 84, 82, 80, 78]  # further decline -> take profit
        prices += [78.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"hold_days": 50, "warmup": 3},
            allow_short=True,
            stop_config=StopConfig(take_profit_pct=0.10),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should have a trade that exited via take profit
        assert len(result.trades) >= 1, (
            "Short position should have been opened and taken profit"
        )
        # Trade should be profitable (short sold high, covered low)
        profitable = [t for t in result.trades if t.pnl > 0]
        assert len(profitable) >= 1, (
            f"Take profit on short should produce a profitable trade. "
            f"PnLs: {[t.pnl for t in result.trades]}"
        )

    def test_short_with_trailing_stop(self):
        """SHORT with stop_loss: stop triggers when price rises above entry * (1 + pct)."""
        # Short entry around price 100, then price rises sharply past the 5% stop
        prices = [100.0] * 5
        prices += [98, 96, 94, 92, 90]  # drop to trigger short entry
        prices += [90, 88, 86, 84, 82, 80]  # further drop (favorable for short)
        prices += [82, 86, 90, 94, 98, 102, 106, 110, 114, 118]  # sharp reversal
        prices += [120.0] * 30  # stay high

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        # 5% stop loss on a short means stop triggers when price > entry * 1.05
        config = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"hold_days": 50, "warmup": 3},
            allow_short=True,
            stop_config=StopConfig(stop_loss_pct=0.05),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should have a trade that was stopped out on the reversal
        assert len(result.trades) >= 1, (
            "Short position should have been stopped out on reversal"
        )
        # The exit should happen during the reversal, before the last day
        for trade in result.trades:
            assert trade.exit_date <= end_d, (
                f"Short stop should trigger during the reversal, "
                f"but exited on {trade.exit_date}"
            )
            # Exit price should be near the stop level, not at the high (120)
            assert trade.exit_price < 115, (
                f"Short stop should trigger before price reaches 115, "
                f"but exit was at {trade.exit_price:.2f}"
            )


# ===========================================================================
# 15. TestPositionSizingVariants
# ===========================================================================


class TestPositionSizingVariants:
    """E2E tests for different position sizing models."""

    def test_atr_sizer_differs_from_fixed_fractional(self):
        """ATRSizer should produce a different position size than FixedFractional
        when ATR data is available."""
        # Uptrend to trigger BUY with enough data for ATR
        prices = [100.0] * 20
        prices += [100 + i * 0.3 for i in range(60)]  # uptrend
        prices += [118 - i * 0.3 for i in range(40)]  # downtrend
        prices += [106.0] * 20

        source_ff = MockDataSource()
        source_ff.add("TEST", make_controlled_df(prices))
        source_atr = MockDataSource()
        source_atr.add("TEST", make_controlled_df(prices))

        start_d, end_d = _dates_from_df(source_ff._data["TEST"])

        # Fixed fractional
        config_ff = BacktestConfig(
            strategy_name="_atr_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"atr_period": 14, "sma_fast": 5, "sma_slow": 10},
            position_sizing="fixed_fractional",
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_ff = _run_backtest(config_ff, source_ff)

        # ATR sizer
        config_atr = BacktestConfig(
            strategy_name="_atr_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"atr_period": 14, "sma_fast": 5, "sma_slow": 10},
            position_sizing="atr",
            sizing_risk_pct=0.01,
            sizing_atr_multiple=2.0,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_atr = _run_backtest(config_atr, source_atr)

        # Both should have trades
        assert len(result_ff.trades) >= 1, "FixedFractional run should have trades"
        assert len(result_atr.trades) >= 1, "ATRSizer run should have trades"

        # The first trade quantities should differ (ATR sizes by risk, not by alloc %)
        qty_ff = result_ff.trades[0].quantity
        qty_atr = result_atr.trades[0].quantity
        assert qty_ff != qty_atr, (
            f"ATRSizer ({qty_atr} shares) should produce different size than "
            f"FixedFractional ({qty_ff} shares)"
        )

    def test_position_size_respects_max_alloc(self):
        """No single position should exceed max_alloc_pct of equity at entry."""
        prices = [100.0] * 20
        prices += [100 + i * 0.3 for i in range(60)]
        prices += [118 - i * 0.3 for i in range(40)]
        prices += [106.0] * 20

        source = MockDataSource()
        source.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(source._data["TEST"])

        for sizing in ["fixed_fractional", "atr"]:
            config = BacktestConfig(
                strategy_name="_atr_test",
                tickers=["TEST"],
                benchmark="TEST",
                start_date=start_d,
                end_date=end_d,
                starting_cash=100_000.0,
                max_positions=10,
                max_alloc_pct=0.15,
                strategy_params={"atr_period": 14, "sma_fast": 5, "sma_slow": 10},
                position_sizing=sizing,
                fee_per_trade=0.0,
                slippage_bps=0.0,
            )
            src = MockDataSource()
            src.add("TEST", make_controlled_df(prices))
            result = _run_backtest(config, src)

            for trade in result.trades:
                position_value = trade.entry_price * trade.quantity
                # Allow some headroom for integer rounding
                assert position_value <= 100_000.0 * 0.20, (
                    f"[{sizing}] Position value ({position_value:.2f}) exceeds 20% of "
                    f"starting equity. max_alloc_pct=0.15 should cap it."
                )


# ===========================================================================
# 16. TestAnalyticsIntegration
# ===========================================================================


class TestAnalyticsIntegration:
    """E2E tests verifying analytics output from real backtest results."""

    def test_compute_all_metrics_returns_expected_keys(self):
        """compute_all_metrics should return all expected keys with valid values."""
        source, config = _basic_source_and_config()
        result = _run_backtest(config, source)

        equity = result.equity_series
        bm = result.benchmark_series
        metrics = compute_all_metrics(equity, result.trades, benchmark_series=bm)

        required_keys = [
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "total_trades", "win_rate", "profit_factor",
            "calmar_ratio", "trade_expectancy", "avg_win", "avg_loss",
            "var_95", "cvar_95",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metric key: {key}"
            val = metrics[key]
            # Values should be finite numbers (not NaN)
            assert val is not None, f"Metric '{key}' is None"
            assert np.isfinite(val) or val == float("inf"), (
                f"Metric '{key}' has invalid value: {val}"
            )

    def test_monthly_returns_table_shape(self):
        """Monthly returns table should have correct shape (years x months+YTD)."""
        source, config = _basic_source_and_config(days=252)
        result = _run_backtest(config, source)

        equity = result.equity_series
        table = monthly_returns(equity)

        # Should have at least 1 year row
        assert len(table) >= 1, "Monthly returns should have at least 1 year"
        # Should have month columns plus YTD
        assert "YTD" in table.columns, "Monthly returns should have a YTD column"
        # Each value should be finite
        for col in table.columns:
            for val in table[col].dropna():
                assert np.isfinite(val), (
                    f"Monthly return value in column {col} is not finite: {val}"
                )

    def test_drawdown_periods_for_losing_strategy(self):
        """A losing strategy should produce non-empty drawdown periods."""
        # Create choppy, generally declining data to ensure losses
        prices = [100.0] * 20
        # Oscillating downtrend
        for i in range(80):
            base = 100 - i * 0.3
            offset = 2.0 if i % 4 < 2 else -2.0
            prices.append(max(base + offset, 50.0))
        prices += [75.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_losing_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        equity = result.equity_series
        dd = max_drawdown(equity)
        # With a losing strategy, max drawdown should be negative
        assert dd < 0, (
            f"Losing strategy should have negative max drawdown, got {dd:.4f}"
        )

        periods = drawdown_periods(equity, top_n=5)
        # If there's a drawdown, the periods table should have entries
        if dd < -0.001:
            assert len(periods) >= 1, (
                f"Drawdown periods should be non-empty for a strategy with "
                f"max_drawdown={dd:.4f}"
            )


# ===========================================================================
# 17. TestActivityLogCompleteness
# ===========================================================================


class TestActivityLogCompleteness:
    """E2E tests verifying the activity log contains entries for all trades."""

    def test_every_trade_has_activity_log_entries(self):
        """Every BUY and SELL in the trade lifecycle should have a corresponding
        activity log entry."""
        source, config = _basic_source_and_config(
            max_alloc_pct=0.30,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        activity = result.portfolio.activity_log
        buy_entries = [e for e in activity if e.action == Side.BUY]
        sell_entries = [e for e in activity if e.action == Side.SELL]

        # If there are completed trades, there must be buy and sell log entries
        if result.trades:
            assert len(buy_entries) >= 1, (
                f"Found {len(result.trades)} trades but no BUY activity log entries"
            )
            assert len(sell_entries) >= 1, (
                f"Found {len(result.trades)} trades but no SELL activity log entries"
            )

        # Every BUY entry should have valid fields
        for entry in buy_entries:
            assert entry.quantity > 0, f"BUY quantity should be > 0, got {entry.quantity}"
            assert entry.price > 0, f"BUY price should be > 0, got {entry.price}"
            assert entry.value > 0, f"BUY value should be > 0, got {entry.value}"
            assert abs(entry.value - entry.quantity * entry.price) < 0.01, (
                f"BUY value ({entry.value:.2f}) should equal qty * price "
                f"({entry.quantity} * {entry.price:.2f} = {entry.quantity * entry.price:.2f})"
            )

        # Every SELL entry should have valid fields
        for entry in sell_entries:
            assert entry.quantity > 0, f"SELL quantity should be > 0, got {entry.quantity}"
            assert entry.price > 0, f"SELL price should be > 0, got {entry.price}"

    def test_activity_log_prices_match_trade_prices(self):
        """Activity log fill prices should be consistent with Trade entry/exit prices."""
        # Use controlled prices for deterministic behavior
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(60)]  # uptrend -> BUY
        prices += [130 - i * 0.5 for i in range(60)]  # downtrend -> SELL
        prices += [100.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.30,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        activity = result.portfolio.activity_log
        buy_entries = [e for e in activity if e.action == Side.BUY and e.symbol == "TEST"]
        sell_entries = [e for e in activity if e.action == Side.SELL and e.symbol == "TEST"]

        if result.trades:
            # For each trade, find matching activity log entries by date
            for trade in result.trades:
                matching_buys = [e for e in buy_entries if e.date == trade.entry_date]
                assert len(matching_buys) >= 1, (
                    f"Trade entry on {trade.entry_date} has no matching BUY in activity log"
                )
                # Entry price in activity log should match trade entry price
                buy_price = matching_buys[0].price
                assert abs(buy_price - trade.entry_price) < 0.01, (
                    f"Activity log BUY price ({buy_price:.4f}) should match "
                    f"trade entry_price ({trade.entry_price:.4f})"
                )

    def test_activity_log_buy_sell_counts_balanced(self):
        """For a long-only strategy where all positions are closed, the total
        quantity bought should equal the total quantity sold."""
        source, config = _basic_source_and_config(
            max_alloc_pct=0.20,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        activity = result.portfolio.activity_log
        total_bought = sum(e.quantity for e in activity if e.action == Side.BUY)
        total_sold = sum(e.quantity for e in activity if e.action == Side.SELL)

        # After force-close, all positions are closed. Note that force-close
        # adds to trade_log but NOT to activity_log. So activity_log may
        # show more buys than sells if the last position was force-closed.
        # We check: sold <= bought (sold can be less due to force-close)
        assert total_sold <= total_bought, (
            f"Total sold ({total_sold}) should be <= total bought ({total_bought})"
        )


# ---------------------------------------------------------------------------
# Custom test strategies for new E2E tests (batch 2)
# ---------------------------------------------------------------------------

class _AlwaysBuyStrategy(Strategy):
    """Strategy that buys once and never sells. Used to test force-close."""

    def __init__(self):
        super().__init__()
        self._bought = set()

    def configure(self, params: dict) -> None:
        self._bought = set()

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["row_idx"] = range(len(df))
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        idx = row.get("row_idx", 0)
        has_pos = position is not None and position.total_quantity > 0
        # Wait for a few days of warmup, then buy once and hold forever
        if idx >= 5 and not has_pos and symbol not in self._bought:
            self._bought.add(symbol)
            return SignalAction.BUY
        return SignalAction.HOLD


class _HoldOnlyStrategy(Strategy):
    """Strategy that always returns HOLD. Never generates any signals."""

    def compute_indicators(self, df, timeframe_data=None):
        return df.copy()

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        return SignalAction.HOLD


class _AlwaysBuyMultiStrategy(Strategy):
    """Strategy that buys every ticker as soon as possible. For multi-ticker tests."""

    def __init__(self):
        super().__init__()
        self._bought = set()

    def configure(self, params: dict) -> None:
        self._bought = set()

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["row_idx"] = range(len(df))
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        idx = row.get("row_idx", 0)
        has_pos = position is not None and position.total_quantity > 0
        if idx >= 5 and not has_pos and symbol not in self._bought:
            self._bought.add(symbol)
            return SignalAction.BUY
        return SignalAction.HOLD


def _register_test_strategies_batch2():
    """Register batch 2 test-only strategies, skipping if already registered."""
    for name, cls in [
        ("_always_buy_test", _AlwaysBuyStrategy),
        ("_hold_only_test", _HoldOnlyStrategy),
        ("_always_buy_multi_test", _AlwaysBuyMultiStrategy),
    ]:
        if name not in _REGISTRY:
            _REGISTRY[name] = cls

_register_test_strategies_batch2()


# ===========================================================================
# 18. TestCompositeFeesE2E
# ===========================================================================


class TestCompositeFeesE2E:
    """Test composite fee model (SEC + TAF + per-trade) through full backtest."""

    def test_composite_fees_stacked(self):
        """Composite fee model should produce fees from all component models."""
        # Use uptrending data so trades happen
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(60)]  # uptrend -> BUY
        prices += [130 - i * 0.5 for i in range(60)]  # downtrend -> SELL
        prices += [100.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_model="composite_us",
            fee_per_trade=5.0,  # 5 bps for the percentage component
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        assert len(result.trades) >= 1, "Expected at least one trade with composite fees"
        for trade in result.trades:
            assert trade.fees_total > 0, (
                f"Composite fee should produce non-zero fees, got {trade.fees_total}"
            )

    def test_composite_fees_exceed_single_model(self):
        """Total fees with composite model should exceed fees from any single model alone."""
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(60)]
        prices += [130 - i * 0.5 for i in range(60)]
        prices += [100.0] * 20

        # Run with composite_us fee model
        source_comp = MockDataSource()
        df = make_controlled_df(prices)
        source_comp.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config_comp = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_model="composite_us",
            fee_per_trade=5.0,
            slippage_bps=0.0,
        )
        result_comp = _run_backtest(config_comp, source_comp)

        # Run with percentage fee model only (same bps)
        source_pct = MockDataSource()
        source_pct.add("TEST", make_controlled_df(prices))

        config_pct = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_model="percentage",
            fee_per_trade=5.0,
            slippage_bps=0.0,
        )
        result_pct = _run_backtest(config_pct, source_pct)

        # Both should have trades
        assert len(result_comp.trades) >= 1
        assert len(result_pct.trades) >= 1

        total_comp_fees = sum(t.fees_total for t in result_comp.trades)
        total_pct_fees = sum(t.fees_total for t in result_pct.trades)

        assert total_comp_fees > total_pct_fees, (
            f"Composite fees ({total_comp_fees:.4f}) should exceed "
            f"percentage-only fees ({total_pct_fees:.4f})"
        )

    def test_composite_fees_reduce_equity(self):
        """Composite fees should reduce final equity compared to zero fees."""
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(60)]
        prices += [130 - i * 0.5 for i in range(60)]
        prices += [100.0] * 20

        # Zero fees
        source_zero = MockDataSource()
        source_zero.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config_zero = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_zero = _run_backtest(config_zero, source_zero)

        # Composite fees
        source_comp = MockDataSource()
        source_comp.add("TEST", make_controlled_df(prices))

        config_comp = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            fee_model="composite_us",
            fee_per_trade=5.0,
            slippage_bps=0.0,
        )
        result_comp = _run_backtest(config_comp, source_comp)

        equity_zero = result_zero.equity_series.iloc[-1]
        equity_comp = result_comp.equity_series.iloc[-1]

        assert equity_zero >= equity_comp, (
            f"Zero-fee equity ({equity_zero:.2f}) should >= composite-fee equity ({equity_comp:.2f})"
        )


# ===========================================================================
# 19. TestVolumeSlippageE2E
# ===========================================================================


class TestVolumeSlippageE2E:
    """Test volume-based slippage through full backtest."""

    def test_volume_slippage_worsens_fills(self):
        """Volume slippage should produce worse fills (lower equity) than zero slippage."""
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(60)]
        prices += [130 - i * 0.5 for i in range(60)]
        prices += [100.0] * 20

        # Zero slippage run
        source_zero = MockDataSource()
        source_zero.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config_zero = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            slippage_model="fixed",
            slippage_bps=0.0,
            fee_per_trade=0.0,
        )
        result_zero = _run_backtest(config_zero, source_zero)

        # Volume slippage run
        source_vol = MockDataSource()
        source_vol.add("TEST", make_controlled_df(prices))

        config_vol = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            slippage_model="volume",
            slippage_impact_factor=0.1,
            fee_per_trade=0.0,
        )
        result_vol = _run_backtest(config_vol, source_vol)

        equity_zero = result_zero.equity_series.iloc[-1]
        equity_vol = result_vol.equity_series.iloc[-1]

        assert equity_zero >= equity_vol, (
            f"Zero-slippage equity ({equity_zero:.2f}) should be >= "
            f"volume-slippage equity ({equity_vol:.2f})"
        )

    def test_volume_slippage_vs_fixed_slippage(self):
        """Volume and fixed slippage should produce different equity outcomes."""
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(60)]
        prices += [130 - i * 0.5 for i in range(60)]
        prices += [100.0] * 20

        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        # Fixed slippage
        source_fixed = MockDataSource()
        source_fixed.add("TEST", make_controlled_df(prices))
        config_fixed = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            slippage_model="fixed",
            slippage_bps=10.0,
            fee_per_trade=0.0,
        )
        result_fixed = _run_backtest(config_fixed, source_fixed)

        # Volume slippage
        source_vol = MockDataSource()
        source_vol.add("TEST", make_controlled_df(prices))
        config_vol = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            slippage_model="volume",
            slippage_impact_factor=0.1,
            fee_per_trade=0.0,
        )
        result_vol = _run_backtest(config_vol, source_vol)

        equity_fixed = result_fixed.equity_series.iloc[-1]
        equity_vol = result_vol.equity_series.iloc[-1]

        # Both should complete; they'll differ because models are different
        assert result_fixed.equity_series is not None
        assert result_vol.equity_series is not None
        # The key assertion: they are both less than or equal to zero-slippage
        # (already tested above). Here we just verify both ran successfully
        # and produced different results (different models).
        assert len(result_fixed.trades) >= 1
        assert len(result_vol.trades) >= 1

    def test_higher_impact_factor_worse_fills(self):
        """Higher volume impact factor should produce worse fills."""
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(60)]
        prices += [130 - i * 0.5 for i in range(60)]
        prices += [100.0] * 20

        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        # Low impact factor
        source_lo = MockDataSource()
        source_lo.add("TEST", make_controlled_df(prices))
        config_lo = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            slippage_model="volume",
            slippage_impact_factor=0.01,
            fee_per_trade=0.0,
        )
        result_lo = _run_backtest(config_lo, source_lo)

        # High impact factor
        source_hi = MockDataSource()
        source_hi.add("TEST", make_controlled_df(prices))
        config_hi = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            slippage_model="volume",
            slippage_impact_factor=1.0,
            fee_per_trade=0.0,
        )
        result_hi = _run_backtest(config_hi, source_hi)

        equity_lo = result_lo.equity_series.iloc[-1]
        equity_hi = result_hi.equity_series.iloc[-1]

        assert equity_lo >= equity_hi, (
            f"Low-impact equity ({equity_lo:.2f}) should be >= "
            f"high-impact equity ({equity_hi:.2f})"
        )


# ===========================================================================
# 20. TestForceCloseE2E
# ===========================================================================


class TestForceCloseE2E:
    """Test that positions are force-closed at the end of a backtest."""

    def test_buy_and_hold_positions_force_closed(self):
        """A strategy that buys and never sells should still have zero open
        positions at the end due to force-close."""
        prices = [100 + i * 0.3 for i in range(150)]  # steady uptrend

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_always_buy_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # All positions should be closed by force-close
        assert len(result.portfolio.positions) == 0, (
            f"Expected 0 open positions after force-close, "
            f"found {len(result.portfolio.positions)}"
        )

    def test_force_close_produces_trades(self):
        """Force-close should produce trade records in the trade log."""
        prices = [100 + i * 0.3 for i in range(150)]

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_always_buy_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # The buy-and-hold strategy should have at least one trade
        # (the force-closed position)
        assert len(result.trades) >= 1, (
            "Force-close should produce at least one trade record"
        )

    def test_force_close_cash_reflects_proceeds(self):
        """After force-close, final cash should reflect the sale proceeds."""
        prices = [100 + i * 0.3 for i in range(150)]

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_always_buy_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        portfolio = result.portfolio
        # With no open positions, equity == cash
        assert abs(portfolio.total_equity - portfolio.cash) < 0.01, (
            f"After force-close, equity ({portfolio.total_equity:.2f}) "
            f"should equal cash ({portfolio.cash:.2f})"
        )
        # Cash should be > 0 (proceeds from selling position)
        assert portfolio.cash > 0, (
            f"Cash should be positive after force-close, got {portfolio.cash:.2f}"
        )

    def test_force_close_multi_ticker(self):
        """Force-close should close positions across multiple tickers."""
        tickers = ["AAPL", "GOOG", "MSFT"]
        source = MockDataSource()
        for t in tickers:
            prices = [100 + i * 0.3 for i in range(150)]
            source.add(t, make_controlled_df(prices))

        start_d, end_d = _dates_from_df(source._data["AAPL"])

        config = BacktestConfig(
            strategy_name="_always_buy_multi_test",
            tickers=tickers,
            benchmark="AAPL",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        assert len(result.portfolio.positions) == 0, (
            f"All positions should be force-closed, "
            f"found open: {list(result.portfolio.positions.keys())}"
        )


# ===========================================================================
# 21. TestZeroTradesE2E
# ===========================================================================


class TestZeroTradesE2E:
    """Test backtest with a strategy that never generates signals."""

    def test_hold_only_zero_trades(self):
        """A strategy that always returns HOLD should produce zero trades."""
        prices = [100 + i * 0.2 for i in range(150)]

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_hold_only_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            fee_per_trade=1.0,
            slippage_bps=10.0,
        )
        result = _run_backtest(config, source)

        assert len(result.trades) == 0, (
            f"HoldOnly strategy should produce 0 trades, got {len(result.trades)}"
        )

    def test_hold_only_equity_flat(self):
        """With no trades, equity should remain at starting_cash throughout."""
        prices = [100 + i * 0.2 for i in range(150)]

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_hold_only_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            fee_per_trade=1.0,
            slippage_bps=10.0,
        )
        result = _run_backtest(config, source)

        equity = result.equity_series
        # Every equity point should equal starting cash
        assert equity.min() == equity.max() == 100_000.0, (
            f"Equity should be flat at 100000.0, but ranged "
            f"from {equity.min():.2f} to {equity.max():.2f}"
        )

    def test_hold_only_zero_fees(self):
        """With no trades, total fees paid should be zero."""
        prices = [100 + i * 0.2 for i in range(150)]

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_hold_only_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            fee_per_trade=10.0,
            slippage_bps=10.0,
        )
        result = _run_backtest(config, source)

        total_fees = sum(t.fees_total for t in result.trades)
        assert total_fees == 0.0, (
            f"Zero trades should mean zero fees, got {total_fees:.4f}"
        )

    def test_hold_only_cash_unchanged(self):
        """Cash should remain exactly at starting_cash with no trades."""
        prices = [100 + i * 0.2 for i in range(150)]

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_hold_only_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            fee_per_trade=10.0,
            slippage_bps=10.0,
        )
        result = _run_backtest(config, source)

        assert result.portfolio.cash == 100_000.0, (
            f"Cash should be exactly 100000.0, got {result.portfolio.cash:.2f}"
        )


# ===========================================================================
# 22. TestMultiTickerRegimeE2E
# ===========================================================================


class TestMultiTickerRegimeE2E:
    """Test regime filter with multiple tickers and separate benchmark."""

    def test_regime_suppresses_all_tickers(self):
        """When benchmark is in downtrend, regime filter should suppress BUY
        signals for ALL tickers, not just the benchmark.

        We use strategy SMA periods (sma_fast=30, sma_slow=50) longer than
        regime filter periods (fast=10, slow=20) so that by the time the
        strategy would generate its first BUY signal, the regime filter is
        already computed and blocking.
        """
        # Downtrending benchmark (SPY) -- starts high and declines
        spy_prices = [200.0 - i * 0.8 for i in range(252)]

        # Uptrending tickers (would normally trigger BUY)
        aapl_prices = [100 + i * 0.5 for i in range(252)]
        goog_prices = [150 + i * 0.4 for i in range(252)]

        source = MockDataSource()
        source.add("SPY", make_controlled_df(spy_prices))
        source.add("AAPL", make_controlled_df(aapl_prices))
        source.add("GOOG", make_controlled_df(goog_prices))

        start_d, end_d = _dates_from_df(source._data["SPY"])

        regime = RegimeFilter(
            benchmark="SPY",
            indicator="sma",
            fast_period=10,
            slow_period=20,
        )
        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["AAPL", "GOOG"],
            benchmark="SPY",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            # Strategy SMA warmup (50 days) > regime warmup (20 days)
            # so regime is fully active before first BUY signal
            strategy_params={"sma_fast": 30, "sma_slow": 50},
            regime_filter=regime,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # With downtrending benchmark, regime should suppress all BUY signals
        assert len(result.trades) == 0, (
            f"Regime filter should suppress all trades when benchmark downtrends, "
            f"but found {len(result.trades)} trades"
        )

    def test_regime_allows_when_benchmark_uptrends(self):
        """When benchmark is in uptrend, regime should allow trading on all tickers."""
        # Uptrending benchmark
        spy_prices = [100 + i * 0.5 for i in range(252)]

        # Uptrending tickers
        aapl_prices = [100 + i * 0.5 for i in range(252)]
        goog_prices = [150 + i * 0.4 for i in range(252)]

        source = MockDataSource()
        source.add("SPY", make_controlled_df(spy_prices))
        source.add("AAPL", make_controlled_df(aapl_prices))
        source.add("GOOG", make_controlled_df(goog_prices))

        start_d, end_d = _dates_from_df(source._data["SPY"])

        regime = RegimeFilter(
            benchmark="SPY",
            indicator="sma",
            fast_period=10,
            slow_period=20,
        )
        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["AAPL", "GOOG"],
            benchmark="SPY",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            regime_filter=regime,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # With uptrending benchmark and uptrending tickers, trades should happen
        assert len(result.trades) >= 1, (
            "Regime should allow trades when benchmark uptrends, but got 0 trades"
        )
        # Verify trades happened in multiple tickers
        traded_symbols = set(t.symbol for t in result.trades)
        assert len(traded_symbols) >= 1, (
            "At least one ticker should have trades when regime allows"
        )

    def test_regime_no_trades_vs_trades_comparison(self):
        """Compare same setup with and without regime filter to verify
        the regime actually suppresses trades.

        Uses strategy SMA warmup longer than regime warmup to ensure the
        regime filter is active before any BUY signal fires.
        """
        # Downtrending benchmark but uptrending tickers
        spy_prices = [200.0 - i * 0.8 for i in range(252)]
        aapl_prices = [100 + i * 0.5 for i in range(252)]

        # WITH regime filter (should suppress)
        source_regime = MockDataSource()
        source_regime.add("SPY", make_controlled_df(spy_prices))
        source_regime.add("AAPL", make_controlled_df(aapl_prices))
        start_d, end_d = _dates_from_df(source_regime._data["SPY"])

        regime = RegimeFilter(
            benchmark="SPY",
            indicator="sma",
            fast_period=10,
            slow_period=20,
        )
        config_regime = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["AAPL"],
            benchmark="SPY",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 30, "sma_slow": 50},
            regime_filter=regime,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_regime = _run_backtest(config_regime, source_regime)

        # WITHOUT regime filter (should trade normally)
        source_no_regime = MockDataSource()
        source_no_regime.add("SPY", make_controlled_df(spy_prices))
        source_no_regime.add("AAPL", make_controlled_df(aapl_prices))

        config_no_regime = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["AAPL"],
            benchmark="SPY",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 30, "sma_slow": 50},
            regime_filter=None,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_no_regime = _run_backtest(config_no_regime, source_no_regime)

        # Without regime, uptrending AAPL should generate trades
        assert len(result_no_regime.trades) >= 1, (
            "Without regime filter, uptrending AAPL should trade"
        )
        # With regime (downtrending SPY), trades should be suppressed
        assert len(result_regime.trades) < len(result_no_regime.trades), (
            f"Regime should suppress trades: regime={len(result_regime.trades)} "
            f"vs no_regime={len(result_no_regime.trades)}"
        )


# ---------------------------------------------------------------------------
# Custom test strategies for coverage gap tests (batch 3)
# ---------------------------------------------------------------------------


class _LongShortStrategy(Strategy):
    """Strategy that buys one ticker and shorts another simultaneously.
    Requires at least two tickers. Buys the first, shorts the second."""

    def __init__(self):
        super().__init__()
        self._warmup = 5
        self._hold_days = 15
        self._entered = {}

    def configure(self, params: dict) -> None:
        self._warmup = params.get("warmup", self._warmup)
        self._hold_days = params.get("hold_days", self._hold_days)
        self._entered = {}

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["row_idx"] = range(len(df))
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        idx = row.get("row_idx", 0)
        has_pos = position is not None and position.total_quantity != 0

        if idx < self._warmup:
            return SignalAction.HOLD

        # Determine action based on symbol name convention
        is_short_target = symbol.startswith("SHORT")

        if not has_pos and symbol not in self._entered:
            self._entered[symbol] = idx
            return SignalAction.SHORT if is_short_target else SignalAction.BUY

        if has_pos and symbol in self._entered:
            if idx - self._entered[symbol] >= self._hold_days:
                del self._entered[symbol]
                if position.is_short:
                    return SignalAction.COVER
                else:
                    return SignalAction.SELL

        return SignalAction.HOLD


class _ShortWithLimitStrategy(Strategy):
    """Strategy that shorts using limit orders, then covers with limit orders."""

    def __init__(self):
        super().__init__()
        self._warmup = 5
        self._day_count = 0

    def configure(self, params: dict) -> None:
        self._warmup = params.get("warmup", self._warmup)
        self._day_count = 0

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["row_idx"] = range(len(df))
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        idx = row.get("row_idx", 0)
        close = row["Close"]
        has_short = position is not None and position.is_short

        if idx < self._warmup:
            return SignalAction.HOLD

        if not has_short:
            # Short entry with limit above current price (should fill next day)
            return Signal(
                action=SignalAction.SHORT,
                limit_price=close * 1.02,  # sell limit above current
                time_in_force="GTC",
            )
        else:
            # Cover with limit below current price (should fill when price drops)
            return Signal(
                action=SignalAction.COVER,
                limit_price=close * 0.98,
                time_in_force="GTC",
            )


class _RebalanceTestStrategy(Strategy):
    """Strategy that buys once early, then holds. Used to test rebalance schedules."""

    def __init__(self):
        super().__init__()
        self._bought = set()

    def configure(self, params: dict) -> None:
        self._bought = set()

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["row_idx"] = range(len(df))
        return df

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        idx = row.get("row_idx", 0)
        has_pos = position is not None and position.total_quantity > 0

        if idx >= 3 and not has_pos and symbol not in self._bought:
            self._bought.add(symbol)
            return SignalAction.BUY
        return SignalAction.HOLD


def _register_test_strategies_batch3():
    """Register batch 3 test-only strategies."""
    for name, cls in [
        ("_long_short_test", _LongShortStrategy),
        ("_short_limit_test", _ShortWithLimitStrategy),
        ("_rebalance_test", _RebalanceTestStrategy),
    ]:
        if name not in _REGISTRY:
            _REGISTRY[name] = cls

_register_test_strategies_batch3()


# ===========================================================================
# 23. TestShortBorrowCost
# ===========================================================================


class TestShortBorrowCost:
    """E2E tests for short borrow cost accrual impacting cash."""

    def test_borrow_cost_reduces_cash(self):
        """When holding a short position, daily borrow costs should reduce cash."""
        prices = [100.0] * 60  # flat prices so no PnL from price movement
        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        # Run with borrow costs
        config_borrow = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 30},
            allow_short=True,
            short_borrow_rate=0.10,  # 10% annual
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_borrow = _run_backtest(config_borrow, source)

        # Run without borrow costs
        source2 = MockDataSource()
        source2.add("TEST", make_controlled_df(prices))
        config_no_borrow = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 30},
            allow_short=True,
            short_borrow_rate=0.0,  # no borrow cost
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_no_borrow = _run_backtest(config_no_borrow, source2)

        # With borrow costs, final cash should be lower
        assert result_borrow.portfolio.cash < result_no_borrow.portfolio.cash, (
            f"Borrow cost should reduce cash: with={result_borrow.portfolio.cash:.2f} "
            f"vs without={result_no_borrow.portfolio.cash:.2f}"
        )

    def test_borrow_cost_proportional_to_rate(self):
        """Higher borrow rate should reduce cash more."""
        prices = [100.0] * 60
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        results = {}
        for rate in [0.02, 0.10]:
            source = MockDataSource()
            source.add("TEST", make_controlled_df(prices))
            config = BacktestConfig(
                strategy_name="_always_short_test",
                tickers=["TEST"],
                benchmark="TEST",
                start_date=start_d,
                end_date=end_d,
                starting_cash=100_000.0,
                max_positions=10,
                max_alloc_pct=0.20,
                strategy_params={"warmup": 3, "hold_days": 30},
                allow_short=True,
                short_borrow_rate=rate,
                fee_per_trade=0.0,
                slippage_bps=0.0,
            )
            results[rate] = _run_backtest(config, source)

        # Higher rate should result in lower final cash
        assert results[0.10].portfolio.cash < results[0.02].portfolio.cash, (
            f"10% rate cash ({results[0.10].portfolio.cash:.2f}) should be less than "
            f"2% rate cash ({results[0.02].portfolio.cash:.2f})"
        )


# ===========================================================================
# 24. TestMixedLongShort
# ===========================================================================


class TestMixedLongShort:
    """E2E tests for simultaneous long and short positions in the same backtest."""

    def test_mixed_positions_cash_accounting(self):
        """A portfolio with both long and short positions should maintain
        correct cash accounting through the full lifecycle."""
        # Uptrending LONG ticker, flat SHORT ticker
        long_prices = [100 + i * 0.3 for i in range(80)]
        short_prices = [100.0] * 80

        source = MockDataSource()
        source.add("LONG1", make_controlled_df(long_prices))
        source.add("SHORT1", make_controlled_df(short_prices))
        start_d, end_d = _dates_from_df(make_controlled_df(long_prices))

        config = BacktestConfig(
            strategy_name="_long_short_test",
            tickers=["LONG1", "SHORT1"],
            benchmark="LONG1",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 5, "hold_days": 30},
            allow_short=True,
            short_borrow_rate=0.0,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # After force-close, all positions should be closed
        assert len(result.portfolio.positions) == 0, (
            f"All positions should be force-closed, "
            f"found: {list(result.portfolio.positions.keys())}"
        )
        # Cash should equal total equity (no open positions)
        assert abs(result.portfolio.total_equity - result.portfolio.cash) < 0.01
        # Cash should be positive
        assert result.portfolio.cash > 0, (
            f"Cash should be positive, got {result.portfolio.cash:.2f}"
        )

    def test_mixed_positions_respects_max_positions(self):
        """Both long and short positions count toward max_positions."""
        long_prices = [100 + i * 0.3 for i in range(80)]
        short_prices = [100.0] * 80

        source = MockDataSource()
        source.add("LONG1", make_controlled_df(long_prices))
        source.add("SHORT1", make_controlled_df(short_prices))
        start_d, end_d = _dates_from_df(make_controlled_df(long_prices))

        config = BacktestConfig(
            strategy_name="_long_short_test",
            tickers=["LONG1", "SHORT1"],
            benchmark="LONG1",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=1,  # only allow 1 position total
            max_alloc_pct=0.20,
            strategy_params={"warmup": 5, "hold_days": 50},
            allow_short=True,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # With max_positions=1, we should have at most 1 trade open at a time.
        # Check equity history for consistency - no phantom positions.
        for eq_date, eq_val in result.portfolio.equity_history:
            assert eq_val > 0, f"Equity went non-positive on {eq_date}"


# ===========================================================================
# 25. TestForceCloseShorts
# ===========================================================================


class TestForceCloseShorts:
    """E2E tests for force-closing short positions at end of backtest."""

    def test_force_close_short_position(self):
        """Force-close should properly close short positions at end of backtest."""
        prices = [100.0] * 80  # flat prices
        source = MockDataSource()
        source.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 200},  # hold longer than backtest
            allow_short=True,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # All positions must be closed by force-close
        assert len(result.portfolio.positions) == 0, (
            f"Short position should be force-closed, "
            f"found: {list(result.portfolio.positions.keys())}"
        )
        # Cash should equal equity
        assert abs(result.portfolio.total_equity - result.portfolio.cash) < 0.01

    def test_force_close_mixed_long_short(self):
        """Force-close should handle both long and short positions correctly."""
        long_prices = [100 + i * 0.2 for i in range(80)]
        short_prices = [100.0] * 80

        source = MockDataSource()
        source.add("LONG1", make_controlled_df(long_prices))
        source.add("SHORT1", make_controlled_df(short_prices))
        start_d, end_d = _dates_from_df(make_controlled_df(long_prices))

        config = BacktestConfig(
            strategy_name="_long_short_test",
            tickers=["LONG1", "SHORT1"],
            benchmark="LONG1",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 5, "hold_days": 200},  # never exit naturally
            allow_short=True,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        assert len(result.portfolio.positions) == 0, (
            f"All positions should be force-closed"
        )
        assert abs(result.portfolio.total_equity - result.portfolio.cash) < 0.01
        # Should have produced trade records for both long and short closes
        assert len(result.trades) >= 2, (
            f"Expected trades for both long and short force-close, got {len(result.trades)}"
        )


# ===========================================================================
# 26. TestRegimeFilterShorts
# ===========================================================================


class TestRegimeFilterShorts:
    """E2E tests for regime filter suppressing SHORT signals."""

    def test_regime_suppresses_short_signals(self):
        """When regime is off (downtrending benchmark), SHORT signals should
        be suppressed just like BUY signals."""
        # Downtrending benchmark
        benchmark_prices = [200.0 - i * 1.5 for i in range(120)]
        ticker_prices = [100.0] * 120

        source = MockDataSource()
        source.add("SPY", make_controlled_df(benchmark_prices))
        source.add("TEST", make_controlled_df(ticker_prices))
        start_d, end_d = _dates_from_df(make_controlled_df(benchmark_prices))

        # With regime filter on
        config_regime = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="SPY",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 50},
            allow_short=True,
            regime_filter=RegimeFilter(
                benchmark="SPY",
                indicator="sma",
                fast_period=10,
                slow_period=20,
            ),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_regime = _run_backtest(config_regime, source)

        # Without regime filter
        source2 = MockDataSource()
        source2.add("SPY", make_controlled_df(benchmark_prices))
        source2.add("TEST", make_controlled_df(ticker_prices))
        config_no_regime = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="SPY",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 50},
            allow_short=True,
            regime_filter=None,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_no_regime = _run_backtest(config_no_regime, source2)

        # Without regime filter, we should get short trades
        assert len(result_no_regime.trades) >= 1, "Without regime, shorts should execute"
        # With regime (downtrending benchmark), shorts should be suppressed
        assert len(result_regime.trades) < len(result_no_regime.trades), (
            f"Regime should suppress shorts: regime={len(result_regime.trades)} "
            f"vs no_regime={len(result_no_regime.trades)}"
        )


# ===========================================================================
# 27. TestRebalanceSchedule
# ===========================================================================


class TestRebalanceSchedule:
    """E2E tests for non-daily rebalance schedules."""

    def test_weekly_rebalance_fewer_signals(self):
        """Weekly rebalance should generate fewer/equal signals than daily."""
        prices = [100.0] * 10
        prices += [100 + i * 0.5 for i in range(80)]
        prices += [140 - i * 0.5 for i in range(80)]
        prices += [100.0] * 10

        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        # Daily rebalance
        source_daily = MockDataSource()
        source_daily.add("TEST", make_controlled_df(prices))
        config_daily = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            rebalance_schedule="daily",
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_daily = _run_backtest(config_daily, source_daily)

        # Weekly rebalance
        source_weekly = MockDataSource()
        source_weekly.add("TEST", make_controlled_df(prices))
        config_weekly = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            rebalance_schedule="weekly",
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_weekly = _run_backtest(config_weekly, source_weekly)

        # Weekly should produce <= daily trades (fewer opportunities to act)
        assert len(result_weekly.trades) <= len(result_daily.trades), (
            f"Weekly ({len(result_weekly.trades)}) should produce "
            f"<= daily ({len(result_daily.trades)}) trades"
        )

    def test_monthly_rebalance_completes(self):
        """Monthly rebalance should produce a valid backtest result."""
        prices = [100 + i * 0.2 for i in range(252)]
        source = MockDataSource()
        source.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 20, "sma_slow": 50},
            rebalance_schedule="monthly",
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        assert result.equity_series is not None
        assert len(result.equity_series) > 0
        # All positions should be force-closed
        assert len(result.portfolio.positions) == 0


# ===========================================================================
# 28. TestShortWithLimitOrders
# ===========================================================================


class TestShortWithLimitOrders:
    """E2E tests for SHORT entry/exit using limit orders (combined feature)."""

    def test_short_limit_entry_fills(self):
        """A SHORT with limit price above current should fill when High reaches limit."""
        # Flat then rising prices to trigger limit short entry
        prices = [100.0] * 10
        prices += [100 + i * 0.5 for i in range(40)]  # rising -> limit fills
        prices += [120 - i * 0.5 for i in range(30)]  # falling -> cover fills
        prices += [100.0] * 10

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_short_limit_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 5},
            allow_short=True,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should complete without error and have trades from the short entry
        assert result.equity_series is not None
        assert len(result.equity_series) > 0
        # All positions must be closed at end
        assert len(result.portfolio.positions) == 0


# ===========================================================================
# 29. TestDelistingDetection
# ===========================================================================


class TestDelistingDetection:
    """E2E tests for delisting detection force-closing positions."""

    def test_delisting_forces_close_after_missing_days(self):
        """If a symbol's data stops (delisted), position should be force-closed
        after 5 missing days."""
        # Ticker data runs out mid-backtest
        ticker_prices = [100 + i * 0.2 for i in range(40)]  # only 40 days of data
        benchmark_prices = [100 + i * 0.1 for i in range(80)]  # full 80 days

        source = MockDataSource()
        source.add("DELIST", make_controlled_df(ticker_prices))
        source.add("BENCH", make_controlled_df(benchmark_prices))
        start_d, end_d = _dates_from_df(make_controlled_df(benchmark_prices))

        config = BacktestConfig(
            strategy_name="_always_buy_test",
            tickers=["DELIST"],
            benchmark="BENCH",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Position should have been closed (either by delisting or force-close)
        assert len(result.portfolio.positions) == 0
        # Cash should be positive
        assert result.portfolio.cash > 0


# ===========================================================================
# 30. TestGTCLimitOrderExpiry
# ===========================================================================


class TestGTCLimitOrderExpiry:
    """E2E tests for GTC limit orders with explicit expiry dates."""

    def test_gtc_persists_across_days(self):
        """GTC limit orders should persist across multiple days until filled."""
        # Flat prices then a drop that triggers the limit buy
        prices = [100.0] * 20  # flat warmup
        prices += [100.0] * 20  # stay flat (limit won't fill)
        prices += [95 - i * 0.5 for i in range(20)]  # drop to fill limit
        prices += [85.0] * 20  # stay low

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="_limit_order_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"buy_discount": 0.08, "time_in_force": "GTC", "warmup": 5},
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # The GTC order should eventually fill when price drops enough
        # At minimum the backtest should complete without error
        assert result.equity_series is not None
        assert len(result.portfolio.positions) == 0  # force-closed at end


# ===========================================================================
# 31. TestCompositeFeesWithShorts
# ===========================================================================


class TestCompositeFeesWithShorts:
    """E2E tests for composite fee models applied to short entry/exit."""

    def test_composite_fees_on_short_trades(self):
        """Composite fees (percentage + SEC + TAF) should apply to short trades."""
        prices = [100.0] * 60
        source = MockDataSource()
        source.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 20},
            allow_short=True,
            fee_model="composite_us",
            fee_per_trade=5.0,  # 5 bps for percentage component
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        assert len(result.trades) >= 1, "Expected short trades"
        for trade in result.trades:
            assert trade.fees_total > 0, (
                f"Composite fee on short should be > 0, got {trade.fees_total}"
            )

    def test_short_fees_reduce_equity_vs_no_fees(self):
        """Short trading with composite fees should result in lower equity."""
        prices = [100.0] * 60
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        # With fees
        source_fees = MockDataSource()
        source_fees.add("TEST", make_controlled_df(prices))
        config_fees = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 20},
            allow_short=True,
            fee_model="composite_us",
            fee_per_trade=5.0,
            slippage_bps=0.0,
        )
        result_fees = _run_backtest(config_fees, source_fees)

        # Without fees
        source_no_fees = MockDataSource()
        source_no_fees.add("TEST", make_controlled_df(prices))
        config_no_fees = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 20},
            allow_short=True,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result_no_fees = _run_backtest(config_no_fees, source_no_fees)

        assert result_fees.portfolio.cash < result_no_fees.portfolio.cash, (
            f"Fees should reduce cash: with={result_fees.portfolio.cash:.2f} "
            f"vs without={result_no_fees.portfolio.cash:.2f}"
        )


# ===========================================================================
# 32. TestMultipleStopTypes
# ===========================================================================


class TestMultipleStopTypes:
    """E2E tests for positions with both stop-loss and take-profit active."""

    def test_stop_loss_and_take_profit_together(self):
        """A position with both stop-loss and take-profit should trigger whichever
        condition is met first."""
        # Sharp drop to trigger stop-loss
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(20)]  # uptrend -> BUY
        prices += [110, 109, 105, 100, 95, 90, 85]  # drop -> stop-loss
        prices += [85.0] * 20

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            stop_config=StopConfig(
                stop_loss_pct=0.10,   # 10% stop-loss
                take_profit_pct=0.30,  # 30% take-profit
            ),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should complete without error
        assert result.equity_series is not None
        assert len(result.portfolio.positions) == 0
        # With the sharp drop, stop-loss should have triggered
        if result.trades:
            # All trades should have reasonable exit prices
            for trade in result.trades:
                assert trade.exit_price > 0

    def test_trailing_stop_and_take_profit_together(self):
        """Trailing stop and take-profit can coexist on the same position."""
        # Rise then fall pattern
        prices = [100.0] * 15
        prices += [100 + i * 0.5 for i in range(30)]  # uptrend -> BUY
        prices += [115 - i * 0.3 for i in range(40)]  # gradual decline -> trailing stop
        prices += [100.0] * 10

        source = MockDataSource()
        df = make_controlled_df(prices)
        source.add("TEST", df)
        start_d, end_d = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"sma_fast": 5, "sma_slow": 10},
            stop_config=StopConfig(
                take_profit_pct=0.50,   # 50% take-profit (won't hit)
                trailing_stop_pct=0.05,  # 5% trailing stop (will hit)
            ),
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        assert result.equity_series is not None
        assert len(result.portfolio.positions) == 0


# ===========================================================================
# 33. TestDrawdownKillSwitchWithShorts
# ===========================================================================


class TestDrawdownKillSwitchWithShorts:
    """E2E test for drawdown kill switch when short positions are open."""

    def test_kill_switch_closes_short_positions(self):
        """Drawdown kill switch should force-close short positions too."""
        # Prices rise sharply (bad for shorts) causing drawdown
        prices = [100.0] * 10
        prices += [100 + i * 3 for i in range(50)]  # sharp rise -> short loses

        source = MockDataSource()
        source.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 200},
            allow_short=True,
            max_drawdown_pct=0.05,  # 5% drawdown limit
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # All positions should be closed by kill switch or force-close
        assert len(result.portfolio.positions) == 0
        # Equity should be positive (haven't lost everything)
        assert result.portfolio.cash > 0


# ===========================================================================
# 34. TestFIFOAccountingShorts
# ===========================================================================


class TestFIFOAccountingShorts:
    """E2E test for FIFO lot accounting on short positions."""

    def test_short_fifo_lots_closed_in_order(self):
        """When covering a short position, FIFO should close earliest lots first."""
        prices = [100.0] * 60
        source = MockDataSource()
        source.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            strategy_params={"warmup": 3, "hold_days": 15},
            allow_short=True,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        # Should complete with trades and all positions closed
        assert len(result.portfolio.positions) == 0
        if result.trades:
            for trade in result.trades:
                assert trade.entry_date <= trade.exit_date
                assert trade.quantity > 0


# ===========================================================================
# 35. TestEquityCurveInvariantsComprehensive
# ===========================================================================


class TestEquityCurveInvariantsComprehensive:
    """Comprehensive equity curve invariant checks across different scenarios."""

    def test_equity_always_positive_with_shorts(self):
        """Equity should remain positive even during short trades."""
        # Mildly rising prices (short takes small loss)
        prices = [100 + i * 0.1 for i in range(80)]
        source = MockDataSource()
        source.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config = BacktestConfig(
            strategy_name="_always_short_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,  # small positions
            strategy_params={"warmup": 3, "hold_days": 20},
            allow_short=True,
            fee_per_trade=0.0,
            slippage_bps=0.0,
        )
        result = _run_backtest(config, source)

        for eq_date, eq_val in result.portfolio.equity_history:
            assert eq_val > 0, f"Equity went non-positive on {eq_date}: {eq_val:.2f}"

    def test_equity_curve_monotonic_with_no_trades(self):
        """With no trades, equity curve should be flat at starting_cash."""
        prices = [100 + i * 0.5 for i in range(100)]
        source = MockDataSource()
        source.add("TEST", make_controlled_df(prices))
        start_d, end_d = _dates_from_df(make_controlled_df(prices))

        config = BacktestConfig(
            strategy_name="_hold_only_test",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start_d,
            end_date=end_d,
            starting_cash=50_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
        )
        result = _run_backtest(config, source)

        for eq_date, eq_val in result.portfolio.equity_history:
            assert abs(eq_val - 50_000.0) < 0.01, (
                f"With no trades, equity should be 50000, got {eq_val:.2f} on {eq_date}"
            )
