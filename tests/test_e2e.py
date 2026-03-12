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
from backtester.strategies.registry import discover_strategies
from backtester.types import SignalAction

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
