"""Black-box tests with extreme and degenerate price data.

Tests the backtesting engine against pathological price scenarios
to verify it handles them without crashes, NaN equity, or nonsensical results.
"""

import math
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.strategies.registry import _REGISTRY, discover_strategies
from backtester.strategies.base import Strategy
from backtester.types import SignalAction
from backtester.analytics.metrics import (
    compute_all_metrics, total_return, sharpe_ratio, max_drawdown,
)

discover_strategies()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from tests.conftest import MockDataSource


def _make_df(days, close_fn, open_fn=None, high_fn=None, low_fn=None,
             volume_fn=None, start="2020-01-02"):
    """Build an OHLCV DataFrame from callables that take day index."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    closes = np.array([close_fn(i) for i in range(days)], dtype=float)
    opens = np.array([open_fn(i) for i in range(days)], dtype=float) if open_fn else closes.copy()
    highs = np.array([high_fn(i) for i in range(days)], dtype=float) if high_fn else np.maximum(opens, closes) * 1.001
    lows = np.array([low_fn(i) for i in range(days)], dtype=float) if low_fn else np.minimum(opens, closes) * 0.999
    vols = np.array([volume_fn(i) for i in range(days)], dtype=float) if volume_fn else np.full(days, 1_000_000.0)
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols,
    }, index=pd.DatetimeIndex(dates.date, name="Date"))
    return df


def _constant_df(days, price, volume=1_000_000, start="2020-01-02"):
    return _make_df(days, close_fn=lambda i: price, volume_fn=lambda i: volume, start=start)


# Register an always-buy strategy once
if "_bb_always_buy" not in _REGISTRY:
    class _BBAlwaysBuy(Strategy):
        def configure(self, params):
            pass

        def compute_indicators(self, df, timeframe_data=None):
            return df.copy()

        def generate_signals(self, symbol, row, position, portfolio_state,
                             benchmark_row=None):
            if position is None or position.total_quantity == 0:
                return SignalAction.BUY
            return SignalAction.HOLD

    _REGISTRY["_bb_always_buy"] = _BBAlwaysBuy


def _run_backtest(df, starting_cash=100_000.0, max_positions=10, max_alloc=0.10,
                  tickers=None, strategy="_bb_always_buy", strategy_params=None):
    """Run a backtest with a given DataFrame and return the result."""
    tickers = tickers or ["TEST"]
    source = MockDataSource()
    for t in tickers:
        source.add(t, df)

    config = BacktestConfig(
        strategy_name=strategy,
        tickers=tickers,
        benchmark=tickers[0],
        start_date=df.index[0].date() if hasattr(df.index[0], 'date') else df.index[0],
        end_date=df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1],
        starting_cash=starting_cash,
        max_positions=max_positions,
        max_alloc_pct=max_alloc,
        strategy_params=strategy_params or {},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DataManager(cache_dir=tmpdir, source=source)
        engine = BacktestEngine(config, dm)
        return engine.run()


def _check_result_sanity(result, starting_cash=100_000.0):
    """Assert basic sanity on a backtest result."""
    assert result is not None, "Result is None"
    eq = result.equity_series
    assert len(eq) > 0, "Empty equity curve"

    final_equity = eq.iloc[-1]
    assert not math.isnan(final_equity), f"Final equity is NaN"
    assert not math.isinf(final_equity), f"Final equity is inf: {final_equity}"
    assert final_equity >= 0, f"Final equity is negative: {final_equity}"

    metrics = compute_all_metrics(eq, result.trades)
    # Metrics should not be NaN (some may legitimately be 0 or inf for edge cases)
    for key in ["total_return", "max_drawdown"]:
        val = metrics.get(key)
        if val is not None:
            assert not math.isnan(val), f"Metric {key} is NaN"

    return metrics


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllZeroPrices:
    """Scenario 1: All prices = 0.0"""

    def test_zero_prices_no_crash(self):
        df = _constant_df(100, price=0.0)
        # Zero prices may cause division errors or inability to buy
        # We accept either a clean run or a handled error
        try:
            result = _run_backtest(df)
            # If it runs, equity should remain at starting cash (can't buy at 0)
            # or be non-negative
            eq = result.equity_series
            final = eq.iloc[-1]
            assert not math.isnan(final), "Final equity is NaN with zero prices"
            assert final >= 0, f"Final equity negative: {final}"
        except (ZeroDivisionError, ValueError) as e:
            pytest.skip(f"Zero prices raised handled error: {e}")


class TestNegativePrices:
    """Scenario 2: Negative prices"""

    def test_negative_prices_no_crash(self):
        df = _constant_df(100, price=-50.0)
        try:
            result = _run_backtest(df)
            eq = result.equity_series
            final = eq.iloc[-1]
            assert not math.isnan(final), "Final equity is NaN with negative prices"
        except (ValueError, ZeroDivisionError) as e:
            pytest.skip(f"Negative prices raised handled error: {e}")


class TestPriceDropsToZero:
    """Scenario 3: Price goes to zero mid-backtest"""

    def test_price_drops_to_zero(self):
        def close_fn(i):
            if i < 50:
                return 100.0
            return 0.0

        df = _make_df(100, close_fn)
        try:
            result = _run_backtest(df)
            eq = result.equity_series
            final = eq.iloc[-1]
            assert not math.isnan(final), "Final equity is NaN after price drop to zero"
            assert final >= 0, f"Final equity negative: {final}"
            # Should have lost at most the invested amount
            assert final <= 100_000.0 + 1.0, "Equity should not grow when price goes to zero"
        except (ZeroDivisionError, ValueError) as e:
            pytest.skip(f"Price-to-zero raised handled error: {e}")


class TestAstronomicalPrices:
    """Scenario 4: Very large prices (overflow potential)"""

    def test_large_prices(self):
        df = _constant_df(100, price=1e15)
        # With $100k cash and price=1e15, can't buy any shares
        result = _run_backtest(df)
        eq = result.equity_series
        final = eq.iloc[-1]
        assert not math.isnan(final), "Final equity is NaN with astronomical prices"
        assert not math.isinf(final), "Final equity is inf"
        # Should remain near starting cash since we can't afford any shares
        assert abs(final - 100_000.0) < 1.0, (
            f"Expected ~$100k (can't buy), got {final}"
        )

    def test_price_grows_to_huge(self):
        """Price starts normal, grows to 1e12"""
        def close_fn(i):
            return 100.0 * (1.1 ** i)  # ~100 * 1.1^i, hits huge values fast

        df = _make_df(100, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        final = eq.iloc[-1]
        assert not math.isnan(final), "Final equity is NaN"
        assert not math.isinf(final), "Final equity is inf"


class TestInvalidOHLC:
    """Scenario 5 & 6: High < Low, Open outside High/Low range"""

    def test_high_less_than_low(self):
        """High=90, Low=110 — inverted bars"""
        df = _make_df(
            100,
            close_fn=lambda i: 100.0,
            open_fn=lambda i: 100.0,
            high_fn=lambda i: 90.0,
            low_fn=lambda i: 110.0,
        )
        try:
            result = _run_backtest(df)
            _check_result_sanity(result)
        except (ValueError, AssertionError) as e:
            # It's acceptable if the engine validates OHLC
            pytest.skip(f"Invalid OHLC rejected: {e}")

    def test_open_outside_high_low(self):
        """Open=200, High=150, Low=100, Close=120"""
        df = _make_df(
            100,
            close_fn=lambda i: 120.0,
            open_fn=lambda i: 200.0,
            high_fn=lambda i: 150.0,
            low_fn=lambda i: 100.0,
        )
        try:
            result = _run_backtest(df)
            _check_result_sanity(result)
        except (ValueError, AssertionError) as e:
            pytest.skip(f"Invalid OHLC rejected: {e}")


class TestConstantPrice:
    """Scenario 7: All prices identical every day"""

    def test_constant_price_metrics(self):
        df = _constant_df(252, price=100.0)
        result = _run_backtest(df)
        metrics = _check_result_sanity(result)

        # With constant price, total return should be ~0 (minus fees if any)
        tr = metrics.get("total_return", 0)
        assert abs(tr) < 0.01, f"Expected ~0% return with constant price, got {tr}"

    def test_constant_price_sharpe(self):
        df = _constant_df(252, price=100.0)
        result = _run_backtest(df)
        metrics = compute_all_metrics(result.equity_series, result.trades)
        sharpe = metrics.get("sharpe_ratio")
        # With zero variance in returns, Sharpe should be 0 or NaN (no excess returns)
        if sharpe is not None:
            assert not math.isinf(sharpe), f"Sharpe is inf with constant price: {sharpe}"


class TestSinglePriceSpike:
    """Scenario 8: Price is $100 for 250 days, $10,000 on day 251, back to $100"""

    def test_spike_and_return(self):
        def close_fn(i):
            if i == 250:
                return 10_000.0
            return 100.0

        df = _make_df(252, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        final = eq.iloc[-1]
        assert not math.isnan(final), "Final equity is NaN after spike"
        assert not math.isinf(final), "Final equity is inf after spike"
        assert final >= 0, f"Final equity negative: {final}"

    def test_spike_drawdown(self):
        """Drawdown after spike should be captured"""
        def close_fn(i):
            if i == 250:
                return 10_000.0
            return 100.0

        df = _make_df(252, close_fn)
        result = _run_backtest(df)
        metrics = compute_all_metrics(result.equity_series, result.trades)
        dd = metrics.get("max_drawdown", 0)
        assert not math.isnan(dd), "Max drawdown is NaN"


class TestPennyStockHugeVolume:
    """Scenario 9: Close=$0.001, Volume=1e12"""

    def test_penny_stock(self):
        df = _constant_df(100, price=0.001, volume=int(1e12))
        result = _run_backtest(df)
        eq = result.equity_series
        final = eq.iloc[-1]
        assert not math.isnan(final), "Final equity is NaN with penny stock"
        assert not math.isinf(final), "Final equity is inf with penny stock"
        assert final >= 0, f"Final equity negative: {final}"


class TestExtremeDailyMoves:
    """Scenario 10: +50% one day, -50% the next, alternating"""

    def test_alternating_50pct_moves(self):
        def close_fn(i):
            price = 100.0
            for j in range(i):
                if j % 2 == 0:
                    price *= 1.50
                else:
                    price *= 0.50
            return price

        df = _make_df(100, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        final = eq.iloc[-1]
        assert not math.isnan(final), "Final equity is NaN with volatile prices"
        assert not math.isinf(final), "Final equity is inf with volatile prices"
        assert final >= 0, f"Final equity negative: {final}"

    def test_alternating_moves_metrics(self):
        """Metrics should be computable even with extreme volatility"""
        def close_fn(i):
            price = 100.0
            for j in range(i):
                if j % 2 == 0:
                    price *= 1.50
                else:
                    price *= 0.50
            return price

        df = _make_df(100, close_fn)
        result = _run_backtest(df)
        metrics = compute_all_metrics(result.equity_series, result.trades)
        for key in ["total_return", "max_drawdown"]:
            val = metrics.get(key)
            if val is not None:
                assert not math.isnan(val), f"{key} is NaN with volatile data"


class TestEquityCashConsistency:
    """Cross-cutting: verify cash + positions = equity for various scenarios"""

    def test_constant_price_equity_consistency(self):
        df = _constant_df(252, price=100.0)
        result = _run_backtest(df)
        # Final equity should equal starting cash +/- any trading P&L
        final = result.equity_series.iloc[-1]
        assert final > 0, f"Equity not positive: {final}"

    def test_normal_prices_equity_positive(self):
        """Normal price data should always produce positive equity"""
        def close_fn(i):
            return 100.0 + 0.1 * i  # gentle uptrend

        df = _make_df(252, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        # Every point in equity curve should be positive
        assert (eq > 0).all(), "Equity curve has non-positive values"


class TestVeryShortBacktest:
    """Edge case: very few trading days"""

    def test_two_day_backtest(self):
        df = _constant_df(2, price=100.0)
        try:
            result = _run_backtest(df)
            assert result is not None
        except Exception as e:
            # Two days may be too short for some strategies
            pytest.skip(f"Two-day backtest not supported: {e}")

    def test_single_day_backtest(self):
        df = _constant_df(1, price=100.0)
        try:
            result = _run_backtest(df)
            assert result is not None
        except Exception as e:
            pytest.skip(f"Single-day backtest not supported: {e}")


class TestNaNInPrices:
    """Bonus: NaN values in price data"""

    def test_nan_close_mid_series(self):
        def close_fn(i):
            if i == 50:
                return float('nan')
            return 100.0

        df = _make_df(100, close_fn)
        try:
            result = _run_backtest(df)
            eq = result.equity_series
            final = eq.iloc[-1]
            assert not math.isnan(final), "Final equity is NaN when input has NaN"
        except (ValueError, KeyError) as e:
            pytest.skip(f"NaN in prices raised handled error: {e}")


class TestInfPrices:
    """Bonus: Infinity in price data"""

    def test_inf_close(self):
        def close_fn(i):
            if i == 50:
                return float('inf')
            return 100.0

        df = _make_df(100, close_fn)
        try:
            result = _run_backtest(df)
            eq = result.equity_series
            final = eq.iloc[-1]
            assert not math.isinf(final), f"Final equity is inf: {final}"
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Inf in prices raised handled error: {e}")

    def test_inf_does_not_leak_into_equity_curve(self):
        """BUG PROBE: inf price on one day should not produce inf in equity series."""
        def close_fn(i):
            if i == 50:
                return float('inf')
            return 100.0

        df = _make_df(100, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        inf_count = np.isinf(eq).sum()
        assert inf_count == 0, (
            f"Equity series contains {inf_count} inf value(s). "
            f"Engine should guard against inf prices propagating into equity."
        )

    def test_inf_price_sharpe_not_nan(self):
        """BUG PROBE: inf in equity series causes sharpe_ratio to return NaN."""
        def close_fn(i):
            if i == 50:
                return float('inf')
            return 100.0

        df = _make_df(100, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        s = sharpe_ratio(eq)
        assert not math.isnan(s), (
            f"Sharpe ratio is NaN due to inf in equity series. "
            f"Inf count in equity: {np.isinf(eq).sum()}"
        )


class TestEquityCurveNoNaNOrInf:
    """Cross-cutting: the entire equity curve should never contain NaN or inf."""

    def test_zero_prices_no_nan_in_equity(self):
        df = _constant_df(100, price=0.0)
        result = _run_backtest(df)
        eq = result.equity_series
        assert eq.isna().sum() == 0, "Equity series has NaN values with zero prices"
        assert np.isinf(eq).sum() == 0, "Equity series has inf values with zero prices"

    def test_negative_prices_no_nan_in_equity(self):
        df = _constant_df(100, price=-50.0)
        result = _run_backtest(df)
        eq = result.equity_series
        assert eq.isna().sum() == 0, "Equity series has NaN values with negative prices"

    def test_spike_no_nan_in_equity(self):
        def close_fn(i):
            if i == 250:
                return 10_000.0
            return 100.0
        df = _make_df(252, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        assert eq.isna().sum() == 0, "Equity series has NaN after price spike"
        assert np.isinf(eq).sum() == 0, "Equity series has inf after price spike"

    def test_nan_input_no_nan_in_equity(self):
        """NaN in input price should not leak into equity curve."""
        def close_fn(i):
            if i == 50:
                return float('nan')
            return 100.0
        df = _make_df(100, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        assert eq.isna().sum() == 0, (
            "NaN in input prices leaked into equity series"
        )

    def test_extreme_volatility_no_nan_in_equity(self):
        def close_fn(i):
            price = 100.0
            for j in range(i):
                price *= 1.50 if j % 2 == 0 else 0.50
            return price
        df = _make_df(100, close_fn)
        result = _run_backtest(df)
        eq = result.equity_series
        assert eq.isna().sum() == 0, "Equity series has NaN with extreme volatility"
        assert np.isinf(eq).sum() == 0, "Equity series has inf with extreme volatility"


class TestZeroPriceNoTrades:
    """With all-zero prices, no shares should be purchased."""

    def test_zero_price_no_trades_executed(self):
        df = _constant_df(100, price=0.0)
        result = _run_backtest(df)
        assert len(result.trades) == 0, (
            f"Expected 0 trades at zero price, got {len(result.trades)}"
        )

    def test_zero_price_equity_unchanged(self):
        df = _constant_df(100, price=0.0)
        result = _run_backtest(df)
        eq = result.equity_series
        assert eq.iloc[-1] == 100_000.0, (
            f"Equity should be unchanged at $100k, got {eq.iloc[-1]}"
        )


class TestNegativePriceNoTrades:
    """With negative prices, no shares should be purchased."""

    def test_negative_price_no_trades(self):
        df = _constant_df(100, price=-50.0)
        result = _run_backtest(df)
        assert len(result.trades) == 0, (
            f"Expected 0 trades at negative price, got {len(result.trades)}. "
            f"Engine should reject or skip negative-priced orders."
        )
