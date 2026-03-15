"""
Black-box tests for data integrity and timing issues.
Tests gaps, misalignment, edge cases in data without reading source code.
"""

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
from backtester.analytics.metrics import compute_all_metrics

discover_strategies()

from backtester.data.sources.base import DataSource


class MockDataSource(DataSource):
    """Data source serving pre-built DataFrames."""

    def __init__(self):
        self._data = {}

    def add(self, symbol, df):
        self._data[symbol] = df

    def fetch(self, symbol, start, end):
        df = self._data[symbol]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]


def make_df(start="2020-01-02", days=252, start_price=100.0, daily_ret=0.001):
    """Create synthetic OHLCV DataFrame with steady upward drift."""
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


def _make_config(tickers, benchmark="SPY", start="2020-01-01", end="2020-12-31",
                 cash=100_000, strategy="sma_crossover", params=None, **kwargs):
    """Helper to build a BacktestConfig."""
    return BacktestConfig(
        tickers=tickers,
        benchmark=benchmark,
        start_date=date.fromisoformat(start),
        end_date=date.fromisoformat(end),
        starting_cash=cash,
        strategy_name=strategy,
        strategy_params=params or {"sma_fast": 5, "sma_slow": 10},
        max_positions=10,
        max_alloc_pct=0.5,
        **kwargs,
    )


def _run_backtest(config, source):
    """Run backtest with mock data source and return result."""
    tmpdir = tempfile.mkdtemp()
    dm = DataManager(cache_dir=tmpdir, source=source)
    engine = BacktestEngine(config, data_manager=dm)
    return engine.run()


def _equity_values(result):
    """Extract equity values from result as a list of floats."""
    series = result.equity_series
    return series.values.tolist()


def _get_metrics(result):
    """Compute metrics from a backtest result."""
    return compute_all_metrics(
        result.equity_series,
        result.trades,
        benchmark_series=result.benchmark_series,
    )


# ---------------------------------------------------------------------------
# Always-buy strategy for deterministic signal generation
# ---------------------------------------------------------------------------

class AlwaysBuyStrategy(Strategy):
    """Buys every ticker on every bar if not already held."""

    name = "always_buy_test"

    def compute_indicators(self, df, params=None, **kwargs):
        df = df.copy()
        return df

    def generate_signals(self, symbol, row, position, portfolio_state,
                         benchmark_row=None):
        return SignalAction.BUY

    def size_order(self, signal, row, indicators, portfolio_state, params, config):
        return -1  # let engine decide


# Register it if not already present
if "always_buy_test" not in _REGISTRY:
    _REGISTRY["always_buy_test"] = AlwaysBuyStrategy


# ---------------------------------------------------------------------------
# Scenario 1: Large gap in data (missing Apr-Jun)
# ---------------------------------------------------------------------------

class TestLargeGapInData:
    """Stock has data Jan-Mar and Jul-Dec, missing Apr-Jun entirely."""

    def _make_gapped_source(self):
        jan_mar = make_df(start="2020-01-02", days=63, start_price=100.0)
        jul_dec = make_df(start="2020-07-01", days=126, start_price=110.0)
        gapped = pd.concat([jan_mar, jul_dec])
        bench = make_df(start="2020-01-02", days=252)
        src = MockDataSource()
        src.add("GAPPY", gapped)
        src.add("SPY", bench)
        return src

    def test_backtest_survives_gap(self):
        src = self._make_gapped_source()
        config = _make_config(tickers=["GAPPY"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        assert result is not None
        equity = _equity_values(result)
        assert len(equity) > 0
        assert np.isfinite(equity[-1]), f"Final equity is not finite: {equity[-1]}"

    def test_gap_no_nan_in_equity(self):
        src = self._make_gapped_source()
        config = _make_config(tickers=["GAPPY"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        equity = _equity_values(result)
        nan_count = sum(1 for v in equity if not np.isfinite(v))
        assert nan_count == 0, f"Equity curve has {nan_count} NaN/inf values"


# ---------------------------------------------------------------------------
# Scenario 2: Two tickers with different date ranges
# ---------------------------------------------------------------------------

class TestDifferentDateRanges:
    """Ticker A has full year, Ticker B only Jun-Dec."""

    def _make_source(self):
        full_year = make_df(start="2020-01-02", days=252, start_price=100.0)
        half_year = make_df(start="2020-06-01", days=126, start_price=50.0)
        bench = make_df(start="2020-01-02", days=252)
        src = MockDataSource()
        src.add("FULL", full_year)
        src.add("HALF", half_year)
        src.add("SPY", bench)
        return src

    def test_mismatched_ranges_no_crash(self):
        src = self._make_source()
        config = _make_config(tickers=["FULL", "HALF"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        assert result is not None
        equity = _equity_values(result)
        assert len(equity) > 0

    def test_both_tickers_traded(self):
        src = self._make_source()
        config = _make_config(tickers=["FULL", "HALF"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        assert len(result.trades) > 0, "No trades were generated at all"


# ---------------------------------------------------------------------------
# Scenario 3: Ticker data starts AFTER backtest start_date
# ---------------------------------------------------------------------------

class TestDataStartsLate:
    """Backtest starts Jan 1, ticker data begins March 1."""

    def _make_source(self):
        late_data = make_df(start="2020-03-01", days=200, start_price=100.0)
        bench = make_df(start="2020-01-02", days=252)
        src = MockDataSource()
        src.add("LATE", late_data)
        src.add("SPY", bench)
        return src

    def test_late_start_no_crash(self):
        src = self._make_source()
        config = _make_config(tickers=["LATE"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        assert result is not None
        equity = _equity_values(result)
        assert len(equity) > 0

    def test_late_start_initial_equity_preserved(self):
        src = self._make_source()
        config = _make_config(tickers=["LATE"], strategy="always_buy_test",
                              params={}, cash=100_000,
                              start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        equity = _equity_values(result)
        first_equity = equity[0]
        # Before any data arrives, equity should be initial cash
        assert first_equity == pytest.approx(100_000, rel=0.01), \
            f"First equity {first_equity} != initial cash 100000"


# ---------------------------------------------------------------------------
# Scenario 4: Ticker data ends BEFORE backtest end_date
# ---------------------------------------------------------------------------

class TestDataEndsEarly:
    """Backtest ends Dec 31, ticker data ends around May."""

    def _make_source(self, ticker_days=100):
        early_data = make_df(start="2020-01-02", days=ticker_days, start_price=100.0)
        bench = make_df(start="2020-01-02", days=252)
        src = MockDataSource()
        src.add("EARLY", early_data)
        src.add("SPY", bench)
        return src

    def test_early_end_no_crash(self):
        src = self._make_source(200)
        config = _make_config(tickers=["EARLY"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        assert result is not None

    def test_early_end_equity_no_nan(self):
        """After data ends, equity should not contain NaN/inf."""
        src = self._make_source(100)
        config = _make_config(tickers=["EARLY"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        equity = _equity_values(result)
        assert len(equity) > 0
        for v in equity:
            assert np.isfinite(v), f"Non-finite equity value: {v}"


# ---------------------------------------------------------------------------
# Scenario 5: Volume = 0 for all days (with volume slippage)
# ---------------------------------------------------------------------------

class TestZeroVolume:
    """Zero volume with volume-based slippage model."""

    def _make_source(self):
        df = make_df(start="2020-01-02", days=252, start_price=100.0)
        df["Volume"] = 0
        bench = make_df(start="2020-01-02", days=252)
        src = MockDataSource()
        src.add("NOVOL", df)
        src.add("SPY", bench)
        return src

    def test_zero_volume_no_crash(self):
        src = self._make_source()
        config = _make_config(tickers=["NOVOL"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31",
                              slippage_bps=10.0)
        result = _run_backtest(config, src)

        assert result is not None
        equity = _equity_values(result)
        assert len(equity) > 0

    def test_zero_volume_metrics_finite(self):
        src = self._make_source()
        config = _make_config(tickers=["NOVOL"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        metrics = _get_metrics(result)
        # FINDING: Multiple ratio metrics return inf when equity is monotonically
        # increasing. Division-by-zero in denominators:
        #   sortino_ratio: zero downside deviation
        #   calmar_ratio: zero max drawdown
        #   profit_factor: zero losing trade PnL
        #   payoff_ratio: zero average loss
        #   omega_ratio: zero downside returns
        KNOWN_INF_METRICS = {
            "sortino_ratio", "calmar_ratio", "profit_factor",
            "payoff_ratio", "omega_ratio",
        }
        non_finite = {k: v for k, v in metrics.items()
                      if isinstance(v, float) and not np.isfinite(v)}
        unexpected = {k: v for k, v in non_finite.items()
                      if k not in KNOWN_INF_METRICS}
        assert not unexpected, f"Unexpected non-finite metrics: {unexpected}"

    def test_zero_volume_inf_ratio_metrics_clamped(self):
        """Ratio metrics with zero denominators are clamped to 99999.0, not inf."""
        src = self._make_source()
        config = _make_config(tickers=["NOVOL"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        metrics = _get_metrics(result)
        for key in ["sortino_ratio", "calmar_ratio", "profit_factor",
                     "payoff_ratio", "omega_ratio"]:
            if key in metrics:
                assert metrics[key] != float("inf"), \
                    f"{key} should be clamped, not inf"
                assert isinstance(metrics[key], (int, float))




# ---------------------------------------------------------------------------
# Scenario 8: Timezone-aware DatetimeIndex
# ---------------------------------------------------------------------------

class TestTimezoneAwareIndex:
    """DatetimeIndex with tz='US/Eastern'."""

    def test_tz_aware_no_crash(self):
        dates = pd.bdate_range(start="2020-01-02", periods=252, freq="B",
                               tz="US/Eastern")
        prices = np.linspace(100, 130, 252)
        df = pd.DataFrame(
            {
                "Open": prices * 0.999,
                "High": prices * 1.005,
                "Low": prices * 0.995,
                "Close": prices,
                "Volume": np.full(252, 1_000_000),
            },
            index=dates,
        )
        df.index.name = "Date"
        bench = make_df(start="2020-01-02", days=252)

        src = MockDataSource()
        src.add("TZAWARE", df)
        src.add("SPY", bench)

        config = _make_config(tickers=["TZAWARE"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")

        # FINDING: TZ-aware DatetimeIndex causes data load failure.
        # DataManager silently fails to load the ticker because Timestamp
        # comparison with tz-aware index raises TypeError.
        # Result: RuntimeError('No data loaded for any ticker').
        with pytest.raises(RuntimeError, match="No data loaded"):
            _run_backtest(config, src)




# ---------------------------------------------------------------------------
# Scenario 10: Two tickers with identical data
# ---------------------------------------------------------------------------

class TestIdenticalTickers:
    """Two tickers with exactly the same prices and dates."""

    def _make_source(self):
        df_a = make_df(start="2020-01-02", days=252, start_price=100.0, daily_ret=0.001)
        df_b = make_df(start="2020-01-02", days=252, start_price=100.0, daily_ret=0.001)
        bench = make_df(start="2020-01-02", days=252)
        src = MockDataSource()
        src.add("AAA", df_a)
        src.add("BBB", df_b)
        src.add("SPY", bench)
        return src

    def test_identical_tickers_equity_tracks(self):
        src = self._make_source()
        config = _make_config(tickers=["AAA", "BBB"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31",
                              cash=100_000)
        result = _run_backtest(config, src)

        assert result is not None
        equity = _equity_values(result)
        final_eq = equity[-1]
        assert np.isfinite(final_eq)
        # With identical upward-drifting data and always buy, should be > initial cash
        assert final_eq > 100_000, f"Expected gain, got {final_eq}"

    def test_identical_tickers_both_generate_trades(self):
        src = self._make_source()
        config = _make_config(tickers=["AAA", "BBB"], strategy="always_buy_test",
                              params={}, start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        traded_symbols = {t.symbol for t in result.trades}
        assert "AAA" in traded_symbols or "BBB" in traded_symbols, \
            f"Neither ticker traded. Trades: {result.trades}"


# ---------------------------------------------------------------------------
# Scenario 11: Very long backtest (20 years of data)
# ---------------------------------------------------------------------------

class TestLongBacktest:
    """5000+ trading days over 20 years."""

    def _make_source(self):
        long_data = make_df(start="2000-01-03", days=5040, start_price=50.0,
                            daily_ret=0.0003)
        bench = make_df(start="2000-01-03", days=5040, start_price=100.0,
                        daily_ret=0.0003)
        src = MockDataSource()
        src.add("LONG", long_data)
        src.add("SPY", bench)
        return src

    def test_20_years_no_crash(self):
        src = self._make_source()
        config = _make_config(tickers=["LONG"], strategy="always_buy_test",
                              params={}, start="2000-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        assert result is not None
        equity = _equity_values(result)
        assert len(equity) > 4000, \
            f"Expected 4000+ equity points, got {len(equity)}"

    def test_20_years_metrics_finite(self):
        src = self._make_source()
        config = _make_config(tickers=["LONG"], strategy="always_buy_test",
                              params={}, start="2000-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        metrics = _get_metrics(result)
        # FINDING: Same inf ratio metrics as Scenario 5 (zero-volume)
        KNOWN_INF_METRICS = {
            "sortino_ratio", "calmar_ratio", "profit_factor",
            "payoff_ratio", "omega_ratio",
        }
        non_finite = {k: v for k, v in metrics.items()
                      if isinstance(v, float) and not np.isfinite(v)}
        unexpected = {k: v for k, v in non_finite.items()
                      if k not in KNOWN_INF_METRICS}
        assert not unexpected, f"Unexpected non-finite metrics: {unexpected}"


# ---------------------------------------------------------------------------
# Scenario 12: Benchmark ticker not in tickers list
# ---------------------------------------------------------------------------

class TestBenchmarkNotInTickers:
    """benchmark='SPY' but tickers=['AAPL']. Benchmark should still load."""

    def _make_source(self):
        aapl_data = make_df(start="2020-01-02", days=252, start_price=300.0,
                            daily_ret=0.002)
        spy_data = make_df(start="2020-01-02", days=252, start_price=330.0,
                           daily_ret=0.001)
        src = MockDataSource()
        src.add("AAPL", aapl_data)
        src.add("SPY", spy_data)
        return src

    def test_benchmark_separate_from_tickers(self):
        src = self._make_source()
        config = _make_config(tickers=["AAPL"], benchmark="SPY",
                              strategy="always_buy_test", params={},
                              start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        assert result is not None
        equity = _equity_values(result)
        assert len(equity) > 0

    def test_benchmark_not_traded(self):
        """SPY should be used for regime filter only, not traded."""
        src = self._make_source()
        config = _make_config(tickers=["AAPL"], benchmark="SPY",
                              strategy="always_buy_test", params={},
                              start="2020-01-01", end="2020-12-31")
        result = _run_backtest(config, src)

        traded_symbols = {t.symbol for t in result.trades}
        assert "SPY" not in traded_symbols, \
            "Benchmark SPY should not appear in trades when not in tickers list"


