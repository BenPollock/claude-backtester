"""
Black-box tests for extreme and degenerate configuration values.

These tests exercise the backtester via its Python API with pathological
config values to check for crashes, NaN/inf metrics, negative equity,
and nonsensical results.

Bugs found (historical):
  - Future start date (2099): DataManager forward-fills 252 real rows across
    ~19k business days, producing a huge equity curve of stale data instead of
    raising an error or returning empty.
  - Zero starting cash: max_drawdown metric returns NaN (division by zero in
    drawdown calculation when peak equity is always zero).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockDataSource(DataSource):
    """Serves pre-loaded DataFrames by symbol."""

    def __init__(self):
        self._data = {}

    def add(self, symbol, df):
        self._data[symbol] = df

    def fetch(self, symbol, start, end):
        df = self._data[symbol]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]


def make_normal_df(start="2020-01-02", days=252, start_price=100.0):
    """Generate deterministic synthetic OHLCV data."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    rng = np.random.default_rng(42)
    prices = np.cumsum(rng.normal(0.1, 1.0, days)) + start_price
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.full(days, 1_000_000),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


# Register a simple always-buy strategy using the current API
_ALWAYS_BUY_REGISTERED = False


def _register_always_buy():
    global _ALWAYS_BUY_REGISTERED
    if _ALWAYS_BUY_REGISTERED:
        return

    class AlwaysBuyStrategy(Strategy):
        name = "always_buy_cfg"

        def compute_indicators(self, df, timeframe_data=None):
            return df.copy()

        def generate_signals(self, symbol, row, position, portfolio_state,
                             benchmark_row=None):
            return SignalAction.BUY

        def size_order(self, symbol, action, row, portfolio_state, max_alloc_pct):
            return -1  # let engine decide

    _REGISTRY["always_buy_cfg"] = AlwaysBuyStrategy
    _ALWAYS_BUY_REGISTERED = True


def _make_config(tickers=None, benchmark="TEST", starting_cash=10000,
                 max_positions=10, max_alloc_pct=0.10, strategy_name="always_buy_cfg",
                 start_date=date(2020, 1, 2), end_date=date(2020, 12, 31),
                 **kwargs):
    if tickers is None:
        tickers = ["TEST"]
    return BacktestConfig(
        strategy_name=strategy_name,
        tickers=tickers,
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
        starting_cash=starting_cash,
        max_positions=max_positions,
        max_alloc_pct=max_alloc_pct,
        **kwargs,
    )


def _run(config, source):
    tmpdir = tempfile.mkdtemp()
    dm = DataManager(cache_dir=tmpdir, source=source)
    engine = BacktestEngine(config, dm)
    return engine.run()


def _make_source(tickers=None, start="2020-01-02", days=252):
    if tickers is None:
        tickers = ["TEST"]
    source = MockDataSource()
    df = make_normal_df(start=start, days=days)
    for t in tickers:
        source.add(t, df)
    return source


def _get_metrics(result):
    return compute_all_metrics(result.equity_series, result.trades)


def _check_metrics_finite(metrics, label=""):
    """Assert no NaN or inf in core metrics."""
    for key in ["total_return", "cagr", "sharpe_ratio", "max_drawdown"]:
        if key in metrics:
            val = metrics[key]
            assert not np.isnan(val), f"{key} is NaN ({label})"
            assert not np.isinf(val), f"{key} is inf ({label})"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def register_strategy():
    _register_always_buy()


@pytest.fixture
def normal_source():
    return _make_source()


# ---------------------------------------------------------------------------
# 1. Negative starting cash
#    Observed: No validation. Cash stays at -1000, no trades execute.
#    Equity curve has 252 points all at -1000.
#    Not a crash, but a missing validation — arguably should reject.
# ---------------------------------------------------------------------------
class TestNegativeStartingCash:
    def test_negative_cash_rejected(self, normal_source):
        """Negative starting cash is now rejected with ValueError."""
        config = _make_config(starting_cash=-1000)
        with pytest.raises(ValueError, match="starting_cash must be positive"):
            _run(config, normal_source)

    def test_zero_cash_rejected(self, normal_source):
        """Zero starting cash is now rejected with ValueError."""
        config = _make_config(starting_cash=0)
        with pytest.raises(ValueError, match="starting_cash must be positive"):
            _run(config, normal_source)


# ---------------------------------------------------------------------------
# 2. Enormous starting cash (1e18)
#    Observed: Runs fine, metrics are finite. Only 1 trade (10% of 1e18
#    buys a huge position). No overflow in float64.
# ---------------------------------------------------------------------------
class TestEnormousStartingCash:
    def test_huge_cash_metrics_finite(self, normal_source):
        """1e18 starting cash: no overflow, metrics are finite."""
        config = _make_config(starting_cash=1e18)
        result = _run(config, normal_source)
        metrics = _get_metrics(result)
        _check_metrics_finite(metrics, "1e18 cash")


# ---------------------------------------------------------------------------
# 3. max_alloc_pct = 2.0 (200% per position)
#    Observed: 1 trade executes. With $10k cash and 200% alloc, engine
#    sizes a $20k position. Cash goes negative? Actually cash=11341
#    after the run (position appreciated). No crash.
# ---------------------------------------------------------------------------
class TestOverAllocation:
    def test_200pct_allocation_runs(self, normal_source):
        """200% allocation: engine allows it, trades execute."""
        config = _make_config(max_alloc_pct=2.0)
        result = _run(config, normal_source)
        assert len(result.trades) >= 1
        metrics = _get_metrics(result)
        _check_metrics_finite(metrics, "200% alloc")


# ---------------------------------------------------------------------------
# 4. Negative allocation (-10%)
#    Observed: No trades. Position sizer computes negative shares -> 0.
#    Silently accepted, no validation error.
# ---------------------------------------------------------------------------
class TestNegativeAllocation:
    def test_negative_alloc_no_trades(self, normal_source):
        """Negative max_alloc_pct: no trades execute (zero shares sized)."""
        config = _make_config(max_alloc_pct=-0.10)
        result = _run(config, normal_source)
        assert len(result.trades) == 0
        assert result.portfolio.cash == 10000


# ---------------------------------------------------------------------------
# 5. Negative fee (fee rebate)
#    Observed: Runs fine. 1 trade. The negative fee effectively gives a rebate.
#    No validation. This could be intentional for maker rebates.
# ---------------------------------------------------------------------------
class TestNegativeFee:
    def test_negative_fee_accepted(self, normal_source):
        """Negative fee_per_trade: silently accepted, acts as rebate."""
        config = _make_config(fee_per_trade=-5.0)
        result = _run(config, normal_source)
        assert len(result.trades) >= 1


# ---------------------------------------------------------------------------
# 6. Negative slippage (-100 bps)
#    Observed: Runs fine. Negative slippage improves the fill price
#    (buys cheaper than open). No validation.
# ---------------------------------------------------------------------------
class TestNegativeSlippage:
    def test_negative_slippage_accepted(self, normal_source):
        """Negative slippage: silently accepted, improves fill price."""
        config = _make_config(slippage_bps=-100)
        result = _run(config, normal_source)
        assert len(result.trades) >= 1


# ---------------------------------------------------------------------------
# 7. Extreme slippage (100000 bps = 10000%)
#    Observed: 1 trade. Fill price is astronomical (100x open price).
#    Buys very few shares. Total return = -88%. Metrics are finite.
# ---------------------------------------------------------------------------
class TestExtremeSlippage:
    def test_10000pct_slippage_massive_loss(self, normal_source):
        """10000% slippage: huge fill price, massive loss, but no crash."""
        config = _make_config(slippage_bps=100_000)
        result = _run(config, normal_source)
        metrics = _get_metrics(result)
        _check_metrics_finite(metrics, "10000% slippage")
        # Should be a big loss
        assert metrics["total_return"] < -0.5, (
            f"Expected large loss with 10000% slippage, got {metrics['total_return']:.4f}"
        )


# ---------------------------------------------------------------------------
# 8. Start date far in the future (2099)
#    BUG: DataManager forward-fills 252 rows of real data across ~19k
#    business days in 2020-2099, producing a massive equity curve of
#    stale data. Should raise an error or return empty.
# ---------------------------------------------------------------------------
class TestFutureStartDate:
    def test_future_start_no_phantom_data(self, normal_source):
        """Future start date should not produce a huge backtest on stale data."""
        config = _make_config(
            start_date=date(2099, 1, 1),
            end_date=date(2099, 12, 31),
        )
        # Engine should either raise or produce a short result (not ~19k days)
        try:
            result = _run(config, normal_source)
            equity = result.equity_series
            assert len(equity) <= 252, (
                f"Got {len(equity)} equity points for a future date range. "
                f"DataManager should not produce phantom data."
            )
        except (RuntimeError, ValueError, IndexError):
            pass  # Raising an error is acceptable


# ---------------------------------------------------------------------------
# 9. end_date before start_date
#    Observed: Raises IndexError from empty data. Not a graceful
#    ValueError("end_date must be >= start_date"), but at least it fails.
# ---------------------------------------------------------------------------
class TestReversedDates:
    def test_end_before_start_raises(self, normal_source):
        """Reversed dates: raises an exception (IndexError, not ValueError)."""
        config = _make_config(
            start_date=date(2020, 12, 31),
            end_date=date(2020, 1, 2),
        )
        with pytest.raises((ValueError, RuntimeError, KeyError, IndexError)):
            _run(config, normal_source)


# ---------------------------------------------------------------------------
# 10. Ticker not in data source
#    Observed: KeyError from MockDataSource.fetch(). Engine does not
#    catch it gracefully.
# ---------------------------------------------------------------------------
class TestMissingTicker:
    def test_missing_ticker_raises(self, normal_source):
        """Missing ticker: raises KeyError."""
        config = _make_config(tickers=["DOESNOTEXIST"])
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            _run(config, normal_source)


# ---------------------------------------------------------------------------
# 11. Very short backtest: 2 trading days
#    Observed: Runs fine. 0 trades (signal on day 1, fill on day 2 which
#    is last day). Metrics are all finite.
# ---------------------------------------------------------------------------
class TestTwoTradingDays:
    def test_two_days_no_crash(self):
        """2 trading days: completes without error."""
        source = _make_source(start="2020-01-02", days=2)
        config = _make_config(
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 4),
        )
        result = _run(config, source)
        assert result is not None
        equity = result.equity_series
        assert len(equity) >= 1


# ---------------------------------------------------------------------------
# 12. Extreme fragmentation: 50 tickers, 0.1% alloc
#    Observed: Runs. With 0.1% of $10k = $10 per position at ~$100/share,
#    likely 0 shares per ticker. No crash.
# ---------------------------------------------------------------------------
class TestExtremeFragmentation:
    def test_fragmented_no_crash(self):
        """50 tickers at 0.1% allocation: no crash."""
        tickers = [f"T{i:04d}" for i in range(50)]
        source = _make_source(tickers=tickers)
        config = _make_config(
            tickers=tickers,
            benchmark="T0000",
            max_positions=1000,
            max_alloc_pct=0.001,
        )
        result = _run(config, source)
        assert result is not None


# ---------------------------------------------------------------------------
# 13. Wrong type in strategy params (sma_fast="hello")
#    Observed: Raises TypeError during compute_indicators (pandas rolling
#    with string window). Fails gracefully.
# ---------------------------------------------------------------------------
class TestWrongParamTypes:
    def test_string_sma_params_raises(self, normal_source):
        """Wrong param types: raises TypeError during indicator computation."""
        config = _make_config(
            strategy_name="sma_crossover",
            strategy_params={"sma_fast": "hello", "sma_slow": "world"},
        )
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _run(config, normal_source)


# ---------------------------------------------------------------------------
# 14. Empty tickers list
#    Observed: Engine raises RuntimeError("No data loaded for any ticker").
# ---------------------------------------------------------------------------
class TestEmptyTickers:
    def test_empty_tickers_raises(self, normal_source):
        """Empty tickers: raises RuntimeError."""
        config = _make_config(tickers=[])
        with pytest.raises((ValueError, RuntimeError)):
            _run(config, normal_source)


# ---------------------------------------------------------------------------
# 15. Zero starting cash
#    BUG: max_drawdown returns NaN. Division by zero when computing
#    drawdown from a peak of 0. Other metrics (cagr, total_return)
#    correctly return 0.0.
# ---------------------------------------------------------------------------
class TestZeroCash:
    def test_zero_cash_rejected(self, normal_source):
        """Zero starting cash is now rejected by engine validation."""
        config = _make_config(starting_cash=0)
        with pytest.raises(ValueError, match="starting_cash must be positive"):
            _run(config, normal_source)


# ---------------------------------------------------------------------------
# 16. max_positions = 0
#    Observed: No trades. Engine respects the limit.
# ---------------------------------------------------------------------------
class TestZeroMaxPositions:
    def test_zero_max_positions_no_trades(self, normal_source):
        """max_positions=0: no trades execute."""
        config = _make_config(max_positions=0)
        result = _run(config, normal_source)
        assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# 17. Same start and end date (single day)
#    Observed: Runs fine. 0 trades. Metrics are finite.
# ---------------------------------------------------------------------------
class TestSameStartEnd:
    def test_same_date_no_crash(self, normal_source):
        """Single-day backtest: completes without error."""
        config = _make_config(
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 2),
        )
        result = _run(config, normal_source)
        assert result is not None
        assert len(result.trades) == 0
