"""Tests for regime classification and performance breakdown."""

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.regime import (
    classify_market_regime,
    classify_volatility_regime,
    regime_performance,
    regime_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n, start="2020-01-02"):
    """Generate n business-day dates as a DatetimeIndex of date objects."""
    dates = pd.bdate_range(start=start, periods=n)
    return pd.DatetimeIndex(dates.date, name="Date")


def _make_equity(values, start="2020-01-02"):
    """Build an equity Series from a list of values."""
    idx = _make_dates(len(values), start)
    return pd.Series(values, index=idx, name="Equity")


def _make_prices(values, start="2020-01-02"):
    """Build a price Series (benchmark Close) from a list of values."""
    idx = _make_dates(len(values), start)
    return pd.Series(values, index=idx, name="Close")


# ---------------------------------------------------------------------------
# classify_market_regime
# ---------------------------------------------------------------------------

class TestClassifyMarketRegime:
    def test_bull_market(self):
        """Price always well above its SMA -> all valid days are 'bull'."""
        # Start at 100 and rise steadily; SMA will lag behind price.
        n = 250
        prices = _make_prices(np.linspace(100, 200, n))
        labels = classify_market_regime(prices, sma_window=50)

        valid = labels[labels != "unknown"]
        assert len(valid) > 0
        # All valid days should be bull (price is far above a lagging SMA)
        assert (valid == "bull").all()

    def test_bear_market(self):
        """Price always well below its SMA -> all valid days are 'bear'."""
        n = 250
        prices = _make_prices(np.linspace(200, 100, n))
        labels = classify_market_regime(prices, sma_window=50)

        valid = labels[labels != "unknown"]
        assert len(valid) > 0
        assert (valid == "bear").all()

    def test_mixed_market(self):
        """Price crosses SMA -> produces both bull and bear labels."""
        n = 300
        # Oscillate around a flat mean to cross the SMA
        t = np.linspace(0, 6 * np.pi, n)
        prices = _make_prices(100 + 20 * np.sin(t))
        labels = classify_market_regime(prices, sma_window=50, sideways_band=0.02)

        valid = labels[labels != "unknown"]
        regimes = valid.unique()
        assert "bull" in regimes
        assert "bear" in regimes

    def test_sideways_detection(self):
        """Price hugging the SMA within the band -> 'sideways'."""
        n = 300
        # Flat price series => SMA equals price => within ±2% band
        prices = _make_prices(np.full(n, 100.0))
        labels = classify_market_regime(prices, sma_window=50, sideways_band=0.02)

        valid = labels[labels != "unknown"]
        assert len(valid) > 0
        assert (valid == "sideways").all()

    def test_first_n_days_unknown(self):
        """First sma_window-1 days should be labeled 'unknown'.

        pandas rolling(window=N, min_periods=N) produces its first valid
        value at index N-1, so indices 0..N-2 are unknown.
        """
        window = 50
        n = 100
        prices = _make_prices(np.linspace(100, 150, n))
        labels = classify_market_regime(prices, sma_window=window)

        # First window-1 days are unknown (rolling needs 'window' points,
        # so the first valid SMA is at position window-1)
        assert (labels.iloc[: window - 1] == "unknown").all()
        # From position window-1 onward, labels should be valid
        assert (labels.iloc[window - 1 :] != "unknown").all()

    def test_empty_series(self):
        """Empty input returns empty output."""
        prices = pd.Series(dtype=float)
        labels = classify_market_regime(prices)
        assert labels.empty

    def test_short_series(self):
        """Series shorter than sma_window -> all 'unknown'."""
        prices = _make_prices([100, 101, 102])
        labels = classify_market_regime(prices, sma_window=50)
        assert (labels == "unknown").all()


# ---------------------------------------------------------------------------
# classify_volatility_regime
# ---------------------------------------------------------------------------

class TestClassifyVolatilityRegime:
    def test_three_regimes_present(self):
        """A long enough series with varying vol should produce all three regimes."""
        rng = np.random.default_rng(42)
        n = 500
        # Low vol first, then high vol, then medium
        returns_low = rng.normal(0, 0.005, 150)
        returns_high = rng.normal(0, 0.04, 150)
        returns_med = rng.normal(0, 0.015, 200)
        all_returns = np.concatenate([returns_low, returns_high, returns_med])

        idx = _make_dates(n)
        ret_series = pd.Series(all_returns, index=idx)
        labels = classify_volatility_regime(ret_series, window=63)

        valid = labels[labels != "unknown"]
        assert "low_vol" in valid.values
        assert "medium_vol" in valid.values
        assert "high_vol" in valid.values

    def test_warmup_unknown(self):
        """First window-1 days should be 'unknown'.

        rolling(window=N, min_periods=N).std() first produces a valid
        value at index N-1, so indices 0..N-2 are unknown.
        """
        window = 63
        rng = np.random.default_rng(7)
        ret_series = pd.Series(
            rng.normal(0, 0.01, 200), index=_make_dates(200)
        )
        labels = classify_volatility_regime(ret_series, window=window)
        assert (labels.iloc[: window - 1] == "unknown").all()
        # Position window-1 onward should have a valid label
        assert labels.iloc[window - 1] != "unknown"

    def test_empty_series(self):
        """Empty input returns empty output."""
        labels = classify_volatility_regime(pd.Series(dtype=float))
        assert labels.empty

    def test_short_series(self):
        """Series shorter than window -> all 'unknown'."""
        ret_series = pd.Series([0.01, -0.01, 0.005], index=_make_dates(3))
        labels = classify_volatility_regime(ret_series, window=63)
        assert (labels == "unknown").all()

    def test_constant_returns(self):
        """Constant returns (zero vol) -> all valid days are 'low_vol'."""
        n = 200
        ret_series = pd.Series(np.full(n, 0.001), index=_make_dates(n))
        labels = classify_volatility_regime(ret_series, window=63)
        valid = labels[labels != "unknown"]
        # With constant returns, std=0 for all windows, all at the lowest percentile
        # They should all be classified the same (low_vol since 0 <= any threshold)
        assert len(valid) > 0
        assert (valid == "low_vol").all()


# ---------------------------------------------------------------------------
# regime_performance
# ---------------------------------------------------------------------------

class TestRegimePerformance:
    def test_known_equity_two_regimes(self):
        """Split equity into two regimes and verify metrics are reasonable."""
        n = 200
        values = np.linspace(100, 150, n)
        equity = _make_equity(values)

        # First half 'bull', second half 'bear' (labels are arbitrary here)
        labels = pd.Series("bull", index=equity.index)
        labels.iloc[n // 2:] = "bear"

        df = regime_performance(equity, labels)

        assert "bull" in df.index
        assert "bear" in df.index
        assert "total_return" in df.columns
        assert "annualized_return" in df.columns
        assert "sharpe_ratio" in df.columns
        assert "max_drawdown" in df.columns
        assert "trading_days" in df.columns
        assert "pct_of_time" in df.columns

        # Both regimes had rising prices, so total_return should be positive
        assert df.loc["bull", "total_return"] > 0
        assert df.loc["bear", "total_return"] > 0

        # Trading days should sum to total
        assert df["trading_days"].sum() == n

        # pct_of_time should sum to ~1
        assert abs(df["pct_of_time"].sum() - 1.0) < 0.01

    def test_single_regime(self):
        """All days same label -> one-row result covering all days."""
        n = 100
        equity = _make_equity(np.linspace(100, 120, n))
        labels = pd.Series("bull", index=equity.index)

        df = regime_performance(equity, labels)
        assert len(df) == 1
        assert df.loc["bull", "trading_days"] == n
        assert df.loc["bull", "pct_of_time"] == pytest.approx(1.0)
        assert df.loc["bull", "total_return"] > 0

    def test_max_drawdown_in_regime(self):
        """Equity with a drawdown in one regime -> negative max_drawdown."""
        # Bull: rising. Bear: drop then partial recovery.
        bull_vals = np.linspace(100, 120, 50)
        bear_vals = np.concatenate([np.linspace(120, 100, 25), np.linspace(100, 110, 25)])
        values = np.concatenate([bull_vals, bear_vals])
        equity = _make_equity(values)

        labels = pd.Series("bull", index=equity.index)
        labels.iloc[50:] = "bear"

        df = regime_performance(equity, labels)

        # Bull regime: monotonically rising, no drawdown
        assert df.loc["bull", "max_drawdown"] == pytest.approx(0.0, abs=1e-10)
        # Bear regime: has a drawdown (price drops)
        assert df.loc["bear", "max_drawdown"] < 0

    def test_empty_equity(self):
        """Empty equity returns empty DataFrame."""
        df = regime_performance(pd.Series(dtype=float), pd.Series(dtype=str))
        assert df.empty

    def test_single_day(self):
        """Single-day equity -> not enough data, returns empty."""
        equity = _make_equity([100])
        labels = pd.Series("bull", index=equity.index)
        df = regime_performance(equity, labels)
        assert df.empty

    def test_annualized_volatility_positive(self):
        """Noisy equity should produce positive annualized vol."""
        rng = np.random.default_rng(42)
        n = 252
        vals = [100.0]
        for _ in range(n - 1):
            vals.append(vals[-1] * (1 + rng.normal(0.0003, 0.01)))
        equity = _make_equity(vals)
        labels = pd.Series("all", index=equity.index)

        df = regime_performance(equity, labels)
        assert df.loc["all", "annualized_volatility"] > 0


# ---------------------------------------------------------------------------
# regime_summary (integration)
# ---------------------------------------------------------------------------

class TestRegimeSummary:
    def test_end_to_end(self):
        """Full pipeline: equity + benchmark -> market and vol regime perf."""
        n = 400
        rng = np.random.default_rng(99)

        # Benchmark: trending up
        bm_vals = [100.0]
        for _ in range(n - 1):
            bm_vals.append(bm_vals[-1] * (1 + rng.normal(0.0003, 0.01)))
        benchmark = _make_prices(bm_vals)

        # Equity: similar trajectory with noise
        eq_vals = [100.0]
        for _ in range(n - 1):
            eq_vals.append(eq_vals[-1] * (1 + rng.normal(0.0005, 0.012)))
        equity = _make_equity(eq_vals)

        result = regime_summary(
            equity, benchmark, sma_window=50, vol_window=63
        )

        assert "market_regime_perf" in result
        assert "vol_regime_perf" in result
        assert "market_labels" in result
        assert "vol_labels" in result

        # market_labels should be indexed like benchmark
        assert len(result["market_labels"]) == len(benchmark)
        # vol_labels should be indexed like the returns (n-1 since pct_change drops first)
        assert len(result["vol_labels"]) == n - 1

        # Both perf DataFrames should have rows
        assert not result["market_regime_perf"].empty
        assert not result["vol_regime_perf"].empty

    def test_explicit_returns(self):
        """Passing pre-computed returns should work the same way."""
        n = 300
        rng = np.random.default_rng(11)

        bm_vals = np.linspace(100, 150, n)
        benchmark = _make_prices(bm_vals)

        eq_vals = [100.0]
        for _ in range(n - 1):
            eq_vals.append(eq_vals[-1] * (1 + rng.normal(0.0003, 0.01)))
        equity = _make_equity(eq_vals)
        returns = equity.pct_change().dropna()

        result = regime_summary(
            equity, benchmark, returns=returns, sma_window=50, vol_window=63
        )
        assert not result["market_regime_perf"].empty

    def test_short_data(self):
        """Very short data -> regime perf DataFrames may be empty but no crash."""
        benchmark = _make_prices([100, 101, 102])
        equity = _make_equity([1000, 1010, 1020])

        result = regime_summary(equity, benchmark, sma_window=200, vol_window=63)

        # Labels should exist (all 'unknown')
        assert len(result["market_labels"]) == 3
        assert (result["market_labels"] == "unknown").all()
