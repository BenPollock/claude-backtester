"""Tests for correlation analysis, portfolio concentration, and sector exposure."""

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.correlation import (
    compute_correlation_matrix,
    compute_rolling_correlation,
    compute_hhi,
    compute_portfolio_concentration,
    compute_sector_exposure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(close_values, start="2020-01-02"):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    dates = pd.bdate_range(start=start, periods=len(close_values))
    close = pd.Series(close_values, index=dates, dtype=float)
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": 1_000_000,
    })


def _make_price_data_from_returns(returns_dict, start="2020-01-02"):
    """Build price_data dict from ticker -> daily return arrays.

    Each return array is used to compound an initial price of 100.
    """
    price_data = {}
    for ticker, rets in returns_dict.items():
        prices = [100.0]
        for r in rets:
            prices.append(prices[-1] * (1 + r))
        price_data[ticker] = _make_ohlcv(prices, start=start)
    return price_data


# ===========================================================================
# compute_correlation_matrix
# ===========================================================================

class TestCorrelationMatrix:
    def test_perfect_positive_correlation(self):
        """Identical price series should have correlation 1.0."""
        values = np.linspace(100, 150, 100)
        price_data = {
            "A": _make_ohlcv(values),
            "B": _make_ohlcv(values),
        }
        corr = compute_correlation_matrix(price_data)
        assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-10)
        assert corr.loc["A", "A"] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative_correlation(self):
        """Perfectly opposite daily returns should have correlation -1.0."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.02, 200)
        price_data = _make_price_data_from_returns({
            "A": rets,
            "B": -rets,
        })
        corr = compute_correlation_matrix(price_data)
        assert corr.loc["A", "B"] == pytest.approx(-1.0, abs=1e-6)

    def test_uncorrelated_series(self):
        """Independent random series should have near-zero correlation."""
        rng = np.random.default_rng(123)
        n = 5000
        price_data = _make_price_data_from_returns({
            "A": rng.normal(0, 0.01, n),
            "B": rng.normal(0, 0.01, n),
        })
        corr = compute_correlation_matrix(price_data)
        # With 5000 points, correlation should be close to zero
        assert abs(corr.loc["A", "B"]) < 0.05

    def test_single_ticker(self):
        """Single ticker produces a 1x1 matrix with correlation 1.0."""
        price_data = {"SPY": _make_ohlcv(np.linspace(100, 120, 50))}
        corr = compute_correlation_matrix(price_data)
        assert corr.shape == (1, 1)
        assert corr.loc["SPY", "SPY"] == pytest.approx(1.0)

    def test_tickers_subset(self):
        """Passing explicit tickers filters the matrix."""
        values = np.linspace(100, 120, 50)
        price_data = {
            "A": _make_ohlcv(values),
            "B": _make_ohlcv(values * 1.1),
            "C": _make_ohlcv(values * 0.9),
        }
        corr = compute_correlation_matrix(price_data, tickers=["A", "B"])
        assert list(corr.columns) == ["A", "B"]
        assert "C" not in corr.columns

    def test_symmetry(self):
        """Correlation matrix must be symmetric."""
        rng = np.random.default_rng(7)
        price_data = _make_price_data_from_returns({
            "X": rng.normal(0.001, 0.02, 200),
            "Y": rng.normal(0.0005, 0.015, 200),
            "Z": rng.normal(-0.0005, 0.025, 200),
        })
        corr = compute_correlation_matrix(price_data)
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_empty_price_data(self):
        """Empty input returns empty DataFrame."""
        corr = compute_correlation_matrix({})
        assert corr.empty


# ===========================================================================
# compute_rolling_correlation
# ===========================================================================

class TestRollingCorrelation:
    def test_basic_calculation(self):
        """Rolling correlation returns a Series with valid values."""
        rng = np.random.default_rng(42)
        n = 200
        shared = rng.normal(0, 0.01, n)
        noise_a = rng.normal(0, 0.005, n)
        noise_b = rng.normal(0, 0.005, n)
        price_data = _make_price_data_from_returns({
            "A": shared + noise_a,
            "B": shared + noise_b,
        })
        rolling = compute_rolling_correlation(price_data, "A", "B", window=30)
        assert isinstance(rolling, pd.Series)
        assert len(rolling) > 0
        # Shared component means correlation should be positive on average
        assert rolling.mean() > 0

    def test_window_size_affects_length(self):
        """Larger window produces fewer valid observations."""
        rng = np.random.default_rng(99)
        n = 200
        price_data = _make_price_data_from_returns({
            "A": rng.normal(0, 0.01, n),
            "B": rng.normal(0, 0.01, n),
        })
        short_window = compute_rolling_correlation(price_data, "A", "B", window=20)
        long_window = compute_rolling_correlation(price_data, "A", "B", window=63)
        assert len(short_window) > len(long_window)

    def test_identical_series_rolling(self):
        """Rolling correlation of identical series should be 1.0 everywhere."""
        values = np.linspace(100, 200, 150)
        price_data = {
            "A": _make_ohlcv(values),
            "B": _make_ohlcv(values),
        }
        rolling = compute_rolling_correlation(price_data, "A", "B", window=20)
        # All values should be exactly 1.0 (or very close)
        assert all(abs(v - 1.0) < 1e-10 for v in rolling.values)


# ===========================================================================
# compute_hhi
# ===========================================================================

class TestHHI:
    def test_equal_weights(self):
        """N equal-weighted positions -> HHI = 1/N."""
        n = 4
        weights = {f"T{i}": 1.0 / n for i in range(n)}
        assert compute_hhi(weights) == pytest.approx(1.0 / n)

    def test_fully_concentrated(self):
        """Single position with weight 1.0 -> HHI = 1.0."""
        assert compute_hhi({"SPY": 1.0}) == pytest.approx(1.0)

    def test_realistic_portfolio(self):
        """Realistic unequal weights produce HHI between 1/N and 1.0."""
        weights = {"AAPL": 0.40, "MSFT": 0.30, "GOOG": 0.20, "AMZN": 0.10}
        hhi = compute_hhi(weights)
        # 0.16 + 0.09 + 0.04 + 0.01 = 0.30
        assert hhi == pytest.approx(0.30)
        assert 1.0 / 4 < hhi < 1.0

    def test_two_positions_60_40(self):
        """60/40 split: HHI = 0.36 + 0.16 = 0.52."""
        assert compute_hhi({"A": 0.60, "B": 0.40}) == pytest.approx(0.52)

    def test_empty_weights(self):
        """Empty weights returns 0.0."""
        assert compute_hhi({}) == 0.0


# ===========================================================================
# compute_portfolio_concentration
# ===========================================================================

class TestPortfolioConcentration:
    def test_multiple_positions(self):
        """Standard multi-position portfolio returns correct metrics."""
        positions = {"AAPL": 5000.0, "MSFT": 3000.0, "GOOG": 2000.0}
        result = compute_portfolio_concentration(positions)

        assert result["weights"]["AAPL"] == pytest.approx(0.50)
        assert result["weights"]["MSFT"] == pytest.approx(0.30)
        assert result["weights"]["GOOG"] == pytest.approx(0.20)

        # HHI = 0.25 + 0.09 + 0.04 = 0.38
        assert result["hhi"] == pytest.approx(0.38)
        assert result["effective_n"] == pytest.approx(1.0 / 0.38)
        assert result["max_weight"] == pytest.approx(0.50)
        assert result["max_weight_ticker"] == "AAPL"

    def test_single_position(self):
        """Single position is fully concentrated."""
        result = compute_portfolio_concentration({"SPY": 10000.0})
        assert result["hhi"] == pytest.approx(1.0)
        assert result["effective_n"] == pytest.approx(1.0)
        assert result["max_weight"] == pytest.approx(1.0)
        assert result["max_weight_ticker"] == "SPY"

    def test_equal_positions(self):
        """Equal dollar positions produce equal weights."""
        positions = {"A": 2500.0, "B": 2500.0, "C": 2500.0, "D": 2500.0}
        result = compute_portfolio_concentration(positions)
        assert result["hhi"] == pytest.approx(0.25)
        assert result["effective_n"] == pytest.approx(4.0)
        for w in result["weights"].values():
            assert w == pytest.approx(0.25)

    def test_empty_positions(self):
        """Empty positions dict returns zero metrics."""
        result = compute_portfolio_concentration({})
        assert result["hhi"] == 0.0
        assert result["effective_n"] == 0.0
        assert result["max_weight"] == 0.0
        assert result["max_weight_ticker"] == ""
        assert result["weights"] == {}

    def test_zero_total_value(self):
        """All zero-value positions handled gracefully."""
        result = compute_portfolio_concentration({"A": 0.0, "B": 0.0})
        assert result["hhi"] == 0.0
        assert result["effective_n"] == 0.0


# ===========================================================================
# compute_sector_exposure
# ===========================================================================

class TestSectorExposure:
    def test_multiple_sectors(self):
        """Positions spread across sectors aggregate correctly."""
        positions = {
            "AAPL": 5000.0,
            "MSFT": 3000.0,
            "JPM": 2000.0,
        }
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
        }
        df = compute_sector_exposure(positions, sector_map)

        assert len(df) == 2
        tech = df[df["sector"] == "Technology"].iloc[0]
        fin = df[df["sector"] == "Financials"].iloc[0]

        assert tech["total_value"] == pytest.approx(8000.0)
        assert tech["weight"] == pytest.approx(0.80)
        assert tech["ticker_count"] == 2

        assert fin["total_value"] == pytest.approx(2000.0)
        assert fin["weight"] == pytest.approx(0.20)
        assert fin["ticker_count"] == 1

    def test_unknown_tickers(self):
        """Tickers missing from sector_map go to 'Unknown'."""
        positions = {"AAPL": 5000.0, "XYZ": 3000.0}
        sector_map = {"AAPL": "Technology"}
        df = compute_sector_exposure(positions, sector_map)

        unknown = df[df["sector"] == "Unknown"].iloc[0]
        assert unknown["total_value"] == pytest.approx(3000.0)
        assert unknown["ticker_count"] == 1

    def test_single_sector(self):
        """All tickers in the same sector produce one row."""
        positions = {"AAPL": 3000.0, "MSFT": 2000.0}
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
        df = compute_sector_exposure(positions, sector_map)

        assert len(df) == 1
        assert df.iloc[0]["sector"] == "Technology"
        assert df.iloc[0]["weight"] == pytest.approx(1.0)
        assert df.iloc[0]["ticker_count"] == 2

    def test_empty_positions(self):
        """Empty positions returns empty DataFrame with correct columns."""
        df = compute_sector_exposure({}, {"AAPL": "Technology"})
        assert df.empty
        assert list(df.columns) == ["sector", "total_value", "weight", "ticker_count"]

    def test_sorted_by_value_descending(self):
        """Sectors are sorted by total_value descending."""
        positions = {"A": 1000.0, "B": 5000.0, "C": 3000.0}
        sector_map = {"A": "Small", "B": "Big", "C": "Mid"}
        df = compute_sector_exposure(positions, sector_map)

        values = df["total_value"].tolist()
        assert values == sorted(values, reverse=True)

    def test_all_unknown_tickers(self):
        """All tickers unmapped -> single 'Unknown' row."""
        positions = {"X": 1000.0, "Y": 2000.0}
        df = compute_sector_exposure(positions, {})

        assert len(df) == 1
        assert df.iloc[0]["sector"] == "Unknown"
        assert df.iloc[0]["total_value"] == pytest.approx(3000.0)
        assert df.iloc[0]["ticker_count"] == 2
