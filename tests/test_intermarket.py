"""Tests for cross-asset intermarket data loading and computation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from datetime import date
from pathlib import Path

from backtester.data.market_data import MarketDataManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n=10, start="2024-01-02"):
    """Return a DatetimeIndex of *n* business days."""
    return pd.bdate_range(start=start, periods=n)


def _make_yf_df(dates, close_values):
    """Build a DataFrame mimicking yfinance.download output."""
    return pd.DataFrame(
        {
            "Close": close_values,
            "Open": close_values,
            "High": close_values,
            "Low": close_values,
        },
        index=dates,
    )


def _mock_intermarket_download(
    dates,
    copper_values,
    gold_values,
    dollar_values,
    copper_dates=None,
    gold_dates=None,
    dollar_dates=None,
):
    """Build a side_effect for yfinance.download returning copper/gold/dollar."""
    copper_dates = copper_dates if copper_dates is not None else dates
    gold_dates = gold_dates if gold_dates is not None else dates
    dollar_dates = dollar_dates if dollar_dates is not None else dates

    copper_df = _make_yf_df(copper_dates, copper_values)
    gold_df = _make_yf_df(gold_dates, gold_values)
    dollar_df = _make_yf_df(dollar_dates, dollar_values)

    def _side_effect(ticker, **kwargs):
        if ticker == "HG=F":
            return copper_df
        elif ticker == "GC=F":
            return gold_df
        elif ticker == "DX-Y.NYB":
            return dollar_df
        return pd.DataFrame()

    return _side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIntermarketData:
    """Cross-asset intermarket data loading and signal computation."""

    def test_cu_au_ratio_computed_correctly(self):
        """Copper / gold ratio is correct."""
        dates = _make_dates(5)
        side_effect = _mock_intermarket_download(
            dates, [4.0] * 5, [2000.0] * 5, [104.0] * 5
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 8))

        np.testing.assert_allclose(df["intermarket_cu_au_ratio"].values, 4.0 / 2000.0)

    def test_cu_au_momentum_positive(self):
        """Ratio increasing over 63 days → positive momentum."""
        n = 70
        dates = _make_dates(n)
        # Copper rising, gold flat → ratio rising
        copper = [3.0 + i * 0.1 for i in range(n)]
        gold = [2000.0] * n
        dollar = [104.0] * n

        side_effect = _mock_intermarket_download(dates, copper, gold, dollar)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 5, 1))

        # After 63 days, momentum should be positive
        last_momentum = df["intermarket_cu_au_momentum"].iloc[-1]
        assert last_momentum > 0

    def test_cu_au_momentum_negative(self):
        """Ratio decreasing over 63 days → negative momentum."""
        n = 70
        dates = _make_dates(n)
        # Copper falling, gold flat → ratio falling
        copper = [5.0 - i * 0.05 for i in range(n)]
        gold = [2000.0] * n
        dollar = [104.0] * n

        side_effect = _mock_intermarket_download(dates, copper, gold, dollar)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 5, 1))

        last_momentum = df["intermarket_cu_au_momentum"].iloc[-1]
        assert last_momentum < 0

    def test_zero_gold_produces_nan_ratio(self):
        """Gold = 0 → division by zero → NaN ratio."""
        dates = _make_dates(3)
        side_effect = _mock_intermarket_download(
            dates, [4.0] * 3, [0.0] * 3, [104.0] * 3
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        assert df["intermarket_cu_au_ratio"].isna().all()

    def test_nan_handling_ffill(self):
        """NaN in inputs is forward-filled before computation."""
        dates = _make_dates(3)
        side_effect = _mock_intermarket_download(
            dates, [4.0, np.nan, 4.0], [2000.0] * 3, [104.0, 104.0, np.nan]
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        # Row 2 NaN copper → ffilled to 4.0; Row 3 NaN dollar → ffilled to 104.0
        np.testing.assert_allclose(
            df["intermarket_cu_au_ratio"].values, [4.0 / 2000.0] * 3
        )
        np.testing.assert_allclose(df["intermarket_dollar"].values, [104.0] * 3)

    def test_correct_column_names(self):
        """Output has the expected columns."""
        dates = _make_dates(3)
        side_effect = _mock_intermarket_download(
            dates, [4.0] * 3, [2000.0] * 3, [104.0] * 3
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        assert set(df.columns) == {
            "intermarket_cu_au_ratio",
            "intermarket_cu_au_momentum",
            "intermarket_dollar",
        }

    def test_dollar_index_passed_through(self):
        """Dollar index is included as-is."""
        dates = _make_dates(3)
        side_effect = _mock_intermarket_download(
            dates, [4.0] * 3, [2000.0] * 3, [103.5, 104.0, 104.5]
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        np.testing.assert_allclose(
            df["intermarket_dollar"].values, [103.5, 104.0, 104.5]
        )

    def test_momentum_lookback_nans(self):
        """First 63 days of momentum should be NaN (lookback period)."""
        n = 70
        dates = _make_dates(n)
        side_effect = _mock_intermarket_download(
            dates, [4.0] * n, [2000.0] * n, [104.0] * n
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 5, 1))

        # First 63 rows should have NaN momentum
        assert df["intermarket_cu_au_momentum"].iloc[:63].isna().all()
        # Row 63 onward should be non-NaN (constant ratio → 0.0 momentum)
        assert df["intermarket_cu_au_momentum"].iloc[63:].notna().all()

    def test_empty_data_returns_empty_dataframe(self):
        """All empty downloads → empty DataFrame."""

        def _empty(ticker, **kwargs):
            return pd.DataFrame()

        with patch("backtester.data.market_data.yf.download", side_effect=_empty):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 8))

        assert df.empty
        assert set(df.columns) == {
            "intermarket_cu_au_ratio",
            "intermarket_cu_au_momentum",
            "intermarket_dollar",
        }

    def test_cache_hit(self, tmp_path):
        """Cached data is returned without calling yfinance."""
        dates = _make_dates(3)
        cache_dir = tmp_path / "cache"
        market_dir = cache_dir / "market"
        market_dir.mkdir(parents=True)

        # Write cache files for all three tickers
        for name, values in [
            ("HG_F", [4.0, 4.1, 4.2]),
            ("GC_F", [2000.0, 2010.0, 2020.0]),
            ("DX_Y_NYB", [103.5, 104.0, 104.5]),
        ]:
            df = pd.DataFrame({"Close": values}, index=dates)
            df.to_parquet(market_dir / f"{name}.parquet")

        with patch("backtester.data.market_data.yf.download") as mock_dl:
            mgr = MarketDataManager(cache_dir=str(cache_dir))
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        mock_dl.assert_not_called()
        assert not df.empty

    def test_different_date_ranges_aligned(self):
        """Assets on different calendars → outer join + ffill."""
        dates_cu = pd.to_datetime(["2024-01-02", "2024-01-03"])
        dates_au = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        dates_dx = pd.to_datetime(["2024-01-03", "2024-01-04"])

        side_effect = _mock_intermarket_download(
            None,
            [4.0, 4.1],
            [2000.0, 2010.0, 2020.0],
            [104.0, 104.5],
            copper_dates=dates_cu,
            gold_dates=dates_au,
            dollar_dates=dates_dx,
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        # Union of all dates → 3 rows
        assert len(df) == 3

    def test_single_day_of_data(self):
        """Single data point works correctly."""
        dates = pd.to_datetime(["2024-01-02"])
        side_effect = _mock_intermarket_download(
            dates, [4.0], [2000.0], [104.0]
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 2))

        assert len(df) == 1
        np.testing.assert_allclose(df["intermarket_cu_au_ratio"].iloc[0], 4.0 / 2000.0)

    def test_all_nan_data(self):
        """All NaN input → all NaN output."""
        dates = _make_dates(5)
        side_effect = _mock_intermarket_download(
            dates, [np.nan] * 5, [np.nan] * 5, [np.nan] * 5
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 8))

        assert df["intermarket_cu_au_ratio"].isna().all()
        assert df["intermarket_dollar"].isna().all()

    def test_large_magnitude_values(self):
        """Copper/gold have very different magnitudes — ratio still correct."""
        dates = _make_dates(3)
        # Copper ~$4, gold ~$2000 → ratio ~0.002
        side_effect = _mock_intermarket_download(
            dates, [4.25, 4.30, 4.35], [2050.0, 2060.0, 2070.0], [104.0] * 3
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        expected_ratios = [4.25 / 2050.0, 4.30 / 2060.0, 4.35 / 2070.0]
        np.testing.assert_allclose(
            df["intermarket_cu_au_ratio"].values, expected_ratios, rtol=1e-10
        )

    def test_index_is_datetimeindex(self):
        """Returned DataFrame has DatetimeIndex."""
        dates = _make_dates(3)
        side_effect = _mock_intermarket_download(
            dates, [4.0] * 3, [2000.0] * 3, [104.0] * 3
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_forward_fill_correctness(self):
        """Missing dates forward-filled correctly."""
        dates_cu = pd.to_datetime(["2024-01-02", "2024-01-04"])  # skip Jan 3
        dates_au = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        dates_dx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

        side_effect = _mock_intermarket_download(
            None,
            [4.0, 4.2],
            [2000.0, 2010.0, 2020.0],
            [104.0, 104.5, 105.0],
            copper_dates=dates_cu,
            gold_dates=dates_au,
            dollar_dates=dates_dx,
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        # Jan 3 copper should be forward-filled from Jan 2 (4.0)
        jan3_ratio = df.loc["2024-01-03", "intermarket_cu_au_ratio"]
        np.testing.assert_allclose(jan3_ratio, 4.0 / 2010.0)

    def test_partial_data_missing_ticker(self):
        """One ticker empty → still returns available data with NaN for missing."""
        dates = _make_dates(3)

        def _partial(ticker, **kwargs):
            if ticker == "HG=F":
                return pd.DataFrame()  # copper missing
            elif ticker == "GC=F":
                return _make_yf_df(dates, [2000.0] * 3)
            elif ticker == "DX-Y.NYB":
                return _make_yf_df(dates, [104.0] * 3)
            return pd.DataFrame()

        with patch("backtester.data.market_data.yf.download", side_effect=_partial):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        # Dollar should still be present
        assert df["intermarket_dollar"].notna().all()
        # Ratio should be NaN (no copper)
        assert df["intermarket_cu_au_ratio"].isna().all()

    def test_yfinance_exception_returns_empty(self):
        """yfinance raising an exception for all tickers → empty DataFrame."""

        def _raise(ticker, **kwargs):
            raise ConnectionError("network down")

        with patch("backtester.data.market_data.yf.download", side_effect=_raise):
            mgr = MarketDataManager()
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 8))

        assert df.empty

    def test_cache_miss_writes_cache(self, tmp_path):
        """Cache miss → fetch from yfinance and write cache files."""
        dates = _make_dates(3)
        cache_dir = tmp_path / "cache"
        side_effect = _mock_intermarket_download(
            dates, [4.0] * 3, [2000.0] * 3, [104.0] * 3
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager(cache_dir=str(cache_dir))
            df = mgr.load_intermarket_data(date(2024, 1, 2), date(2024, 1, 4))

        assert not df.empty
        assert (cache_dir / "market" / "HG_F.parquet").exists()
        assert (cache_dir / "market" / "GC_F.parquet").exists()
        assert (cache_dir / "market" / "DX_Y_NYB.parquet").exists()
