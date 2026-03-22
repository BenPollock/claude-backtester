"""Tests for VIX term structure data loading and computation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import date
from pathlib import Path

from backtester.data.market_data import MarketDataManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n=10, start="2024-01-02"):
    """Return a DatetimeIndex of *n* business days starting from *start*."""
    return pd.bdate_range(start=start, periods=n)


def _mock_yf_download(vix_dates, vix_values, vix3m_dates=None, vix3m_values=None):
    """Build a side_effect for yfinance.download that returns VIX / VIX3M data."""
    vix3m_dates = vix3m_dates if vix3m_dates is not None else vix_dates
    vix3m_values = vix3m_values if vix3m_values is not None else vix_values

    vix_df = pd.DataFrame(
        {"Close": vix_values, "Open": vix_values, "High": vix_values, "Low": vix_values},
        index=vix_dates,
    )
    vix3m_df = pd.DataFrame(
        {"Close": vix3m_values, "Open": vix3m_values, "High": vix3m_values, "Low": vix3m_values},
        index=vix3m_dates,
    )

    def _side_effect(ticker, **kwargs):
        if ticker == "^VIX":
            return vix_df
        elif ticker == "^VIX3M":
            return vix3m_df
        return pd.DataFrame()

    return _side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVixTermStructure:
    """VIX term structure loading and regime classification."""

    def test_contango_regime(self):
        """VIX < VIX3M → ratio < 1.0 → contango."""
        dates = _make_dates(5)
        side_effect = _mock_yf_download(dates, [15.0] * 5, vix3m_values=[20.0] * 5)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 8))

        assert (df["vix_regime"] == "contango").all()
        assert (df["vix_ratio"] < 1.0).all()

    def test_backwardation_regime(self):
        """VIX > VIX3M → ratio > 1.0 → backwardation."""
        dates = _make_dates(5)
        side_effect = _mock_yf_download(dates, [25.0] * 5, vix3m_values=[20.0] * 5)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 8))

        assert (df["vix_regime"] == "backwardation").all()
        assert (df["vix_ratio"] > 1.0).all()

    def test_flat_term_structure(self):
        """VIX ≈ VIX3M → ratio ≈ 1.0."""
        dates = _make_dates(5)
        side_effect = _mock_yf_download(dates, [20.0] * 5, vix3m_values=[20.0] * 5)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 8))

        np.testing.assert_allclose(df["vix_ratio"].values, 1.0)

    def test_zero_vix3m_produces_nan_ratio(self):
        """Division by zero in VIX3M → NaN ratio."""
        dates = _make_dates(3)
        side_effect = _mock_yf_download(dates, [15.0] * 3, vix3m_values=[0.0] * 3)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert df["vix_ratio"].isna().all()

    def test_nan_handling_ffill(self):
        """NaN in VIX or VIX3M is forward-filled before ratio computation."""
        dates = _make_dates(3)
        side_effect = _mock_yf_download(
            dates, [15.0, np.nan, 15.0], vix3m_values=[20.0, 20.0, np.nan]
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        # Row 2 NaN vix → ffilled to 15.0; Row 3 NaN vix3m → ffilled to 20.0
        # So ratio is computed on ffilled values, no NaNs remain
        assert df["vix_ratio"].notna().all()
        np.testing.assert_allclose(df["vix_ratio"].values, [15.0 / 20.0] * 3)

    def test_correct_column_names(self):
        """Output has the expected columns."""
        dates = _make_dates(3)
        side_effect = _mock_yf_download(dates, [15.0] * 3, vix3m_values=[20.0] * 3)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert set(df.columns) == {"vix_close", "vix_3m", "vix_ratio", "vix_regime"}

    def test_correct_dtypes(self):
        """Float for numeric columns, object/string for regime."""
        dates = _make_dates(3)
        side_effect = _mock_yf_download(dates, [15.0] * 3, vix3m_values=[20.0] * 3)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert pd.api.types.is_float_dtype(df["vix_close"])
        assert pd.api.types.is_float_dtype(df["vix_3m"])
        assert pd.api.types.is_float_dtype(df["vix_ratio"])
        assert pd.api.types.is_string_dtype(df["vix_regime"])

    def test_date_alignment_different_calendars(self):
        """VIX and VIX3M on different date ranges → outer join + ffill."""
        dates_vix = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        dates_3m = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])

        side_effect = _mock_yf_download(
            dates_vix, [15.0, 16.0, 17.0],
            vix3m_dates=dates_3m, vix3m_values=[20.0, 21.0, 22.0],
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 5))

        # Should have dates from both series (outer join)
        assert len(df) >= 3

    def test_empty_yfinance_result(self):
        """Empty download → empty DataFrame, no crash."""

        def _empty(ticker, **kwargs):
            return pd.DataFrame()

        with patch("backtester.data.market_data.yf.download", side_effect=_empty):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 8))

        assert df.empty
        assert set(df.columns) == {"vix_close", "vix_3m", "vix_ratio", "vix_regime"}

    def test_cache_hit(self, tmp_path):
        """Cached data is returned without calling yfinance."""
        dates = _make_dates(3)
        cache_dir = tmp_path / "cache"
        market_dir = cache_dir / "market"
        market_dir.mkdir(parents=True)

        # Write cache files
        for name in ["VIX", "VIX3M"]:
            df = pd.DataFrame({"Close": [15.0, 16.0, 17.0]}, index=dates)
            (market_dir / f"{name}.parquet").parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(market_dir / f"{name}.parquet")

        with patch("backtester.data.market_data.yf.download") as mock_dl:
            mgr = MarketDataManager(cache_dir=str(cache_dir))
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        mock_dl.assert_not_called()
        assert not df.empty

    def test_cache_miss_fetches_and_caches(self, tmp_path):
        """Cache miss → fetch from yfinance and write cache."""
        dates = _make_dates(3)
        cache_dir = tmp_path / "cache"
        side_effect = _mock_yf_download(dates, [15.0] * 3, vix3m_values=[20.0] * 3)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager(cache_dir=str(cache_dir))
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert not df.empty
        assert (cache_dir / "market" / "VIX.parquet").exists()
        assert (cache_dir / "market" / "VIX3M.parquet").exists()

    def test_regime_boundary_at_exactly_one(self):
        """Ratio == 1.0 → contango (boundary: backwardation requires > 1.0)."""
        dates = _make_dates(3)
        side_effect = _mock_yf_download(dates, [20.0] * 3, vix3m_values=[20.0] * 3)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert (df["vix_regime"] == "contango").all()

    def test_regime_transitions(self):
        """Multiple days with transitions between regimes."""
        dates = _make_dates(4)
        # Day 1-2: contango, Day 3-4: backwardation
        side_effect = _mock_yf_download(
            dates, [15.0, 15.0, 25.0, 30.0],
            vix3m_values=[20.0, 20.0, 20.0, 20.0],
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 5))

        regimes = df["vix_regime"].tolist()
        assert regimes[:2] == ["contango", "contango"]
        assert regimes[2:] == ["backwardation", "backwardation"]

    def test_forward_fill_behavior(self):
        """Gaps in data are forward-filled."""
        dates_vix = pd.to_datetime(["2024-01-02", "2024-01-04"])  # skip Jan 3
        dates_3m = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

        side_effect = _mock_yf_download(
            dates_vix, [15.0, 17.0],
            vix3m_dates=dates_3m, vix3m_values=[20.0, 21.0, 22.0],
        )

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        # Jan 3 should have VIX forward-filled from Jan 2
        jan3 = df.loc["2024-01-03"]
        assert jan3["vix_close"] == 15.0  # ffilled from Jan 2

    def test_index_is_datetimeindex(self):
        """Returned DataFrame has DatetimeIndex."""
        dates = _make_dates(3)
        side_effect = _mock_yf_download(dates, [15.0] * 3, vix3m_values=[20.0] * 3)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_single_day_of_data(self):
        """Single data point works correctly."""
        dates = pd.to_datetime(["2024-01-02"])
        side_effect = _mock_yf_download(dates, [18.0], vix3m_values=[22.0])

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 2))

        assert len(df) == 1
        assert df.iloc[0]["vix_close"] == 18.0
        assert df.iloc[0]["vix_3m"] == 22.0

    def test_yfinance_exception_returns_empty(self):
        """yfinance raising an exception → empty DataFrame."""

        def _raise(ticker, **kwargs):
            raise ConnectionError("network down")

        with patch("backtester.data.market_data.yf.download", side_effect=_raise):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 8))

        assert df.empty

    def test_no_cache_dir(self):
        """MarketDataManager without cache_dir still works (no caching)."""
        dates = _make_dates(3)
        side_effect = _mock_yf_download(dates, [15.0] * 3, vix3m_values=[20.0] * 3)

        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager(cache_dir=None)
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert not df.empty
        assert set(df.columns) == {"vix_close", "vix_3m", "vix_ratio", "vix_regime"}

    def test_partial_data_vix_only(self):
        """VIX present but VIX3M empty → VIX3M column is NaN, ratio NaN."""
        dates = _make_dates(3)

        def _partial(ticker, **kwargs):
            if ticker == "^VIX":
                return pd.DataFrame(
                    {"Close": [15.0, 16.0, 17.0], "Open": [15.0] * 3,
                     "High": [15.0] * 3, "Low": [15.0] * 3},
                    index=dates,
                )
            return pd.DataFrame()

        with patch("backtester.data.market_data.yf.download", side_effect=_partial):
            mgr = MarketDataManager()
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert not df.empty
        assert df["vix_close"].notna().all()
        assert df["vix_3m"].isna().all()

    def test_corrupted_cache_falls_back_to_fetch(self, tmp_path):
        """Corrupted cache file → falls back to yfinance fetch."""
        dates = _make_dates(3)
        cache_dir = tmp_path / "cache"
        market_dir = cache_dir / "market"
        market_dir.mkdir(parents=True)

        # Write corrupted cache (invalid parquet)
        (market_dir / "VIX.parquet").write_bytes(b"not a parquet file")
        (market_dir / "VIX3M.parquet").write_bytes(b"not a parquet file")

        side_effect = _mock_yf_download(dates, [15.0] * 3, vix3m_values=[20.0] * 3)
        with patch("backtester.data.market_data.yf.download", side_effect=side_effect):
            mgr = MarketDataManager(cache_dir=str(cache_dir))
            df = mgr.load_vix_data(date(2024, 1, 2), date(2024, 1, 4))

        assert not df.empty
        assert (df["vix_regime"] == "contango").all()
