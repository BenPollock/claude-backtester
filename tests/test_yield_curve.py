"""Tests for FRED yield curve data source."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fred_source(cache_dir=None):
    with patch("backtester.data.fred_source.Fred") as MockFred:
        mock_fred_instance = MagicMock()
        MockFred.return_value = mock_fred_instance
        from backtester.data.fred_source import FredDataSource

        src = FredDataSource(api_key="test-key", cache_dir=cache_dir)
    return src, mock_fred_instance


def _date_index(start="2020-01-02", days=60):
    return pd.bdate_range(start=start, periods=days, freq="B")


def _make_series(index, value):
    return pd.Series(value, index=index, dtype=float)


def _setup_mock_series(mock_fred, series_map):
    def side_effect(series_id, **kwargs):
        if series_id in series_map:
            return series_map[series_id]
        return pd.Series(dtype=float)

    mock_fred.get_series.side_effect = side_effect


def _full_yield_series(idx, values=None):
    """Return a series_map with all yield series populated."""
    if values is None:
        values = {
            "DGS3MO": 0.5, "DGS2": 1.5, "DGS5": 2.0, "DGS10": 3.0,
            "DGS30": 3.5, "T5YIE": 1.2, "T10YIE": 1.8, "DFF": 0.25,
        }
    return {sid: _make_series(idx, val) for sid, val in values.items()}


# ---------------------------------------------------------------------------
# Tests: Yield spread
# ---------------------------------------------------------------------------


class TestYieldSpread:
    def test_positive_spread_normal_curve(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx, {
            "DGS3MO": 0.5, "DGS2": 1.5, "DGS5": 2.0, "DGS10": 3.0,
            "DGS30": 3.5, "T5YIE": 1.2, "T10YIE": 1.8, "DFF": 0.25,
        }))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        assert "yield_spread" in result.columns
        np.testing.assert_almost_equal(result["yield_spread"].iloc[-1], 1.5)

    def test_negative_spread_inverted_curve(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx, {
            "DGS3MO": 3.0, "DGS2": 3.5, "DGS5": 2.5, "DGS10": 2.0,
            "DGS30": 1.5, "T5YIE": 1.2, "T10YIE": 1.8, "DFF": 3.0,
        }))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        np.testing.assert_almost_equal(result["yield_spread"].iloc[-1], -1.5)

    def test_flat_curve_zero_spread(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx, {
            "DGS3MO": 2.0, "DGS2": 2.0, "DGS5": 2.0, "DGS10": 2.0,
            "DGS30": 2.0, "T5YIE": 1.0, "T10YIE": 1.0, "DFF": 2.0,
        }))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        np.testing.assert_almost_equal(result["yield_spread"].iloc[-1], 0.0)


# ---------------------------------------------------------------------------
# Tests: Real yield
# ---------------------------------------------------------------------------


class TestRealYield:
    def test_real_yield_computation(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx, {
            "DGS3MO": 0.5, "DGS2": 1.5, "DGS5": 2.0, "DGS10": 3.0,
            "DGS30": 3.5, "T5YIE": 1.2, "T10YIE": 1.8, "DFF": 0.25,
        }))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        assert "yield_real_10y" in result.columns
        # 3.0 - 1.8 = 1.2
        np.testing.assert_almost_equal(result["yield_real_10y"].iloc[-1], 1.2)

    def test_negative_real_yield(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx, {
            "DGS3MO": 0.1, "DGS2": 0.2, "DGS5": 0.5, "DGS10": 1.0,
            "DGS30": 1.5, "T5YIE": 2.0, "T10YIE": 2.5, "DFF": 0.0,
        }))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        # 1.0 - 2.5 = -1.5
        np.testing.assert_almost_equal(result["yield_real_10y"].iloc[-1], -1.5)


# ---------------------------------------------------------------------------
# Tests: All series present
# ---------------------------------------------------------------------------


class TestYieldCurveColumns:
    def test_all_yield_series_present(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        expected = {
            "yield_3m", "yield_2y", "yield_5y", "yield_10y", "yield_30y",
            "yield_breakeven_5y", "yield_breakeven_10y", "yield_fed_funds",
            "yield_spread", "yield_real_10y",
        }
        assert expected.issubset(set(result.columns))

    def test_partial_yields_missing_nan_for_computed(self):
        """If 10y or 2y missing, spread column absent."""
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, {
            "DGS3MO": _make_series(idx, 0.5),
            "DGS2": pd.Series(dtype=float),  # missing
            "DGS5": _make_series(idx, 2.0),
            "DGS10": pd.Series(dtype=float),  # missing
            "DGS30": _make_series(idx, 3.5),
            "T5YIE": pd.Series(dtype=float),
            "T10YIE": pd.Series(dtype=float),
            "DFF": _make_series(idx, 0.25),
        })
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        # yield_spread and yield_real_10y should not be present
        assert "yield_spread" not in result.columns
        assert "yield_real_10y" not in result.columns

    def test_correct_column_prefixes(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        for col in result.columns:
            assert col.startswith("yield_"), f"Column {col} missing yield_ prefix"


# ---------------------------------------------------------------------------
# Tests: Data handling
# ---------------------------------------------------------------------------


class TestYieldDataHandling:
    def test_empty_data_returns_empty_df(self):
        src, mock = _make_fred_source()
        mock.get_series.return_value = pd.Series(dtype=float)
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        assert result.empty

    def test_forward_fill_across_gaps(self):
        src, mock = _make_fred_source()
        dates = pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-10"])
        rate = pd.Series([1.0, 1.5, 2.0], index=dates)
        series_map = {sid: pd.Series(dtype=float) for sid in [
            "DGS3MO", "DGS2", "DGS5", "DGS30", "T5YIE", "T10YIE", "DFF"
        ]}
        series_map["DGS10"] = rate
        _setup_mock_series(mock, series_map)
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 1, 10))
        # After ffill, intermediate dates should be filled
        assert not result["yield_10y"].isna().all()

    def test_single_day(self):
        src, mock = _make_fred_source()
        idx = pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
        _setup_mock_series(mock, _full_yield_series(idx))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 1, 2))
        assert len(result) == 1

    def test_zero_rates(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx, {
            "DGS3MO": 0.0, "DGS2": 0.0, "DGS5": 0.0, "DGS10": 0.0,
            "DGS30": 0.0, "T5YIE": 0.0, "T10YIE": 0.0, "DFF": 0.0,
        }))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        assert "yield_spread" in result.columns
        np.testing.assert_almost_equal(result["yield_spread"].iloc[-1], 0.0)

    def test_large_positive_yields(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx, {
            "DGS3MO": 10.0, "DGS2": 12.0, "DGS5": 14.0, "DGS10": 16.0,
            "DGS30": 18.0, "T5YIE": 5.0, "T10YIE": 6.0, "DFF": 10.0,
        }))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        np.testing.assert_almost_equal(result["yield_spread"].iloc[-1], 4.0)

    def test_fed_funds_included(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        assert "yield_fed_funds" in result.columns
        assert result["yield_fed_funds"].iloc[-1] == 0.25

    def test_index_is_datetimeindex(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_breakeven_computation(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, _full_yield_series(idx, {
            "DGS3MO": 0.5, "DGS2": 1.5, "DGS5": 2.5, "DGS10": 3.5,
            "DGS30": 4.0, "T5YIE": 2.0, "T10YIE": 2.2, "DFF": 0.25,
        }))
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 2, 14))
        assert "yield_breakeven_5y" in result.columns
        assert result["yield_breakeven_5y"].iloc[-1] == 2.0
        assert result["yield_breakeven_10y"].iloc[-1] == 2.2

    def test_multiple_curve_shapes_across_time(self):
        """Yield curve changes shape over time."""
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        # First 30 days: normal curve, last 30 days: inverted
        ten_y = pd.Series(
            [3.0] * 30 + [1.0] * 30, index=idx
        )
        two_y = pd.Series(
            [1.5] * 30 + [2.5] * 30, index=idx
        )
        base = {sid: pd.Series(dtype=float) for sid in [
            "DGS3MO", "DGS5", "DGS30", "T5YIE", "T10YIE", "DFF"
        ]}
        base["DGS10"] = ten_y
        base["DGS2"] = two_y
        _setup_mock_series(mock, base)
        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 3, 25))
        # First half: spread = 3.0 - 1.5 = 1.5
        assert result["yield_spread"].iloc[0] == 1.5
        # Second half: spread = 1.0 - 2.5 = -1.5
        assert result["yield_spread"].iloc[-1] == -1.5


# ---------------------------------------------------------------------------
# Tests: Cache behavior
# ---------------------------------------------------------------------------


class TestYieldCurveCache:
    def test_cache_hit(self, tmp_path):
        cache_dir = tmp_path / "cache"
        fred_dir = cache_dir / "fred"
        fred_dir.mkdir(parents=True)

        idx = _date_index(days=10)
        cached = pd.DataFrame({"DGS10": _make_series(idx, 3.0)})
        cached.to_parquet(fred_dir / "DGS10.parquet")

        src, mock = _make_fred_source(cache_dir=str(cache_dir))
        base = {sid: pd.Series(dtype=float) for sid in [
            "DGS3MO", "DGS2", "DGS5", "DGS30", "T5YIE", "T10YIE", "DFF"
        ]}
        _setup_mock_series(mock, base)

        result = src.load_yield_curve(date(2020, 1, 2), date(2020, 1, 15))
        assert "yield_10y" in result.columns
        # DGS10 should not have been fetched from FRED
        dgs10_calls = [
            c for c in mock.get_series.call_args_list
            if c[0][0] == "DGS10"
        ]
        assert len(dgs10_calls) == 0
