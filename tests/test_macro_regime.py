"""Tests for FRED macro regime data source."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fred_source(cache_dir=None):
    """Create a FredDataSource with a mocked fredapi.Fred."""
    with patch("backtester.data.fred_source.Fred") as MockFred:
        mock_fred_instance = MagicMock()
        MockFred.return_value = mock_fred_instance
        from backtester.data.fred_source import FredDataSource

        src = FredDataSource(api_key="test-key", cache_dir=cache_dir)
    return src, mock_fred_instance


def _date_index(start="2020-01-02", days=60):
    return pd.bdate_range(start=start, periods=days, freq="B")


def _make_series(index, value):
    """Create a pd.Series with the given constant value over index."""
    return pd.Series(value, index=index, dtype=float)


def _setup_mock_series(mock_fred, series_map):
    """Configure mock_fred.get_series to return data from series_map."""
    def side_effect(series_id, **kwargs):
        if series_id in series_map:
            return series_map[series_id]
        return pd.Series(dtype=float)

    mock_fred.get_series.side_effect = side_effect


# ---------------------------------------------------------------------------
# Tests: Regime score computation
# ---------------------------------------------------------------------------


class TestMacroRegimeScore:
    """Test the composite regime score computation."""

    def test_positive_yield_spread_is_bullish(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 1.5),   # positive → bullish
            "T10Y3M": _make_series(idx, 2.0),
            "BAMLH0A0HYM2": _make_series(idx, 3.0),
            "DBAA": _make_series(idx, 5.0),
            "DAAA": _make_series(idx, 3.0),
            "USSLIND": _make_series(idx, 100.0),
            "ICSA": _make_series(idx, 200_000),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        assert "fred_yield_spread_10y2y" in result.columns
        # Positive spread should contribute to bullish score
        assert result["fred_yield_spread_10y2y"].iloc[-1] == 1.5

    def test_negative_yield_spread_is_bearish(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, -0.5),  # inverted → bearish
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        # With only yield_spread available, regime should be 0.0 (bearish)
        assert result["fred_macro_regime"].iloc[-1] == 0.0

    def test_all_bullish_conditions_score_1(self):
        """All conditions bullish → regime = 1.0."""
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        # LEI rising: start at 100, go to 110
        lei_vals = np.linspace(100, 110, 60)
        lei = pd.Series(lei_vals, index=idx)
        # Claims falling: start at 300k, go to 200k
        claims_vals = np.linspace(300_000, 200_000, 60)
        claims = pd.Series(claims_vals, index=idx)
        # Credit spread: constant low value (below any rolling median)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 2.0),       # positive
            "T10Y3M": _make_series(idx, 2.5),
            "BAMLH0A0HYM2": _make_series(idx, 3.0), # constant → equal to median
            "DBAA": _make_series(idx, 5.0),
            "DAAA": _make_series(idx, 3.0),
            "USSLIND": lei,                           # rising
            "ICSA": claims,                           # falling
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 3, 25))
        last = result["fred_macro_regime"].iloc[-1]
        # yield_spread > 0: bullish. LEI rising: bullish. Claims falling: bullish.
        # Credit may not contribute since constant = median. At least 3/4 = 0.75
        assert last >= 0.75

    def test_all_bearish_conditions_score_0(self):
        """All conditions bearish → regime = 0.0."""
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        # LEI falling
        lei_vals = np.linspace(110, 100, 60)
        lei = pd.Series(lei_vals, index=idx)
        # Claims rising
        claims_vals = np.linspace(200_000, 300_000, 60)
        claims = pd.Series(claims_vals, index=idx)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, -0.5),       # inverted
            "T10Y3M": _make_series(idx, -0.3),
            "BAMLH0A0HYM2": _make_series(idx, 8.0),  # constant high
            "DBAA": _make_series(idx, 7.0),
            "DAAA": _make_series(idx, 3.0),
            "USSLIND": lei,                            # falling
            "ICSA": claims,                            # rising
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 3, 25))
        last = result["fred_macro_regime"].iloc[-1]
        assert last <= 0.25

    def test_mixed_conditions_score_around_half(self):
        """Some bullish, some bearish → score near 0.5."""
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        lei_vals = np.linspace(100, 110, 60)  # rising → bullish
        lei = pd.Series(lei_vals, index=idx)
        claims_vals = np.linspace(200_000, 300_000, 60)  # rising → bearish
        claims = pd.Series(claims_vals, index=idx)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 1.0),   # positive → bullish
            "T10Y3M": _make_series(idx, 1.5),
            "BAMLH0A0HYM2": _make_series(idx, 5.0),
            "DBAA": _make_series(idx, 6.0),
            "DAAA": _make_series(idx, 3.0),
            "USSLIND": lei,
            "ICSA": claims,
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 3, 25))
        last = result["fred_macro_regime"].iloc[-1]
        assert 0.25 <= last <= 0.75

    def test_partial_nan_still_computes_score(self):
        """Some series missing → regime computed from available conditions."""
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 2.0),  # positive → bullish
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        # Only yield_spread available and bullish → 1.0
        assert result["fred_macro_regime"].iloc[-1] == 1.0

    def test_all_nan_regime_is_nan(self):
        """No data for any series → empty DataFrame."""
        src, mock = _make_fred_source()
        _setup_mock_series(mock, {
            "T10Y2Y": pd.Series(dtype=float),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: BAA-AAA credit spread
# ---------------------------------------------------------------------------


class TestCreditSpread:
    def test_baa_aaa_spread_computed(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 1.0),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": _make_series(idx, 5.5),
            "DAAA": _make_series(idx, 3.2),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        assert "fred_credit_spread_baa_aaa" in result.columns
        np.testing.assert_almost_equal(
            result["fred_credit_spread_baa_aaa"].iloc[-1], 2.3, decimal=1
        )


# ---------------------------------------------------------------------------
# Tests: Forward-fill and data handling
# ---------------------------------------------------------------------------


class TestDataHandling:
    def test_forward_fill_across_gaps(self):
        """FRED data has weekend/holiday gaps — forward-fill handles them."""
        src, mock = _make_fred_source()
        # Create sparse index with gaps
        dates = pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-10"])
        spread = pd.Series([1.0, 1.5, 2.0], index=dates)
        _setup_mock_series(mock, {
            "T10Y2Y": spread,
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 1, 10))
        # After ffill, gaps should be filled
        assert not result["fred_yield_spread_10y2y"].isna().all()

    def test_empty_fred_response_returns_empty_df(self):
        src, mock = _make_fred_source()
        mock.get_series.return_value = pd.Series(dtype=float)
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        assert result.empty

    def test_single_day_of_data(self):
        src, mock = _make_fred_source()
        idx = pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 1.0),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 1, 2))
        assert len(result) == 1

    def test_column_names_correct(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 1.0),
            "T10Y3M": _make_series(idx, 1.5),
            "BAMLH0A0HYM2": _make_series(idx, 4.0),
            "DBAA": _make_series(idx, 5.0),
            "DAAA": _make_series(idx, 3.0),
            "USSLIND": _make_series(idx, 100.0),
            "ICSA": _make_series(idx, 200_000),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        expected_cols = {
            "fred_yield_spread_10y2y",
            "fred_yield_spread_10y3m",
            "fred_credit_spread_hy",
            "fred_credit_spread_baa_aaa",
            "fred_lei",
            "fred_claims",
            "fred_macro_regime",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_column_dtypes_are_float(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=30)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 1.0),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        for col in result.columns:
            assert result[col].dtype in [np.float64, float], f"{col} is not float"

    def test_date_range_filtering(self):
        src, mock = _make_fred_source()
        idx = _date_index(start="2020-01-02", days=100)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 1.0),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 5, 29))
        assert not result.empty
        assert isinstance(result.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Tests: Cache behavior
# ---------------------------------------------------------------------------


class TestMacroCache:
    def test_cache_hit_path(self, tmp_path):
        """When cache file exists, FRED API is not called."""
        cache_dir = tmp_path / "cache"
        fred_dir = cache_dir / "fred"
        fred_dir.mkdir(parents=True)

        # Pre-populate cache
        idx = _date_index(days=10)
        cached = pd.DataFrame({"T10Y2Y": _make_series(idx, 1.5)})
        cached.to_parquet(fred_dir / "T10Y2Y.parquet")

        src, mock = _make_fred_source(cache_dir=str(cache_dir))

        # Make FRED return empty for everything else
        _setup_mock_series(mock, {
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })

        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 1, 15))
        # T10Y2Y should NOT have been fetched from FRED (cache hit)
        calls = [
            c for c in mock.get_series.call_args_list
            if c[0][0] == "T10Y2Y"
        ]
        assert len(calls) == 0
        assert "fred_yield_spread_10y2y" in result.columns

    def test_cache_miss_fetches_from_fred(self, tmp_path):
        """When cache is empty, FRED API is called."""
        cache_dir = tmp_path / "cache"
        src, mock = _make_fred_source(cache_dir=str(cache_dir))
        idx = _date_index(days=10)
        _setup_mock_series(mock, {
            "T10Y2Y": _make_series(idx, 1.0),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 1, 15))
        assert not result.empty
        # Cache file should now exist
        assert (cache_dir / "fred" / "T10Y2Y.parquet").exists()


# ---------------------------------------------------------------------------
# Tests: LEI and Claims specifics
# ---------------------------------------------------------------------------


class TestLEIClaims:
    def test_lei_rising_is_bullish(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        lei = pd.Series(np.linspace(100, 120, 60), index=idx)
        _setup_mock_series(mock, {
            "T10Y2Y": pd.Series(dtype=float),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": lei,
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 3, 25))
        # Last value: LEI is rising → bullish → 1.0
        last = result["fred_macro_regime"].iloc[-1]
        assert last == 1.0

    def test_lei_falling_is_bearish(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        lei = pd.Series(np.linspace(120, 100, 60), index=idx)
        _setup_mock_series(mock, {
            "T10Y2Y": pd.Series(dtype=float),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": lei,
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 3, 25))
        last = result["fred_macro_regime"].iloc[-1]
        assert last == 0.0

    def test_claims_falling_is_bullish(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        claims = pd.Series(np.linspace(300_000, 200_000, 60), index=idx)
        _setup_mock_series(mock, {
            "T10Y2Y": pd.Series(dtype=float),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": claims,
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 3, 25))
        last = result["fred_macro_regime"].iloc[-1]
        assert last == 1.0

    def test_claims_rising_is_bearish(self):
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        claims = pd.Series(np.linspace(200_000, 300_000, 60), index=idx)
        _setup_mock_series(mock, {
            "T10Y2Y": pd.Series(dtype=float),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": pd.Series(dtype=float),
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": claims,
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 3, 25))
        last = result["fred_macro_regime"].iloc[-1]
        assert last == 0.0


# ---------------------------------------------------------------------------
# Tests: Constructor
# ---------------------------------------------------------------------------


class TestFredSourceInit:
    def test_missing_fredapi_raises(self):
        with patch("backtester.data.fred_source.Fred", None):
            from backtester.data.fred_source import FredDataSource
            with pytest.raises(ImportError, match="fredapi is required"):
                FredDataSource(api_key="key")

    def test_missing_api_key_raises(self):
        with patch("backtester.data.fred_source.Fred") as MockFred:
            MockFred.return_value = MagicMock()
            from backtester.data.fred_source import FredDataSource
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="FRED API key required"):
                    FredDataSource(api_key=None)

    def test_api_key_from_env(self):
        """API key can be read from FRED_API_KEY env var."""
        with patch("backtester.data.fred_source.Fred") as MockFred:
            MockFred.return_value = MagicMock()
            from backtester.data.fred_source import FredDataSource
            with patch.dict("os.environ", {"FRED_API_KEY": "env-key"}):
                src = FredDataSource(api_key=None)
            MockFred.assert_called_with(api_key="env-key")


class TestFredFetchErrors:
    def test_fred_api_exception_returns_empty_series(self):
        """FRED API raising exception for a series → treated as empty, no crash."""
        src, mock = _make_fred_source()
        idx = _date_index(days=30)

        def _side_effect(series_id, **kwargs):
            if series_id == "T10Y2Y":
                return _make_series(idx, 1.0)
            raise ConnectionError("FRED API unreachable")

        mock.get_series.side_effect = _side_effect
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 2, 14))
        # Should still produce results from available series
        assert not result.empty
        assert "fred_yield_spread_10y2y" in result.columns

    def test_credit_spread_hy_below_median_is_bullish(self):
        """HY credit spread below rolling median → bullish credit condition."""
        src, mock = _make_fred_source()
        idx = _date_index(days=60)
        # Credit spread dropping: starts at 6.0, ends at 2.0
        hy_vals = np.linspace(6.0, 2.0, 60)
        hy = pd.Series(hy_vals, index=idx)
        _setup_mock_series(mock, {
            "T10Y2Y": pd.Series(dtype=float),
            "T10Y3M": pd.Series(dtype=float),
            "BAMLH0A0HYM2": hy,
            "DBAA": pd.Series(dtype=float),
            "DAAA": pd.Series(dtype=float),
            "USSLIND": pd.Series(dtype=float),
            "ICSA": pd.Series(dtype=float),
        })
        result = src.load_macro_regime(date(2020, 1, 2), date(2020, 3, 25))
        # With only credit available and spread below median → bullish → 1.0
        last = result["fred_macro_regime"].iloc[-1]
        assert last == 1.0
