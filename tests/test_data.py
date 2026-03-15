"""Tests for data cache and manager."""

import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.data.cache import ParquetCache
from backtester.data.manager import DataManager
from tests.conftest import make_price_df, MockDataSource


class TestParquetCache:
    def test_save_and_load(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            cache.save("TEST", sample_df)

            assert cache.has("TEST")
            loaded = cache.load("TEST")
            assert loaded is not None
            assert len(loaded) == len(sample_df)
            pd.testing.assert_frame_equal(loaded, sample_df, check_index_type=False)

    def test_date_range(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            cache.save("TEST", sample_df)

            dr = cache.date_range("TEST")
            assert dr is not None
            assert dr[0] == sample_df.index[0].date()
            assert dr[1] == sample_df.index[-1].date()

    def test_missing_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            assert not cache.has("MISSING")
            assert cache.load("MISSING") is None
            assert cache.date_range("MISSING") is None

    def test_merge_and_save(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            # Save first half
            half = len(sample_df) // 2
            cache.save("TEST", sample_df.iloc[:half])

            # Merge second half (with some overlap)
            overlap_start = half - 10
            result = cache.merge_and_save("TEST", sample_df.iloc[overlap_start:])

            assert len(result) == len(sample_df)


class TestCacheTimestampNormalization:
    """Regression tests: cached data with non-midnight or timezone-aware
    timestamps must be normalized so reindex() matches the trading calendar.

    Without normalization, cached data from older yfinance/pandas versions
    (e.g. timestamps with 05:00 UTC or timezone info) causes _prepare() to
    drop most rows, producing sparse benchmark/price data.
    """

    def _make_df_with_timestamps(self, timestamps, start_price=100.0):
        """Create OHLCV DataFrame with specific timestamps as the index."""
        n = len(timestamps)
        rng = np.random.default_rng(42)
        prices = start_price + np.cumsum(rng.normal(0, 1, n))
        prices = np.maximum(prices, 1.0)
        return pd.DataFrame(
            {
                "Open": prices * 0.999,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": np.full(n, 1_000_000),
            },
            index=pd.DatetimeIndex(timestamps, name="Date"),
        )

    def test_cache_normalizes_non_midnight_timestamps(self):
        """Cached data stored with non-midnight timestamps (e.g. 05:00 UTC)
        should be normalized to midnight on load."""
        dates = pd.bdate_range("2020-01-02", periods=50, freq="B")
        # Simulate old yfinance data stored with 5am timestamps
        timestamps = [d + pd.Timedelta(hours=5) for d in dates]
        df = self._make_df_with_timestamps(timestamps)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            cache.save("TEST", df)
            loaded = cache.load("TEST")

            # All timestamps should be at midnight after normalization
            for ts in loaded.index:
                assert ts.hour == 0 and ts.minute == 0, (
                    f"Cached timestamp {ts} should be normalized to midnight"
                )
            assert len(loaded) == len(df)

    def test_cache_normalizes_timezone_aware_timestamps(self):
        """Cached data with timezone-aware timestamps should be converted
        to timezone-naive midnight on load."""
        dates = pd.bdate_range("2020-01-02", periods=50, freq="B")
        timestamps = dates.tz_localize("US/Eastern")
        df = self._make_df_with_timestamps(timestamps)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            cache.save("TEST", df)
            loaded = cache.load("TEST")

            assert loaded.index.tz is None, "Loaded index should be tz-naive"
            assert len(loaded) == len(df)

    def test_prepare_handles_non_midnight_timestamps(self):
        """DataManager._prepare() should normalize timestamps before
        reindexing, so non-midnight data doesn't get dropped."""
        dates = pd.bdate_range("2020-01-02", periods=50, freq="B")
        timestamps = [d + pd.Timedelta(hours=5) for d in dates]
        df = self._make_df_with_timestamps(timestamps)

        source = MockDataSource()
        source.add("TEST", df)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DataManager(cache_dir=tmpdir, source=source)
            # _prepare normalizes timestamps before reindex, so data
            # should NOT be dropped due to timestamp mismatch
            result = mgr.load("TEST", date(2020, 1, 2), date(2020, 3, 20))
            assert len(result) > 0, (
                "Data with non-midnight timestamps should not be dropped"
            )
            # Should retain most of the data (some dates might not be
            # NYSE trading days)
            assert len(result) > 30, (
                f"Expected ~40+ trading days, got {len(result)}"
            )

    def test_merge_normalizes_mixed_timestamps(self):
        """merge_and_save with old (non-midnight) and new (midnight) data
        should produce a clean merged result."""
        dates = pd.bdate_range("2020-01-02", periods=100, freq="B")

        # Old data: first 60 days with 5am timestamps
        old_ts = [d + pd.Timedelta(hours=5) for d in dates[:60]]
        old_df = self._make_df_with_timestamps(old_ts)

        # New data: last 60 days with midnight timestamps (overlap of 20)
        new_ts = dates[40:]
        new_df = self._make_df_with_timestamps(
            [pd.Timestamp(d.date()) for d in new_ts],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            cache.save("TEST", old_df)
            merged = cache.merge_and_save("TEST", new_df)

            # After normalization + dedup, should have ~100 unique dates
            assert len(merged) >= 95, (
                f"Merged data has {len(merged)} rows, expected ~100 "
                f"(old non-midnight + new midnight should merge cleanly)"
            )


class TestDataManager:
    def test_load_from_source(self):
        source = MockDataSource()
        df = make_price_df(start="2020-01-02", days=252)
        source.add("TEST", df)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DataManager(cache_dir=tmpdir, source=source)
            result = mgr.load("TEST", date(2020, 1, 2), date(2020, 12, 31))
            assert result is not None
            assert len(result) > 0
            assert "Close" in result.columns

    def test_load_many(self):
        source = MockDataSource()
        source.add("A", make_price_df(start="2020-01-02", days=252))
        source.add("B", make_price_df(start="2020-01-02", days=252, start_price=50))

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DataManager(cache_dir=tmpdir, source=source)
            results = mgr.load_many(["A", "B"], date(2020, 1, 2), date(2020, 12, 31))
            assert "A" in results
            assert "B" in results

    def test_load_start_equals_end(self):
        """Loading data where start == end on a trading day returns 1 row."""
        source = MockDataSource()
        df = make_price_df(start="2020-01-02", days=252)
        source.add("TEST", df)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DataManager(cache_dir=tmpdir, source=source)
            # 2020-01-02 is a Thursday (trading day)
            result = mgr.load("TEST", date(2020, 1, 2), date(2020, 1, 2))
            assert len(result) == 1
            assert "Close" in result.columns


class TestParquetCacheSingleDay:
    """Edge cases for cache with minimal data."""

    def test_cache_single_day(self):
        """Save and load a single day of data."""
        single = make_price_df(start="2020-01-02", days=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            cache.save("TEST", single)

            assert cache.has("TEST")
            loaded = cache.load("TEST")
            assert loaded is not None
            assert len(loaded) == 1
            pd.testing.assert_frame_equal(loaded, single, check_index_type=False)

    def test_date_range_single_row(self):
        """date_range with one row: start == end."""
        single = make_price_df(start="2020-01-02", days=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            cache.save("TEST", single)

            dr = cache.date_range("TEST")
            assert dr is not None
            assert dr[0] == dr[1]  # single day: start == end

    def test_merge_into_empty_cache(self):
        """merge_and_save on a symbol not in cache should just save the data."""
        df = make_price_df(start="2020-01-02", days=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(tmpdir)
            result = cache.merge_and_save("NEW", df)
            assert len(result) == 10
            assert cache.has("NEW")


class TestForwardFillEdgeCases:
    """Test the MAX_FFILL_DAYS=5 limit in DataManager._prepare()."""

    def test_ffill_within_limit(self):
        """A 5-day gap (at the limit) should be forward-filled."""
        source = MockDataSource()
        df = make_price_df(start="2020-01-02", days=252)
        # Remove 5 consecutive trading days to create a gap
        # Days at index positions 20-24 (business days)
        gap_dates = df.index[20:25]
        df_with_gap = df.drop(gap_dates)
        source.add("TEST", df_with_gap)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DataManager(cache_dir=tmpdir, source=source)
            result = mgr.load("TEST", date(2020, 1, 2), date(2020, 12, 31))
            # The gap should be forward-filled, so we should have data for those days
            for gap_date in gap_dates:
                gap_d = gap_date.date() if hasattr(gap_date, "date") else gap_date
                matching = result.index[result.index == pd.Timestamp(gap_d)]
                assert len(matching) == 1, (
                    f"Date {gap_d} in 5-day gap should be forward-filled"
                )

    def test_ffill_beyond_limit_drops_rows(self):
        """A 6+ day gap exceeds MAX_FFILL_DAYS=5 and should NOT be fully filled."""
        source = MockDataSource()
        df = make_price_df(start="2020-01-02", days=252)
        # Remove 7 consecutive trading days to create a gap beyond the limit
        gap_dates = df.index[20:27]
        df_with_gap = df.drop(gap_dates)
        source.add("TEST", df_with_gap)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DataManager(cache_dir=tmpdir, source=source)
            result = mgr.load("TEST", date(2020, 1, 2), date(2020, 12, 31))
            # With a 7-day gap, only the first 5 are forward-filled;
            # the remaining 2 days should be dropped (no Close)
            gap_beyond = gap_dates[5:]  # days 6 and 7 of the gap
            for gap_date in gap_beyond:
                gap_d = gap_date.date() if hasattr(gap_date, "date") else gap_date
                matching = result.index[result.index == pd.Timestamp(gap_d)]
                assert len(matching) == 0, (
                    f"Date {gap_d} beyond ffill limit should be dropped"
                )
