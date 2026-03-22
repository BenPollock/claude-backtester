"""Tests for data cache and manager."""

import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.data.cache import ParquetCache
from backtester.data.manager import DataManager, resample_ohlcv
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


class TestResampleOHLCV:
    """Tests for the resample_ohlcv function."""

    def _make_daily(self, days=60):
        """Create daily OHLCV data."""
        dates = pd.bdate_range("2020-01-02", periods=days, freq="B")
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 1, days))
        close = np.maximum(close, 1.0)
        return pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(days, 1_000_000),
            },
            index=pd.DatetimeIndex(dates, name="Date"),
        )

    def test_weekly_resample_basic(self):
        """Weekly resample produces fewer rows with OHLCV columns."""
        daily = self._make_daily(60)
        weekly = resample_ohlcv(daily, "weekly")
        assert len(weekly) < len(daily)
        assert len(weekly) > 0
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in weekly.columns

    def test_monthly_resample_basic(self):
        """Monthly resample produces fewer rows."""
        daily = self._make_daily(120)
        monthly = resample_ohlcv(daily, "monthly")
        assert len(monthly) < len(daily)
        assert len(monthly) > 0

    def test_invalid_timeframe_raises(self):
        """Invalid timeframe raises ValueError."""
        daily = self._make_daily(10)
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            resample_ohlcv(daily, "quarterly")

    def test_weekly_high_is_max(self):
        """Weekly High should be the max of daily Highs in that week."""
        daily = self._make_daily(10)
        weekly = resample_ohlcv(daily, "weekly")
        # For the first week, the High should be the max of daily Highs
        first_week_end = weekly.index[0]
        daily_in_week = daily.loc[daily.index <= first_week_end]
        assert weekly["High"].iloc[0] == daily_in_week["High"].max()

    def test_weekly_low_is_min(self):
        """Weekly Low should be the min of daily Lows."""
        daily = self._make_daily(10)
        weekly = resample_ohlcv(daily, "weekly")
        first_week_end = weekly.index[0]
        daily_in_week = daily.loc[daily.index <= first_week_end]
        assert weekly["Low"].iloc[0] == daily_in_week["Low"].min()

    def test_weekly_volume_is_sum(self):
        """Weekly Volume should be sum of daily Volumes."""
        daily = self._make_daily(10)
        weekly = resample_ohlcv(daily, "weekly")
        first_week_end = weekly.index[0]
        daily_in_week = daily.loc[daily.index <= first_week_end]
        assert weekly["Volume"].iloc[0] == daily_in_week["Volume"].sum()

    def test_index_aligns_with_trading_days(self):
        """Resampled index values should exist in the original daily index."""
        daily = self._make_daily(60)
        weekly = resample_ohlcv(daily, "weekly")
        for dt in weekly.index:
            assert dt in daily.index, f"Resampled date {dt} not in daily index"

    def test_does_not_mutate_input(self):
        """resample_ohlcv should not mutate the input DataFrame."""
        daily = self._make_daily(30)
        original_len = len(daily)
        original_cols = list(daily.columns)
        resample_ohlcv(daily, "weekly")
        assert len(daily) == original_len
        assert list(daily.columns) == original_cols


class TestResampleOHLCVEdgeCases:
    """Edge case tests for resample_ohlcv."""

    def _make_daily(self, days=60):
        dates = pd.bdate_range("2020-01-02", periods=days, freq="B")
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 1, days))
        close = np.maximum(close, 1.0)
        return pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(days, 1_000_000),
            },
            index=pd.DatetimeIndex(dates, name="Date"),
        )

    def test_single_day_input(self):
        """resample_ohlcv with 1 row produces 1 row."""
        daily = self._make_daily(1)
        weekly = resample_ohlcv(daily, "weekly")
        assert len(weekly) == 1
        assert weekly["Close"].iloc[0] == daily["Close"].iloc[0]

    def test_two_day_input(self):
        """resample_ohlcv with 2 rows in same week produces 1 weekly row."""
        daily = self._make_daily(2)
        weekly = resample_ohlcv(daily, "weekly")
        assert len(weekly) >= 1
        # High should be the max of both days
        assert weekly["High"].iloc[0] == daily["High"].max()

    def test_monthly_with_few_days(self):
        """Monthly resample with < 1 month of data produces 1 row."""
        daily = self._make_daily(10)
        monthly = resample_ohlcv(daily, "monthly")
        assert len(monthly) >= 1

    def test_weekly_close_is_last(self):
        """Weekly Close should be the last Close of the week."""
        daily = self._make_daily(10)
        weekly = resample_ohlcv(daily, "weekly")
        first_week_end = weekly.index[0]
        daily_in_week = daily.loc[daily.index <= first_week_end]
        assert weekly["Close"].iloc[0] == daily_in_week["Close"].iloc[-1]

    def test_weekly_open_is_first(self):
        """Weekly Open should be the first Open of the week."""
        daily = self._make_daily(10)
        weekly = resample_ohlcv(daily, "weekly")
        first_week_end = weekly.index[0]
        daily_in_week = daily.loc[daily.index <= first_week_end]
        assert weekly["Open"].iloc[0] == daily_in_week["Open"].iloc[0]


class TestDataManagerEdgeCases:
    """Additional DataManager edge cases."""

    def test_load_many_with_missing_symbol(self):
        """load_many skips symbols that the source doesn't have."""
        source = MockDataSource()
        source.add("A", make_price_df(start="2020-01-02", days=50))
        # "B" is not added to source

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DataManager(cache_dir=tmpdir, source=source)
            results = mgr.load_many(["A", "B"], date(2020, 1, 2), date(2020, 3, 31))
            assert "A" in results
            # "B" should either be missing or empty, depending on implementation
            if "B" in results:
                assert results["B"] is None or len(results["B"]) == 0


class TestCacheCoverageFallback:
    """Test cache coverage validation in DataManager.load."""

    def test_low_coverage_refetches(self):
        """When cache covers < 90% of trading days, data is re-fetched."""
        source = MockDataSource()
        full_df = make_price_df(start="2020-01-02", days=252)
        source.add("TEST", full_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DataManager(cache_dir=tmpdir, source=source)
            # Pre-populate cache with sparse data (only every 20th day)
            sparse_df = full_df.iloc[::20]  # ~5% coverage
            mgr._cache.save("TEST", sparse_df)

            # Load should detect low coverage and re-fetch
            result = mgr.load("TEST", date(2020, 1, 2), date(2020, 12, 31))
            assert len(result) > 100, "Should re-fetch when cache coverage is low"
