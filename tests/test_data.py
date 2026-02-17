"""Tests for data cache and manager."""

import tempfile
from datetime import date

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
