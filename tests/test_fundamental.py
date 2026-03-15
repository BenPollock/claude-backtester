"""Tests for FundamentalDataManager."""

import os
import tempfile
from datetime import date

import pytest

from backtester.data.fundamental import FundamentalDataManager


def _write_csv(tmpdir, filename, content):
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


class TestFundamentalDataManager:
    def test_load_and_get_basic(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            "2020-01-15,AAPL,pe_ratio,25.0\n"
            "2020-04-15,AAPL,pe_ratio,22.5\n"
            "2020-07-15,AAPL,pe_ratio,28.0\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        # Query after all data
        val = mgr.get("AAPL", "pe_ratio", date(2020, 12, 31))
        assert val == 28.0

    def test_point_in_time_lookup(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            "2020-01-15,AAPL,pe_ratio,25.0\n"
            "2020-04-15,AAPL,pe_ratio,22.5\n"
            "2020-07-15,AAPL,pe_ratio,28.0\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        # Query between first and second: should get first value
        val = mgr.get("AAPL", "pe_ratio", date(2020, 3, 1))
        assert val == 25.0

        # Query on exact date of second report
        val = mgr.get("AAPL", "pe_ratio", date(2020, 4, 15))
        assert val == 22.5

    def test_query_before_first_date_returns_none(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            "2020-06-01,AAPL,pe_ratio,25.0\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        val = mgr.get("AAPL", "pe_ratio", date(2020, 1, 1))
        assert val is None

    def test_unknown_symbol_returns_none(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            "2020-01-15,AAPL,pe_ratio,25.0\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        val = mgr.get("MSFT", "pe_ratio", date(2020, 12, 31))
        assert val is None

    def test_unknown_field_returns_none(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            "2020-01-15,AAPL,pe_ratio,25.0\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        val = mgr.get("AAPL", "revenue", date(2020, 12, 31))
        assert val is None

    def test_symbol_case_insensitive(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            "2020-01-15,aapl,pe_ratio,25.0\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        # Stored as "AAPL" (uppercased), queried as "aapl" (lowercased)
        val = mgr.get("aapl", "pe_ratio", date(2020, 12, 31))
        assert val == 25.0

    def test_multiple_symbols_and_fields(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            "2020-01-15,AAPL,pe_ratio,25.0\n"
            "2020-01-15,AAPL,revenue,100000.0\n"
            "2020-01-15,MSFT,pe_ratio,30.0\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        assert mgr.get("AAPL", "pe_ratio", date(2020, 12, 31)) == 25.0
        assert mgr.get("AAPL", "revenue", date(2020, 12, 31)) == 100000.0
        assert mgr.get("MSFT", "pe_ratio", date(2020, 12, 31)) == 30.0

    def test_invalid_value_skipped(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            "2020-01-15,AAPL,pe_ratio,not_a_number\n"
            "2020-04-15,AAPL,pe_ratio,22.5\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        # Invalid row skipped, valid row loaded
        val = mgr.get("AAPL", "pe_ratio", date(2020, 2, 1))
        assert val is None  # Only non-numeric row before this date

        val = mgr.get("AAPL", "pe_ratio", date(2020, 12, 31))
        assert val == 22.5

    def test_nonexistent_file(self, tmp_path):
        path = os.path.join(str(tmp_path), "does_not_exist.csv")
        mgr = FundamentalDataManager(path)

        val = mgr.get("AAPL", "pe_ratio", date(2020, 12, 31))
        assert val is None

    def test_empty_csv(self, tmp_path):
        csv_content = "date,symbol,field,value\n"
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        val = mgr.get("AAPL", "pe_ratio", date(2020, 12, 31))
        assert val is None

    def test_whitespace_handling(self, tmp_path):
        csv_content = (
            "date,symbol,field,value\n"
            " 2020-01-15 , aapl , pe_ratio ,25.0\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        val = mgr.get("AAPL", "pe_ratio", date(2020, 12, 31))
        assert val == 25.0

    def test_data_sorted_by_date(self, tmp_path):
        # Input intentionally out of order
        csv_content = (
            "date,symbol,field,value\n"
            "2020-07-15,AAPL,pe_ratio,28.0\n"
            "2020-01-15,AAPL,pe_ratio,25.0\n"
            "2020-04-15,AAPL,pe_ratio,22.5\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)

        # Despite out-of-order input, lookup should work correctly
        assert mgr.get("AAPL", "pe_ratio", date(2020, 3, 1)) == 25.0
        assert mgr.get("AAPL", "pe_ratio", date(2020, 5, 1)) == 22.5
        assert mgr.get("AAPL", "pe_ratio", date(2020, 8, 1)) == 28.0
