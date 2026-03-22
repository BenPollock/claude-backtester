"""Tests for UniverseProvider and HistoricalUniverse — all HTTP calls mocked."""

import csv
import json
import time
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from backtester.data.universe import UniverseProvider, HistoricalUniverse


@pytest.fixture
def provider(tmp_path):
    """UniverseProvider with a temporary cache directory."""
    return UniverseProvider(cache_dir=str(tmp_path / "universe_cache"))


@pytest.fixture
def sp500_table():
    """Mock S&P 500 Wikipedia table."""
    return pd.DataFrame({"Symbol": ["AAPL", "MSFT", "GOOG", "BRK.B"]})


@pytest.fixture
def tsx_table():
    """Mock TSX Composite Wikipedia table."""
    return pd.DataFrame({"Symbol": ["RY", "TD", "BNS"]})


class TestCaching:
    def test_cache_write_and_read(self, provider, sp500_table):
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", return_value=[sp500_table]):
            first = provider.get_tickers(market="us", universe="index")

        # Second call should use cache (no read_html call)
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua") as mock_html:
            second = provider.get_tickers(market="us", universe="index")
            mock_html.assert_not_called()

        assert first == second

    def test_cache_staleness(self, provider, sp500_table, tmp_path):
        cache_path = tmp_path / "universe_cache" / "us_index.json"

        # Seed cache
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", return_value=[sp500_table]):
            provider.get_tickers(market="us", universe="index")

        assert cache_path.exists()

        # Make cache appear old (>7 days)
        old_time = time.time() - 8 * 24 * 60 * 60
        import os
        os.utime(cache_path, (old_time, old_time))

        # Should refetch
        updated_table = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "GOOG", "NVDA"]})
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", return_value=[updated_table]):
            result = provider.get_tickers(market="us", universe="index")

        assert "NVDA" in result


class TestFetching:
    def test_sp500_parse(self, provider, sp500_table):
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", return_value=[sp500_table]):
            result = provider.get_tickers(market="us", universe="index")
        # BRK.B → BRK-B for yfinance
        assert "BRK-B" in result
        assert "AAPL" in result

    def test_tsx_appends_to_suffix(self, provider, tsx_table):
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", return_value=[tsx_table]):
            result = provider.get_tickers(market="ca", universe="index")
        assert all(t.endswith(".TO") for t in result)
        assert "RY.TO" in result

    def test_us_ca_combines(self, provider, sp500_table, tsx_table):
        def mock_read_html(url):
            if "S%26P_500" in url:
                return [sp500_table]
            return [tsx_table]

        with patch.object(UniverseProvider, "_read_html_with_ua", side_effect=mock_read_html):
            result = provider.get_tickers(market="us_ca", universe="index")

        assert "AAPL" in result
        assert "RY.TO" in result
        # Should be sorted and deduplicated
        assert result == sorted(set(result))


class TestFallback:
    def test_stale_cache_fallback_on_error(self, provider, sp500_table, tmp_path):
        cache_path = tmp_path / "universe_cache" / "us_index.json"

        # Seed cache
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", return_value=[sp500_table]):
            original = provider.get_tickers(market="us", universe="index")

        # Make cache stale
        old_time = time.time() - 8 * 24 * 60 * 60
        import os
        os.utime(cache_path, (old_time, old_time))

        # Fetch fails → should fall back to stale cache
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", side_effect=Exception("network error")):
            result = provider.get_tickers(market="us", universe="index")

        assert result == original

    def test_no_cache_no_network_raises(self, provider):
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", side_effect=Exception("network error")):
            with pytest.raises(RuntimeError, match="no cached data"):
                provider.get_tickers(market="us", universe="index")


class TestUniverseProviderEdgeCases:
    def test_invalid_market_raises(self, provider):
        with pytest.raises(ValueError, match="Unknown market"):
            provider.get_tickers(market="jp", universe="index")

    def test_universe_all_falls_back_to_index(self, provider, sp500_table):
        """universe='all' warns and falls back to 'index'."""
        with patch("backtester.data.universe.UniverseProvider._read_html_with_ua", return_value=[sp500_table]):
            result = provider.get_tickers(market="us", universe="all")
        assert "AAPL" in result


# ---------------------------------------------------------------------------
# HistoricalUniverse tests
# ---------------------------------------------------------------------------

class TestHistoricalUniverse:
    """Tests for point-in-time universe membership via CSV."""

    def _write_csv(self, path, rows):
        """Write a CSV with 'date' and 'symbol' columns."""
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "symbol"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_members_on_exact_date(self, tmp_path):
        csv_path = tmp_path / "universe.csv"
        self._write_csv(csv_path, [
            {"date": "2020-01-01", "symbol": "AAPL"},
            {"date": "2020-01-01", "symbol": "MSFT"},
            {"date": "2020-06-01", "symbol": "AAPL"},
            {"date": "2020-06-01", "symbol": "GOOG"},
        ])
        hu = HistoricalUniverse(str(csv_path))
        members = hu.members_on(date(2020, 1, 1))
        assert members == {"AAPL", "MSFT"}

    def test_members_on_between_snapshots(self, tmp_path):
        """Query date between snapshots returns the earlier snapshot."""
        csv_path = tmp_path / "universe.csv"
        self._write_csv(csv_path, [
            {"date": "2020-01-01", "symbol": "AAPL"},
            {"date": "2020-06-01", "symbol": "GOOG"},
        ])
        hu = HistoricalUniverse(str(csv_path))
        members = hu.members_on(date(2020, 3, 15))
        assert members == {"AAPL"}

    def test_members_on_after_last_snapshot(self, tmp_path):
        """Query date after last snapshot returns the last snapshot."""
        csv_path = tmp_path / "universe.csv"
        self._write_csv(csv_path, [
            {"date": "2020-01-01", "symbol": "AAPL"},
        ])
        hu = HistoricalUniverse(str(csv_path))
        members = hu.members_on(date(2025, 12, 31))
        assert members == {"AAPL"}

    def test_members_on_before_first_snapshot(self, tmp_path):
        """Query date before first snapshot returns None."""
        csv_path = tmp_path / "universe.csv"
        self._write_csv(csv_path, [
            {"date": "2020-06-01", "symbol": "AAPL"},
        ])
        hu = HistoricalUniverse(str(csv_path))
        assert hu.members_on(date(2019, 1, 1)) is None

    def test_empty_csv_returns_none(self, tmp_path):
        """Empty CSV (header only) → members_on returns None."""
        csv_path = tmp_path / "universe.csv"
        self._write_csv(csv_path, [])
        hu = HistoricalUniverse(str(csv_path))
        assert hu.members_on(date(2020, 1, 1)) is None

    def test_all_symbols_property(self, tmp_path):
        csv_path = tmp_path / "universe.csv"
        self._write_csv(csv_path, [
            {"date": "2020-01-01", "symbol": "AAPL"},
            {"date": "2020-01-01", "symbol": "MSFT"},
            {"date": "2020-06-01", "symbol": "GOOG"},
        ])
        hu = HistoricalUniverse(str(csv_path))
        assert hu.all_symbols == {"AAPL", "MSFT", "GOOG"}

    def test_symbols_uppercased(self, tmp_path):
        """Symbols are uppercased on load."""
        csv_path = tmp_path / "universe.csv"
        self._write_csv(csv_path, [
            {"date": "2020-01-01", "symbol": "aapl"},
            {"date": "2020-01-01", "symbol": " msft "},
        ])
        hu = HistoricalUniverse(str(csv_path))
        members = hu.members_on(date(2020, 1, 1))
        assert "AAPL" in members
        assert "MSFT" in members

    def test_multiple_snapshots_binary_search(self, tmp_path):
        """Binary search correctly finds the right snapshot among many."""
        csv_path = tmp_path / "universe.csv"
        rows = []
        for month in range(1, 13):
            d = f"2020-{month:02d}-01"
            rows.append({"date": d, "symbol": f"SYM_{month}"})
        self._write_csv(csv_path, rows)
        hu = HistoricalUniverse(str(csv_path))

        # Query mid-July → should get July snapshot
        members = hu.members_on(date(2020, 7, 15))
        assert members == {"SYM_7"}

        # Query end of December → should get December snapshot
        members = hu.members_on(date(2020, 12, 31))
        assert members == {"SYM_12"}
