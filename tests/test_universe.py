"""Tests for UniverseProvider — all HTTP calls mocked."""

import json
import time
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from backtester.data.universe import UniverseProvider


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
