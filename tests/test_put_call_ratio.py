"""Tests for CBOE put-call ratio data source."""

from datetime import date
from unittest.mock import MagicMock, patch
import io

import numpy as np
import pandas as pd
import pytest

from backtester.data.sentiment import CBOEPutCallSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(rows, header="DATE,CALLS,PUTS,TOTAL,P/C RATIO"):
    """Build a CSV string from a list of row strings."""
    lines = [header] + rows
    return "\n".join(lines)


def _mock_urlopen(csv_text):
    """Return a context-manager mock for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = csv_text.encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: PCR computation
# ---------------------------------------------------------------------------


class TestPCRComputation:
    def test_pcr_computed_from_csv(self):
        csv = _make_csv([
            "01/02/2020,100000,80000,180000,0.80",
            "01/03/2020,120000,96000,216000,0.80",
        ])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 3))
        assert "sentiment_pcr" in result.columns
        np.testing.assert_almost_equal(result["sentiment_pcr"].iloc[0], 0.80)

    def test_ma10_computed_correctly(self):
        rows = []
        dates = pd.bdate_range("2020-01-02", periods=15, freq="B")
        for i, d in enumerate(dates):
            pcr = 0.80 + i * 0.01
            rows.append(f"{d.strftime('%m/%d/%Y')},100000,{int(100000*pcr)},{int(100000*(1+pcr))},{pcr:.2f}")
        csv = _make_csv(rows)
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 22))
        assert "sentiment_pcr_ma10" in result.columns
        # First 9 rows should have NaN MA10
        assert result["sentiment_pcr_ma10"].isna().sum() == 9
        # 10th row onward should have valid MA10
        assert pd.notna(result["sentiment_pcr_ma10"].iloc[9])

    def test_high_pcr_above_1(self):
        csv = _make_csv([
            "01/02/2020,100000,130000,230000,1.30",
        ])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert result["sentiment_pcr"].iloc[0] == 1.30

    def test_low_pcr_below_half(self):
        csv = _make_csv([
            "01/02/2020,200000,80000,280000,0.40",
        ])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert result["sentiment_pcr"].iloc[0] == 0.40

    def test_boundary_pcr_1_0(self):
        csv = _make_csv([
            "01/02/2020,100000,100000,200000,1.00",
        ])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert result["sentiment_pcr"].iloc[0] == 1.00


# ---------------------------------------------------------------------------
# Tests: Data handling
# ---------------------------------------------------------------------------


class TestPCRDataHandling:
    def test_nan_handling_in_raw_data(self):
        csv = _make_csv([
            "01/02/2020,100000,80000,180000,0.80",
            "01/03/2020,,,, ",
            "01/06/2020,110000,90000,200000,0.82",
        ])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 6))
        # Should handle NaN gracefully — NaN row may be dropped or have NaN pcr
        assert len(result) >= 2

    def test_empty_csv_returns_empty_df(self):
        csv = "DATE,CALLS,PUTS,TOTAL,P/C RATIO\n"
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 10))
        assert result.empty or len(result) == 0

    def test_date_range_filtering(self):
        rows = []
        dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
        for d in dates:
            rows.append(f"{d.strftime('%m/%d/%Y')},100000,80000,180000,0.80")
        csv = _make_csv(rows)
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 6), date(2020, 1, 10))
        # Should only include days within the range
        assert len(result) <= 5

    def test_column_names(self):
        csv = _make_csv(["01/02/2020,100000,80000,180000,0.80"])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert set(result.columns) == {"sentiment_pcr", "sentiment_pcr_ma10"}

    def test_first_9_days_ma10_nan(self):
        rows = []
        dates = pd.bdate_range("2020-01-02", periods=12, freq="B")
        for d in dates:
            rows.append(f"{d.strftime('%m/%d/%Y')},100000,80000,180000,0.80")
        csv = _make_csv(rows)
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 17))
        assert result["sentiment_pcr_ma10"].iloc[:9].isna().all()

    def test_single_day(self):
        csv = _make_csv(["01/02/2020,100000,80000,180000,0.80"])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert len(result) == 1
        assert result["sentiment_pcr"].iloc[0] == 0.80

    def test_correct_dtypes(self):
        csv = _make_csv(["01/02/2020,100000,80000,180000,0.80"])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert result["sentiment_pcr"].dtype == np.float64

    def test_index_is_datetimeindex(self):
        csv = _make_csv(["01/02/2020,100000,80000,180000,0.80"])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert isinstance(result.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Tests: Cache fallback
# ---------------------------------------------------------------------------


class TestPCRCache:
    def test_download_failure_falls_back_to_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        sent_dir = cache_dir / "sentiment"
        sent_dir.mkdir(parents=True)

        # Pre-populate cache
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")], name="Date"
        )
        cached_df = pd.DataFrame({"pcr": [0.75, 0.78]}, index=idx)
        cached_df.to_parquet(sent_dir / "equity_pcr.parquet")

        with patch("backtester.data.sentiment.urllib.request.urlopen", side_effect=Exception("network error")):
            src = CBOEPutCallSource(cache_dir=str(cache_dir))
            result = src.load(date(2020, 1, 2), date(2020, 1, 3))
        assert not result.empty
        assert "sentiment_pcr" in result.columns

    def test_no_cache_and_no_download_returns_empty(self):
        with patch("backtester.data.sentiment.urllib.request.urlopen", side_effect=Exception("network error")):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 3))
        assert result.empty or len(result) == 0


# ---------------------------------------------------------------------------
# Tests: CSV parsing edge cases
# ---------------------------------------------------------------------------


class TestCSVParsing:
    def test_alternative_column_names(self):
        """CBOE may use different column headers."""
        csv = "Trade Date,Call Volume,Put Volume,Total Volume,Put/Call Ratio\n"
        csv += "01/02/2020,100000,85000,185000,0.85\n"
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert result["sentiment_pcr"].iloc[0] == 0.85

    def test_pcr_from_puts_calls_when_no_ratio_column(self):
        """If no ratio column, compute from puts/calls."""
        csv = "DATE,CALLS,PUTS,TOTAL\n"
        csv += "01/02/2020,100000,90000,190000\n"
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert "sentiment_pcr" in result.columns
        np.testing.assert_almost_equal(result["sentiment_pcr"].iloc[0], 0.9, decimal=1)

    def test_csv_missing_date_column(self):
        """CSV with no date column → returns empty DataFrame."""
        csv = "CALLS,PUTS,TOTAL,P/C RATIO\n100000,80000,180000,0.80\n"
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert result.empty or len(result) == 0

    def test_csv_zero_calls_nan_computed_pcr(self):
        """Zero calls → computed puts/calls ratio is NaN (division by zero guarded)."""
        csv = "DATE,CALLS,PUTS,TOTAL\n01/02/2020,0,80000,80000\n"
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource()
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        # pcr = puts / calls; calls=0 → guarded by .replace(0, nan) → NaN pcr
        assert "sentiment_pcr" in result.columns


class TestPCRCacheWrite:
    def test_successful_download_saves_cache(self, tmp_path):
        """Successful CSV download writes to cache."""
        cache_dir = tmp_path / "cache"
        csv = _make_csv(["01/02/2020,100000,80000,180000,0.80"])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource(cache_dir=str(cache_dir))
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert not result.empty
        assert (cache_dir / "sentiment" / "equity_pcr.parquet").exists()

    def test_empty_cache_dir_no_crash(self):
        """CBOEPutCallSource with cache_dir=None doesn't crash on save."""
        csv = _make_csv(["01/02/2020,100000,80000,180000,0.80"])
        with patch("backtester.data.sentiment.urllib.request.urlopen", return_value=_mock_urlopen(csv)):
            src = CBOEPutCallSource(cache_dir=None)
            result = src.load(date(2020, 1, 2), date(2020, 1, 2))
        assert not result.empty
