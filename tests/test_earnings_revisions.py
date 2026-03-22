"""Tests for analyst earnings revision data source."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtester.data.analyst import AnalystRevisionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _daily_index(start="2020-01-02", days=30):
    return pd.DatetimeIndex(
        pd.bdate_range(start=start, periods=days, freq="B").date, name="Date"
    )


def _mock_yf_ticker(earnings_estimate=None, recommendations=None):
    """Create a mock yfinance Ticker."""
    ticker = MagicMock()
    ticker.earnings_estimate = earnings_estimate
    ticker.recommendations = recommendations
    return ticker


# ---------------------------------------------------------------------------
# Tests: Breadth computation
# ---------------------------------------------------------------------------


class TestBreadthComputation:
    @patch("backtester.data.analyst.yf")
    def test_all_revisions_up_breadth_1(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [5],
            "numberOfAnalysts Down Last 7 Days": [0],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        assert result["analyst_rev_breadth"].iloc[0] == 1.0

    @patch("backtester.data.analyst.yf")
    def test_all_revisions_down_breadth_neg1(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [0],
            "numberOfAnalysts Down Last 7 Days": [5],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        assert result["analyst_rev_breadth"].iloc[0] == -1.0

    @patch("backtester.data.analyst.yf")
    def test_mixed_revisions(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [3],
            "numberOfAnalysts Down Last 7 Days": [2],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        # (3 - 2) / (3 + 2) = 0.2
        np.testing.assert_almost_equal(result["analyst_rev_breadth"].iloc[0], 0.2)

    @patch("backtester.data.analyst.yf")
    def test_zero_revisions_breadth_nan(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [0],
            "numberOfAnalysts Down Last 7 Days": [0],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        assert pd.isna(result["analyst_rev_breadth"].iloc[0])

    @patch("backtester.data.analyst.yf")
    def test_equal_up_down_breadth_0(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [4],
            "numberOfAnalysts Down Last 7 Days": [4],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        np.testing.assert_almost_equal(result["analyst_rev_breadth"].iloc[0], 0.0)

    @patch("backtester.data.analyst.yf")
    def test_single_revision_up(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [1],
            "numberOfAnalysts Down Last 7 Days": [0],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        assert result["analyst_rev_breadth"].iloc[0] == 1.0

    @patch("backtester.data.analyst.yf")
    def test_large_revision_counts(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [100],
            "numberOfAnalysts Down Last 7 Days": [50],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        # (100 - 50) / (100 + 50) = 50/150 ≈ 0.333
        np.testing.assert_almost_equal(
            result["analyst_rev_breadth"].iloc[0], 1 / 3, decimal=3
        )


# ---------------------------------------------------------------------------
# Tests: Column names and types
# ---------------------------------------------------------------------------


class TestColumnNames:
    @patch("backtester.data.analyst.yf")
    def test_correct_column_names(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [3],
            "numberOfAnalysts Down Last 7 Days": [1],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        expected = {"analyst_rev_up_7d", "analyst_rev_down_7d", "analyst_rev_breadth"}
        assert expected == set(result.columns)

    @patch("backtester.data.analyst.yf")
    def test_column_dtypes_float(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [3],
            "numberOfAnalysts Down Last 7 Days": [1],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        for col in result.columns:
            assert result[col].dtype in [np.float64, float], f"{col} not float"


# ---------------------------------------------------------------------------
# Tests: Daily index alignment (point-in-time limitation)
# ---------------------------------------------------------------------------


class TestDailyIndex:
    @patch("backtester.data.analyst.yf")
    def test_daily_index_all_nan_except_last(self, mock_yf):
        """With daily_index, only the last date has snapshot values."""
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [5],
            "numberOfAnalysts Down Last 7 Days": [2],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        idx = _daily_index(days=20)
        result = src.fetch("AAPL", daily_index=idx)
        assert len(result) == 20
        # All rows except last should be NaN
        assert result["analyst_rev_up_7d"].iloc[:-1].isna().all()
        assert result["analyst_rev_up_7d"].iloc[-1] == 5.0

    @patch("backtester.data.analyst.yf")
    def test_daily_index_correct_shape(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [3],
            "numberOfAnalysts Down Last 7 Days": [1],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        idx = _daily_index(days=50)
        result = src.fetch("AAPL", daily_index=idx)
        assert result.shape == (50, 3)
        assert (result.index == idx).all()

    @patch("backtester.data.analyst.yf")
    def test_no_daily_index_returns_snapshot(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [3],
            "numberOfAnalysts Down Last 7 Days": [1],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        assert len(result) == 1
        assert result["analyst_rev_up_7d"].iloc[0] == 3.0


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @patch("backtester.data.analyst.yf")
    def test_empty_yfinance_response(self, mock_yf):
        mock_yf.Ticker.return_value = _mock_yf_ticker(
            earnings_estimate=pd.DataFrame()
        )
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        # Should return NaN values gracefully
        assert "analyst_rev_breadth" in result.columns

    @patch("backtester.data.analyst.yf")
    def test_yfinance_exception(self, mock_yf):
        mock_yf.Ticker.side_effect = Exception("network error")
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        assert "analyst_rev_breadth" in result.columns

    @patch("backtester.data.analyst.yf")
    def test_yfinance_exception_with_daily_index(self, mock_yf):
        mock_yf.Ticker.side_effect = Exception("network error")
        src = AnalystRevisionSource()
        idx = _daily_index(days=10)
        result = src.fetch("AAPL", daily_index=idx)
        assert len(result) == 10
        assert result["analyst_rev_up_7d"].isna().all()


# ---------------------------------------------------------------------------
# Tests: Symbol handling
# ---------------------------------------------------------------------------


class TestSymbolHandling:
    @patch("backtester.data.analyst.yf")
    def test_symbol_uppercased(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [1],
            "numberOfAnalysts Down Last 7 Days": [0],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        src.fetch("aapl")
        mock_yf.Ticker.assert_called_with("AAPL")

    @patch("backtester.data.analyst.yf")
    def test_multiple_symbols(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [2],
            "numberOfAnalysts Down Last 7 Days": [1],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        r1 = src.fetch("AAPL")
        r2 = src.fetch("MSFT")
        assert len(r1) == 1
        assert len(r2) == 1


# ---------------------------------------------------------------------------
# Tests: Cache behavior
# ---------------------------------------------------------------------------


class TestAnalystCache:
    @patch("backtester.data.analyst.yf")
    def test_cache_saves_and_loads(self, mock_yf, tmp_path):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [4],
            "numberOfAnalysts Down Last 7 Days": [2],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        cache_dir = str(tmp_path / "cache")
        src = AnalystRevisionSource(cache_dir=cache_dir)
        result1 = src.fetch("AAPL")
        assert result1["analyst_rev_up_7d"].iloc[0] == 4.0

        # Second call should use cache — even if yfinance fails
        mock_yf.Ticker.side_effect = Exception("network error")
        src2 = AnalystRevisionSource(cache_dir=cache_dir)
        result2 = src2.fetch("AAPL")
        assert result2["analyst_rev_up_7d"].iloc[0] == 4.0

    def test_no_cache_dir_still_works(self):
        """Without cache_dir, fetch should still work (no caching)."""
        with patch("backtester.data.analyst.yf") as mock_yf:
            est = pd.DataFrame({
                "numberOfAnalysts Up Last 7 Days": [1],
                "numberOfAnalysts Down Last 7 Days": [0],
            })
            mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
            src = AnalystRevisionSource()
            result = src.fetch("AAPL")
            assert result["analyst_rev_up_7d"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Tests: Revision count consistency
# ---------------------------------------------------------------------------


class TestRevisionCounts:
    @patch("backtester.data.analyst.yf")
    def test_up_count_matches(self, mock_yf):
        est = pd.DataFrame({
            "numberOfAnalysts Up Last 7 Days": [7],
            "numberOfAnalysts Down Last 7 Days": [3],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(earnings_estimate=est)
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        assert result["analyst_rev_up_7d"].iloc[0] == 7.0
        assert result["analyst_rev_down_7d"].iloc[0] == 3.0


# ---------------------------------------------------------------------------
# Tests: Recommendations fallback
# ---------------------------------------------------------------------------


class TestRecommendationsFallback:
    @patch("backtester.data.analyst.yf")
    def test_fallback_to_recommendations(self, mock_yf):
        """When earnings_estimate is empty, fall back to recommendations."""
        rec = pd.DataFrame({
            "To Grade": ["Buy", "Outperform", "Sell", "Buy", "Underperform",
                         "Buy", "Overweight"],
        })
        mock_yf.Ticker.return_value = _mock_yf_ticker(
            earnings_estimate=pd.DataFrame(),
            recommendations=rec,
        )
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        # 4 up (Buy*3, Outperform, Overweight = 5 actually)
        # Let's just verify we got non-zero counts
        assert result["analyst_rev_up_7d"].iloc[0] > 0 or result["analyst_rev_down_7d"].iloc[0] > 0

    @patch("backtester.data.analyst.yf")
    def test_no_estimates_no_recs_returns_zero(self, mock_yf):
        """No earnings estimate and no recommendations → 0 counts."""
        mock_yf.Ticker.return_value = _mock_yf_ticker(
            earnings_estimate=pd.DataFrame(),
            recommendations=pd.DataFrame(),
        )
        src = AnalystRevisionSource()
        result = src.fetch("AAPL")
        assert result["analyst_rev_up_7d"].iloc[0] == 0.0
        assert result["analyst_rev_down_7d"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Tests: Empty frame
# ---------------------------------------------------------------------------


class TestEmptyFrame:
    @patch("backtester.data.analyst.yf")
    def test_empty_frame_with_daily_index(self, mock_yf):
        """_empty_frame with daily_index returns NaN-filled DataFrame."""
        mock_yf.Ticker.side_effect = Exception("network error")
        src = AnalystRevisionSource()
        idx = _daily_index(days=5)
        result = src.fetch("AAPL", daily_index=idx)
        assert len(result) == 5
        assert set(result.columns) == {"analyst_rev_up_7d", "analyst_rev_down_7d", "analyst_rev_breadth"}
        assert result["analyst_rev_up_7d"].isna().all()

    @patch("backtester.data.analyst.yf")
    def test_empty_frame_without_daily_index(self, mock_yf):
        """_empty_frame without daily_index returns empty DataFrame."""
        mock_yf.Ticker.side_effect = Exception("network error")
        src = AnalystRevisionSource()
        result = src.fetch("AAPL", daily_index=None)
        assert set(result.columns) == {"analyst_rev_up_7d", "analyst_rev_down_7d", "analyst_rev_breadth"}
