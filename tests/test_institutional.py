"""Unit tests for institutional (13F) data merging and derived metrics."""

import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.data.fundamental import EdgarDataManager
from backtester.data.fundamental_cache import EdgarCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_daily_df(start="2020-01-02", days=100, start_price=100.0):
    """Create a simple daily OHLCV DataFrame."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    close = np.linspace(start_price, start_price * 1.1, days)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(days, 1_000_000),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


def make_institutional_data(quarters):
    """Create synthetic 13F institutional holdings data.

    Each entry in quarters is a dict with:
        filed_date, report_date, total_holders, total_shares, total_value
    """
    return pd.DataFrame(quarters)


def _make_manager_with_institutional(symbol, inst_df):
    """Create EdgarDataManager with pre-cached institutional data."""
    tmpdir = tempfile.mkdtemp()
    cache = EdgarCache(tmpdir, "institutional")
    cache.save(symbol, inst_df)
    mgr = EdgarDataManager(
        cache_dir=tmpdir,
        use_edgar=False,
        enable_financials=False,
        enable_insider=False,
        enable_institutional=True,
        enable_events=False,
    )
    mgr._institutional_cache = cache
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInstitutionalMerge:
    """Test institutional holdings data merging."""

    def test_merge_creates_inst_columns(self):
        """After merge, DataFrame should contain inst_ prefixed columns."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
        ])
        daily = make_daily_df(start="2020-03-01", days=30)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        expected_cols = [
            "inst_holders_change_qoq",
            "inst_shares_change_pct",
            "inst_ownership_concentration",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_qoq_holders_change(self):
        """QoQ holders change = current - previous."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
            {"filed_date": date(2020, 5, 15), "report_date": date(2020, 3, 31),
             "total_holders": 120, "total_shares": 1_100_000, "total_value": 55_000_000},
        ])
        daily = make_daily_df(start="2020-06-01", days=30)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        change = result["inst_holders_change_qoq"].iloc[0]
        assert change == 20.0

    def test_qoq_shares_change_pct(self):
        """QoQ shares change pct = (current - prev) / prev."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
            {"filed_date": date(2020, 5, 15), "report_date": date(2020, 3, 31),
             "total_holders": 110, "total_shares": 1_100_000, "total_value": 55_000_000},
        ])
        daily = make_daily_df(start="2020-06-01", days=30)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        pct = result["inst_shares_change_pct"].iloc[0]
        assert pct == pytest.approx(0.10)

    def test_forward_fill_quarterly(self):
        """Institutional data fills forward from filing date to next filing."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
            {"filed_date": date(2020, 5, 15), "report_date": date(2020, 3, 31),
             "total_holders": 120, "total_shares": 1_200_000, "total_value": 60_000_000},
        ])
        # Daily data spans both filings
        daily = make_daily_df(start="2020-03-01", days=80)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        # Between Q1 and Q2 filings (after Feb 14, before May 15)
        mid = result.loc[(result.index >= pd.Timestamp(2020, 3, 1)) & (result.index < pd.Timestamp(2020, 5, 15))]
        if not mid.empty:
            # Should show Q1 data (NaN for QoQ since it's the first quarter)
            assert pd.isna(mid["inst_shares_change_pct"].iloc[0])

        # After Q2 filing, should show Q2 data
        after = result.loc[result.index >= pd.Timestamp(2020, 5, 15)]
        if not after.empty:
            assert after["inst_shares_change_pct"].iloc[0] == pytest.approx(0.20)

    def test_concentration_placeholder(self):
        """Concentration metric is NaN (placeholder for future implementation)."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
        ])
        daily = make_daily_df(start="2020-03-01", days=30)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert result["inst_ownership_concentration"].isna().all()

    def test_empty_institutional_data(self):
        """No 13F data -> DataFrame unchanged (no inst_ columns)."""
        mgr = EdgarDataManager(
            use_edgar=False, enable_financials=False,
            enable_insider=False, enable_institutional=True, enable_events=False,
        )
        daily = make_daily_df(days=10)
        result = mgr.merge_all_onto_daily("TEST", daily)

        inst_cols = [c for c in result.columns if c.startswith("inst_")]
        assert len(inst_cols) == 0

    def test_single_quarter_changes_nan(self):
        """First quarter of data -> QoQ changes are NaN."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
        ])
        daily = make_daily_df(start="2020-03-01", days=30)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert pd.isna(result["inst_holders_change_qoq"].iloc[0])
        assert pd.isna(result["inst_shares_change_pct"].iloc[0])

    def test_negative_shares_change(self):
        """Institutional selling -> negative shares change."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
            {"filed_date": date(2020, 5, 15), "report_date": date(2020, 3, 31),
             "total_holders": 90, "total_shares": 800_000, "total_value": 40_000_000},
        ])
        daily = make_daily_df(start="2020-06-01", days=10)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        change_pct = result["inst_shares_change_pct"].iloc[0]
        assert change_pct == pytest.approx(-0.20)
        assert result["inst_holders_change_qoq"].iloc[0] == -10.0

    def test_row_count_preserved(self):
        """Merge should not change the number of rows."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
        ])
        daily = make_daily_df(start="2020-03-01", days=50)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)
        assert len(result) == len(daily)

    def test_does_not_mutate_input(self):
        """Merge should not mutate the input DataFrame."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
        ])
        daily = make_daily_df(start="2020-03-01", days=10)
        original_cols = list(daily.columns)
        mgr = _make_manager_with_institutional("TEST", inst)
        mgr.merge_all_onto_daily("TEST", daily)
        assert list(daily.columns) == original_cols

    def test_three_quarters_change_uses_latest(self):
        """With 3 quarters, the most recent QoQ change is visible after the last filing."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
            {"filed_date": date(2020, 5, 15), "report_date": date(2020, 3, 31),
             "total_holders": 110, "total_shares": 1_100_000, "total_value": 55_000_000},
            {"filed_date": date(2020, 8, 14), "report_date": date(2020, 6, 30),
             "total_holders": 130, "total_shares": 1_300_000, "total_value": 65_000_000},
        ])
        daily = make_daily_df(start="2020-09-01", days=10)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        # Latest QoQ: 1300000 vs 1100000
        pct = result["inst_shares_change_pct"].iloc[0]
        expected = (1_300_000 - 1_100_000) / 1_100_000
        assert pct == pytest.approx(expected, rel=0.01)

    def test_point_in_time(self):
        """Data should only be visible after its filed_date."""
        inst = make_institutional_data([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
            {"filed_date": date(2020, 5, 15), "report_date": date(2020, 3, 31),
             "total_holders": 120, "total_shares": 1_200_000, "total_value": 60_000_000},
        ])
        daily = make_daily_df(start="2020-01-02", days=120)
        mgr = _make_manager_with_institutional("TEST", inst)
        result = mgr.merge_all_onto_daily("TEST", daily)

        # Before first filing (Feb 14), should have NaN for all inst_ columns
        before_q1 = result.loc[result.index < pd.Timestamp(2020, 2, 14)]
        if not before_q1.empty:
            assert before_q1["inst_holders_change_qoq"].isna().all()
            assert before_q1["inst_shares_change_pct"].isna().all()

        # Between Q1 and Q2 filings: Q1 data visible but QoQ is NaN (no prev quarter)
        between = result.loc[
            (result.index >= pd.Timestamp(2020, 2, 14))
            & (result.index < pd.Timestamp(2020, 5, 15))
        ]
        if not between.empty:
            assert between["inst_holders_change_qoq"].isna().all()

        # After Q2 filing: Q2 QoQ change should be visible and correct
        after_q2 = result.loc[result.index >= pd.Timestamp(2020, 5, 15)]
        if not after_q2.empty:
            assert after_q2["inst_holders_change_qoq"].iloc[0] == 20.0
            assert after_q2["inst_shares_change_pct"].iloc[0] == pytest.approx(0.20)
