"""Unit tests for institutional (13F) data merging and derived metrics."""

import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtester.data.fundamental import EdgarDataManager
from backtester.data.fundamental_cache import EdgarCache
from backtester.data.edgar_institutional import (
    EdgarInstitutionalSource,
    _COLUMNS,
    _DEFAULT_MANAGERS,
)


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


# ---------------------------------------------------------------------------
# Fetch test helpers
# ---------------------------------------------------------------------------


def _make_mock_holdings_df(entries):
    """Create a mock 13F holdings DataFrame.

    entries: list of dicts with keys like nameOfIssuer, ticker, shares, value.
    If no 'ticker' key is provided, one is NOT added (to test name-based matching).
    """
    return pd.DataFrame(entries)


def _make_mock_13f_filing(filed_date, report_date, holdings_df):
    """Create a mock 13F filing object."""
    filing = MagicMock()
    filing.filing_date = filed_date
    filing.report_date = report_date

    parsed = MagicMock()
    # Make the infotable attribute return our DataFrame
    parsed.infotable = holdings_df
    parsed.holdings = holdings_df
    # Disable to_dataframe fallback so _extract_holdings picks up infotable
    parsed.data = None
    del parsed.to_dataframe
    filing.obj.return_value = parsed
    return filing


def _make_mock_company(filings):
    """Create a mock Company that returns the given filings."""
    company = MagicMock()
    company.get_filings.return_value = filings
    return company


# ---------------------------------------------------------------------------
# Fetch tests
# ---------------------------------------------------------------------------


class TestEdgarInstitutionalFetch:
    """Tests for EdgarInstitutionalSource.fetch() with mocked edgartools."""

    @patch("backtester.data.edgar_institutional.Company")
    def test_finds_holdings_across_managers(self, mock_company_cls):
        """Multiple managers holding same stock -> aggregated correctly."""
        # Manager A holds 1000 shares of AAPL worth 150k
        holdings_a = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 1000, "value": 150_000},
        ])
        filing_a = _make_mock_13f_filing(
            date(2024, 2, 14), date(2023, 12, 31), holdings_a
        )

        # Manager B holds 2000 shares of AAPL worth 300k
        holdings_b = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 2000, "value": 300_000},
        ])
        filing_b = _make_mock_13f_filing(
            date(2024, 2, 15), date(2023, 12, 31), holdings_b
        )

        managers = {"111": "Manager A", "222": "Manager B"}

        def side_effect(cik):
            if cik == "111":
                return _make_mock_company([filing_a])
            elif cik == "222":
                return _make_mock_company([filing_b])
            return _make_mock_company([])

        mock_company_cls.side_effect = side_effect

        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        source.user_agent = "test"
        source.max_filings = 50
        source._managers = managers
        source._filings_per_manager = 4

        result = source.fetch("AAPL")

        assert not result.empty
        # Both managers hold AAPL for same report_date, so aggregated
        row = result[result["report_date"] == date(2023, 12, 31)]
        assert len(row) == 1
        assert row.iloc[0]["total_holders"] == 2
        assert row.iloc[0]["total_shares"] == 3000
        assert row.iloc[0]["total_value"] == 450_000

    @patch("backtester.data.edgar_institutional.Company")
    def test_empty_when_no_manager_holds_symbol(self, mock_company_cls):
        """No managers hold the target symbol -> empty DataFrame."""
        # Manager holds MSFT only
        holdings = _make_mock_holdings_df([
            {"nameOfIssuer": "MICROSOFT CORP", "ticker": "MSFT", "shares": 5000, "value": 500_000},
        ])
        filing = _make_mock_13f_filing(
            date(2024, 2, 14), date(2023, 12, 31), holdings
        )

        mock_company_cls.return_value = _make_mock_company([filing])

        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        source.user_agent = "test"
        source.max_filings = 50
        source._managers = {"111": "Manager A"}
        source._filings_per_manager = 4

        result = source.fetch("AAPL")

        assert result.empty
        assert list(result.columns) == _COLUMNS

    @patch("backtester.data.edgar_institutional.Company")
    def test_aggregates_by_report_date(self, mock_company_cls):
        """Holdings from same quarter aggregated into one row."""
        # Two managers, same report_date
        holdings_a = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 1000, "value": 100_000},
        ])
        holdings_b = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 3000, "value": 300_000},
        ])

        filing_a = _make_mock_13f_filing(
            date(2024, 2, 14), date(2023, 12, 31), holdings_a
        )
        filing_b = _make_mock_13f_filing(
            date(2024, 2, 15), date(2023, 12, 31), holdings_b
        )

        managers = {"111": "Manager A", "222": "Manager B"}

        def side_effect(cik):
            if cik == "111":
                return _make_mock_company([filing_a])
            elif cik == "222":
                return _make_mock_company([filing_b])
            return _make_mock_company([])

        mock_company_cls.side_effect = side_effect

        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        source.user_agent = "test"
        source.max_filings = 50
        source._managers = managers
        source._filings_per_manager = 4

        result = source.fetch("AAPL")

        # Should be one row for the one report_date
        assert len(result) == 1
        assert result.iloc[0]["total_shares"] == 4000
        assert result.iloc[0]["total_value"] == 400_000

    @patch("backtester.data.edgar_institutional.Company")
    def test_total_holders_counts_unique_managers(self, mock_company_cls):
        """total_holders = number of unique managers holding the stock."""
        holdings = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 500, "value": 50_000},
        ])

        managers = {}
        companies = {}
        for i in range(5):
            cik = str(1000 + i)
            managers[cik] = f"Manager {i}"
            filing = _make_mock_13f_filing(
                date(2024, 2, 14 + i), date(2023, 12, 31), holdings
            )
            companies[cik] = _make_mock_company([filing])

        def side_effect(cik):
            return companies.get(cik, _make_mock_company([]))

        mock_company_cls.side_effect = side_effect

        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        source.user_agent = "test"
        source.max_filings = 50
        source._managers = managers
        source._filings_per_manager = 4

        result = source.fetch("AAPL")

        assert len(result) == 1
        assert result.iloc[0]["total_holders"] == 5

    @patch("backtester.data.edgar_institutional.Company")
    def test_custom_manager_list(self, mock_company_cls):
        """Constructor accepts custom manager list."""
        custom_managers = {"999999": "Custom Fund"}

        holdings = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 100, "value": 10_000},
        ])
        filing = _make_mock_13f_filing(
            date(2024, 2, 14), date(2023, 12, 31), holdings
        )
        mock_company_cls.return_value = _make_mock_company([filing])

        source = EdgarInstitutionalSource(
            user_agent="test",
            managers=custom_managers,
            filings_per_manager=2,
        )
        assert source._managers == custom_managers
        assert source._filings_per_manager == 2

        result = source.fetch("AAPL")
        assert not result.empty
        assert result.iloc[0]["total_holders"] == 1

    @patch("backtester.data.edgar_institutional.Company")
    def test_output_format_matches_columns(self, mock_company_cls):
        """Output has exactly the expected columns."""
        holdings = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 100, "value": 10_000},
        ])
        filing = _make_mock_13f_filing(
            date(2024, 2, 14), date(2023, 12, 31), holdings
        )
        mock_company_cls.return_value = _make_mock_company([filing])

        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        source.user_agent = "test"
        source.max_filings = 50
        source._managers = {"111": "Manager A"}
        source._filings_per_manager = 4

        result = source.fetch("AAPL")

        assert list(result.columns) == _COLUMNS

    @patch("backtester.data.edgar_institutional.Company")
    def test_handles_manager_fetch_failure_gracefully(self, mock_company_cls):
        """If one manager's fetch fails, others still work."""
        holdings = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 500, "value": 50_000},
        ])
        filing = _make_mock_13f_filing(
            date(2024, 2, 14), date(2023, 12, 31), holdings
        )

        # Manager A raises a non-rate-limit exception
        bad_company = MagicMock()
        bad_company.get_filings.side_effect = ValueError("network error")

        good_company = _make_mock_company([filing])

        def side_effect(cik):
            if cik == "111":
                return bad_company
            return good_company

        mock_company_cls.side_effect = side_effect

        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        source.user_agent = "test"
        source.max_filings = 50
        source._managers = {"111": "Bad Manager", "222": "Good Manager"}
        source._filings_per_manager = 4

        result = source.fetch("AAPL")

        # Should still have data from the good manager
        assert not result.empty
        assert result.iloc[0]["total_holders"] == 1
        assert result.iloc[0]["total_shares"] == 500

    @patch("backtester.data.edgar_institutional.Company")
    def test_filter_by_ticker_column(self, mock_company_cls):
        """Holdings with a ticker column use exact match."""
        holdings = _make_mock_holdings_df([
            {"ticker": "AAPL", "nameOfIssuer": "APPLE INC",
             "shares": 1000, "value": 150_000},
            {"ticker": "MSFT", "nameOfIssuer": "MICROSOFT CORP",
             "shares": 2000, "value": 300_000},
        ])
        filing = _make_mock_13f_filing(
            date(2024, 2, 14), date(2023, 12, 31), holdings
        )
        mock_company_cls.return_value = _make_mock_company([filing])

        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        source.user_agent = "test"
        source.max_filings = 50
        source._managers = {"111": "Manager A"}
        source._filings_per_manager = 4

        result = source.fetch("AAPL")

        assert result.iloc[0]["total_shares"] == 1000
        assert result.iloc[0]["total_value"] == 150_000

    @patch("backtester.data.edgar_institutional.Company")
    def test_multiple_quarters(self, mock_company_cls):
        """Multiple filing periods produce multiple output rows."""
        holdings_q4 = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 1000, "value": 100_000},
        ])
        holdings_q1 = _make_mock_holdings_df([
            {"nameOfIssuer": "APPLE INC", "ticker": "AAPL", "shares": 1500, "value": 160_000},
        ])

        filing_q4 = _make_mock_13f_filing(
            date(2024, 2, 14), date(2023, 12, 31), holdings_q4
        )
        filing_q1 = _make_mock_13f_filing(
            date(2024, 5, 15), date(2024, 3, 31), holdings_q1
        )
        mock_company_cls.return_value = _make_mock_company(
            [filing_q1, filing_q4]
        )

        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        source.user_agent = "test"
        source.max_filings = 50
        source._managers = {"111": "Manager A"}
        source._filings_per_manager = 4

        result = source.fetch("AAPL")

        assert len(result) == 2
        # Sorted by filed_date
        assert result.iloc[0]["report_date"] == date(2023, 12, 31)
        assert result.iloc[1]["report_date"] == date(2024, 3, 31)

    def test_default_managers_populated(self):
        """_DEFAULT_MANAGERS has a reasonable set of institutional investors."""
        assert len(_DEFAULT_MANAGERS) >= 15
        # Spot-check a well-known manager
        assert "Berkshire Hathaway" in _DEFAULT_MANAGERS.values()
        assert "BlackRock" in _DEFAULT_MANAGERS.values()

    def test_empty_df_has_correct_columns(self):
        """_empty_df() returns DataFrame with correct columns."""
        source = EdgarInstitutionalSource.__new__(EdgarInstitutionalSource)
        result = source._empty_df()
        assert list(result.columns) == _COLUMNS
        assert result.empty
