"""Unit tests for insider trading data merging and derived metrics."""

import tempfile
from datetime import date, timedelta

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


def make_insider_data(transactions):
    """Create synthetic insider transaction DataFrame.

    Each entry in transactions is a dict with:
        filed_date, insider_name, insider_title, shares, price, transaction_type
    Positive shares = purchase, negative = sale.
    """
    rows = []
    for txn in transactions:
        rows.append(
            {
                "filed_date": txn.get("filed_date"),
                "transaction_date": txn.get("transaction_date", txn.get("filed_date")),
                "insider_name": txn.get("insider_name", "John Doe"),
                "insider_title": txn.get("insider_title", "Director"),
                "transaction_type": txn.get("transaction_type", "P" if txn.get("shares", 0) > 0 else "S"),
                "shares": txn.get("shares", 0),
                "price": txn.get("price", 50.0),
                "shares_after": txn.get("shares_after", 10000),
                "is_direct": txn.get("is_direct", True),
            }
        )
    return pd.DataFrame(rows)


def _make_manager_with_insider(symbol, insider_df):
    """Create EdgarDataManager with pre-cached insider data."""
    tmpdir = tempfile.mkdtemp()
    cache = EdgarCache(tmpdir, "insider")
    cache.save(symbol, insider_df)
    mgr = EdgarDataManager(
        cache_dir=tmpdir,
        use_edgar=False,
        enable_financials=False,
        enable_insider=True,
        enable_institutional=False,
        enable_events=False,
    )
    mgr._insider_cache = cache
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInsiderMerge:
    """Test insider data merging and rolling window aggregations."""

    def test_merge_creates_insider_columns(self):
        """After merge, DataFrame should contain insider_ prefixed columns."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 5000, "price": 50.0,
             "insider_name": "CEO", "insider_title": "Chief Executive Officer"},
        ])
        daily = make_daily_df(start="2020-01-02", days=30)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        expected_cols = [
            "insider_net_shares_30d",
            "insider_buy_count_90d",
            "insider_sell_count_90d",
            "insider_buy_ratio_90d",
            "insider_net_value_30d",
            "insider_officer_buys_90d",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_net_shares_30d(self):
        """Net shares in 30-day window sums purchases and sales."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 5000, "price": 50.0},
            {"filed_date": date(2020, 1, 15), "shares": -2000, "price": 50.0},
        ])
        daily = make_daily_df(start="2020-01-02", days=30)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        # On Jan 20 (within 30 days of both transactions), net = 5000 - 2000 = 3000
        jan20_idx = result.index >= pd.Timestamp(2020, 1, 20)
        if jan20_idx.any():
            row = result.loc[jan20_idx].iloc[0]
            assert row["insider_net_shares_30d"] == 3000.0

    def test_buy_sell_counts_90d(self):
        """Count of buy and sell transactions in 90-day window."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 1000},  # buy
            {"filed_date": date(2020, 1, 15), "shares": 2000},  # buy
            {"filed_date": date(2020, 1, 20), "shares": -500},  # sell
        ])
        daily = make_daily_df(start="2020-01-02", days=50)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        # On Feb 1, all 3 transactions within 90d
        feb1 = result.loc[result.index >= pd.Timestamp(2020, 2, 1)]
        if not feb1.empty:
            row = feb1.iloc[0]
            assert row["insider_buy_count_90d"] == 2.0
            assert row["insider_sell_count_90d"] == 1.0

    def test_buy_ratio_90d(self):
        """Buy ratio = buys / (buys + sells) in 90 days."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 1000},
            {"filed_date": date(2020, 1, 15), "shares": 2000},
            {"filed_date": date(2020, 1, 20), "shares": -500},
        ])
        daily = make_daily_df(start="2020-01-02", days=50)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        feb1 = result.loc[result.index >= pd.Timestamp(2020, 2, 1)]
        if not feb1.empty:
            ratio = feb1.iloc[0]["insider_buy_ratio_90d"]
            # 2 buys / 3 total = 0.6667
            assert ratio == pytest.approx(2.0 / 3.0, rel=0.01)

    def test_officer_filtering(self):
        """Only CEO/CFO/COO titles count as officer buys."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 1000, "insider_title": "Chief Executive Officer"},
            {"filed_date": date(2020, 1, 12), "shares": 2000, "insider_title": "Chief Financial Officer"},
            {"filed_date": date(2020, 1, 14), "shares": 3000, "insider_title": "Director"},
            {"filed_date": date(2020, 1, 16), "shares": 4000, "insider_title": "VP Marketing"},
        ])
        daily = make_daily_df(start="2020-01-02", days=50)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        feb1 = result.loc[result.index >= pd.Timestamp(2020, 2, 1)]
        if not feb1.empty:
            officer_buys = feb1.iloc[0]["insider_officer_buys_90d"]
            # Only CEO and CFO qualify
            assert officer_buys == 2.0

    def test_point_in_time_insider(self):
        """Insider data should only be visible on/after filed_date."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 20), "shares": 10000},
        ])
        daily = make_daily_df(start="2020-01-02", days=30)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        # Before Jan 20, should not see any shares
        before = result.loc[result.index < pd.Timestamp(2020, 1, 20)]
        assert (before["insider_net_shares_30d"] == 0).all()

        # On/after Jan 20, should see the shares
        after = result.loc[result.index >= pd.Timestamp(2020, 1, 20)]
        if not after.empty:
            assert after.iloc[0]["insider_net_shares_30d"] == 10000.0

    def test_net_value_30d(self):
        """Net value = sum(shares * price) in 30 days."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 100, "price": 50.0},
            {"filed_date": date(2020, 1, 15), "shares": -50, "price": 60.0},
        ])
        daily = make_daily_df(start="2020-01-02", days=30)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        jan20 = result.loc[result.index >= pd.Timestamp(2020, 1, 20)]
        if not jan20.empty:
            val = jan20.iloc[0]["insider_net_value_30d"]
            # 100*50 + (-50)*60 = 5000 - 3000 = 2000
            assert val == pytest.approx(2000.0)

    def test_empty_insider_data(self):
        """No insider transactions -> DataFrame unchanged (no insider_ columns)."""
        mgr = EdgarDataManager(use_edgar=False, enable_insider=True,
                               enable_financials=False, enable_events=False)
        daily = make_daily_df(days=10)
        result = mgr.merge_all_onto_daily("TEST", daily)

        insider_cols = [c for c in result.columns if c.startswith("insider_")]
        assert len(insider_cols) == 0

    def test_purchase_positive_sale_negative(self):
        """Purchase = positive shares, Sale = negative shares."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 500, "transaction_type": "P"},
            {"filed_date": date(2020, 1, 12), "shares": -300, "transaction_type": "S"},
        ])
        daily = make_daily_df(start="2020-01-02", days=30)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        jan15 = result.loc[result.index >= pd.Timestamp(2020, 1, 15)]
        if not jan15.empty:
            net = jan15.iloc[0]["insider_net_shares_30d"]
            assert net == 200.0

    def test_heavy_selling(self):
        """Large negative net_shares_30d indicates heavy selling."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": -50000, "price": 100.0,
             "insider_title": "Chief Executive Officer"},
            {"filed_date": date(2020, 1, 12), "shares": -30000, "price": 100.0,
             "insider_title": "Chief Financial Officer"},
        ])
        daily = make_daily_df(start="2020-01-02", days=30)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        jan15 = result.loc[result.index >= pd.Timestamp(2020, 1, 15)]
        if not jan15.empty:
            net = jan15.iloc[0]["insider_net_shares_30d"]
            assert net == -80000.0

    def test_multiple_insiders(self):
        """Multiple insiders buying/selling in same window."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 1000, "insider_name": "Alice",
             "insider_title": "CEO"},
            {"filed_date": date(2020, 1, 11), "shares": 2000, "insider_name": "Bob",
             "insider_title": "CFO"},
            {"filed_date": date(2020, 1, 12), "shares": -500, "insider_name": "Charlie",
             "insider_title": "Director"},
        ])
        daily = make_daily_df(start="2020-01-02", days=30)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        jan15 = result.loc[result.index >= pd.Timestamp(2020, 1, 15)]
        if not jan15.empty:
            row = jan15.iloc[0]
            assert row["insider_net_shares_30d"] == 2500.0
            assert row["insider_buy_count_90d"] == 2.0
            assert row["insider_sell_count_90d"] == 1.0

    def test_buy_ratio_no_transactions(self):
        """Buy ratio is NaN when there are no transactions in the window."""
        insider = make_insider_data([
            # Transaction far in the past, outside 90d window of our daily data
            {"filed_date": date(2019, 1, 10), "shares": 1000},
        ])
        daily = make_daily_df(start="2020-06-01", days=10)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        # No transactions in 90d window -> ratio is NaN
        assert pd.isna(result["insider_buy_ratio_90d"].iloc[-1])

    def test_row_count_preserved(self):
        """Merge should not change the number of rows."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 1000},
        ])
        daily = make_daily_df(start="2020-01-02", days=50)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)
        assert len(result) == len(daily)

    def test_does_not_mutate_input(self):
        """Merge should not mutate the input DataFrame."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 1000},
        ])
        daily = make_daily_df(start="2020-01-02", days=10)
        original_cols = list(daily.columns)
        mgr = _make_manager_with_insider("TEST", insider)
        mgr.merge_all_onto_daily("TEST", daily)
        assert list(daily.columns) == original_cols

    def test_coo_counted_as_officer(self):
        """COO title should count as an officer buy."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 10), "shares": 1000,
             "insider_title": "Chief Operating Officer"},
        ])
        daily = make_daily_df(start="2020-01-02", days=30)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        jan15 = result.loc[result.index >= pd.Timestamp(2020, 1, 15)]
        if not jan15.empty:
            assert jan15.iloc[0]["insider_officer_buys_90d"] == 1.0

    def test_30d_window_expiry(self):
        """Transaction outside 30d window should not appear in net_shares_30d."""
        insider = make_insider_data([
            {"filed_date": date(2020, 1, 5), "shares": 10000},
        ])
        daily = make_daily_df(start="2020-01-02", days=60)
        mgr = _make_manager_with_insider("TEST", insider)
        result = mgr.merge_all_onto_daily("TEST", daily)

        # After 30+ days from Jan 5, should not appear in 30d window
        late = result.loc[result.index > pd.Timestamp(2020, 2, 5)]
        if not late.empty:
            assert late.iloc[0]["insider_net_shares_30d"] == 0.0


# ---------------------------------------------------------------------------
# Fetch logic tests (mocked edgartools)
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch, PropertyMock


def _make_mock_filing(
    filed_date,
    owner_name,
    owner_title,
    issuer_name,
    issuer_cik,
    transactions_df,
):
    """Create a mock Form 4 filing object.

    *transactions_df* is a DataFrame that the parsed Form 4 exposes via
    ``non_derivative_table``.
    """
    filing = MagicMock()
    filing.filing_date = filed_date

    parsed = MagicMock()
    parsed.owner_name = owner_name
    parsed.owner_title = owner_title
    parsed.issuer = MagicMock()
    parsed.issuer.name = issuer_name
    parsed.issuer.cik = issuer_cik
    parsed.non_derivative_table = transactions_df
    parsed.to_dataframe.return_value = transactions_df
    # Legacy attributes should be None so the code falls through to the DF path
    parsed.transactions = None
    parsed.non_derivative_transactions = None

    filing.obj.return_value = parsed
    return filing


def _make_txn_df(rows):
    """Build a transactions DataFrame with standard column names."""
    return pd.DataFrame(rows, columns=[
        "transaction_code", "transaction_shares",
        "transaction_price_per_share", "transaction_date",
        "acquired_disposed_code", "shares_owned_following",
        "direct_or_indirect_ownership",
    ])


class TestEdgarInsiderFetch:
    """Tests for EdgarInsiderSource.fetch() with mocked edgartools."""

    @patch("backtester.data.edgar_insider.Company")
    def test_filters_out_zero_price_transactions(self, MockCompany):
        """RSU grants with $0 price should be excluded."""
        txn_df = _make_txn_df([
            # $0 grant — should be filtered out
            ["P", 1000, 0.0, date(2020, 1, 8), "A", 5000, "D"],
            # Real purchase — should be kept
            ["P", 500, 50.0, date(2020, 1, 9), "A", 5500, "D"],
        ])
        filing = _make_mock_filing(
            filed_date=date(2020, 1, 10),
            owner_name="Jane CEO",
            owner_title="CEO",
            issuer_name="TEST Inc",
            issuer_cik=12345,
            transactions_df=txn_df,
        )
        company = MagicMock()
        company.cik = 12345
        company.get_filings.return_value = [filing]
        MockCompany.return_value = company

        from backtester.data.edgar_insider import EdgarInsiderSource
        source = EdgarInsiderSource.__new__(EdgarInsiderSource)
        source.user_agent = "test"
        source.max_filings = 50
        result = source.fetch("TEST")

        assert len(result) == 1
        assert result.iloc[0]["price"] == 50.0
        assert result.iloc[0]["shares"] == 500.0

    @patch("backtester.data.edgar_insider.Company")
    def test_only_open_market_codes(self, MockCompany):
        """Only P and S transaction codes should be included."""
        txn_df = _make_txn_df([
            ["P", 1000, 50.0, date(2020, 1, 8), "A", 5000, "D"],   # purchase — keep
            ["S", 500, 55.0, date(2020, 1, 8), "D", 4500, "D"],     # sale — keep
            ["A", 2000, 0.0, date(2020, 1, 8), "A", 6500, "D"],     # award — exclude
            ["M", 1500, 30.0, date(2020, 1, 8), "A", 8000, "D"],    # derivative exercise — exclude
        ])
        filing = _make_mock_filing(
            filed_date=date(2020, 1, 10),
            owner_name="Jane CEO",
            owner_title="CEO",
            issuer_name="TEST Inc",
            issuer_cik=12345,
            transactions_df=txn_df,
        )
        company = MagicMock()
        company.cik = 12345
        company.get_filings.return_value = [filing]
        MockCompany.return_value = company

        from backtester.data.edgar_insider import EdgarInsiderSource
        source = EdgarInsiderSource.__new__(EdgarInsiderSource)
        source.user_agent = "test"
        source.max_filings = 50
        result = source.fetch("TEST")

        codes = set(result["transaction_type"].tolist())
        assert codes == {"P", "S"}
        assert len(result) == 2

    @patch("backtester.data.edgar_insider.Company")
    def test_issuer_mismatch_filtered(self, MockCompany):
        """Filings where issuer doesn't match target symbol should be skipped."""
        # Filing where issuer CIK (99999) != target company CIK (12345)
        txn_df = _make_txn_df([
            ["P", 1000, 50.0, date(2020, 1, 8), "A", 5000, "D"],
        ])
        wrong_issuer_filing = _make_mock_filing(
            filed_date=date(2020, 1, 10),
            owner_name="Jane CEO",
            owner_title="CEO",
            issuer_name="OTHER Corp",
            issuer_cik=99999,
            transactions_df=txn_df,
        )
        # Filing where issuer CIK matches target
        right_issuer_filing = _make_mock_filing(
            filed_date=date(2020, 1, 11),
            owner_name="Bob CFO",
            owner_title="CFO",
            issuer_name="TEST Inc",
            issuer_cik=12345,
            transactions_df=txn_df,
        )

        company = MagicMock()
        company.cik = 12345
        company.get_filings.return_value = [wrong_issuer_filing, right_issuer_filing]
        MockCompany.return_value = company

        from backtester.data.edgar_insider import EdgarInsiderSource
        source = EdgarInsiderSource.__new__(EdgarInsiderSource)
        source.user_agent = "test"
        source.max_filings = 50
        result = source.fetch("TEST")

        # Only the matching-issuer filing should produce rows
        assert len(result) == 1
        assert result.iloc[0]["insider_name"] == "Bob CFO"

    @patch("backtester.data.edgar_insider.Company")
    def test_purchase_positive_sale_negative_in_fetch(self, MockCompany):
        """Verify sign convention in fetch: P -> positive, S -> negative."""
        txn_df = _make_txn_df([
            ["P", 1000, 50.0, date(2020, 1, 8), "A", 5000, "D"],
            ["S", 800, 55.0, date(2020, 1, 9), "D", 4200, "D"],
        ])
        filing = _make_mock_filing(
            filed_date=date(2020, 1, 10),
            owner_name="Jane CEO",
            owner_title="CEO",
            issuer_name="TEST Inc",
            issuer_cik=12345,
            transactions_df=txn_df,
        )
        company = MagicMock()
        company.cik = 12345
        company.get_filings.return_value = [filing]
        MockCompany.return_value = company

        from backtester.data.edgar_insider import EdgarInsiderSource
        source = EdgarInsiderSource.__new__(EdgarInsiderSource)
        source.user_agent = "test"
        source.max_filings = 50
        result = source.fetch("TEST")

        purchase = result[result["transaction_type"] == "P"].iloc[0]
        sale = result[result["transaction_type"] == "S"].iloc[0]
        assert purchase["shares"] == 1000.0  # positive
        assert sale["shares"] == -800.0      # negative

    @patch("backtester.data.edgar_insider.Company")
    def test_empty_filings_returns_empty_df(self, MockCompany):
        """No filings -> empty DataFrame with correct columns."""
        company = MagicMock()
        company.cik = 12345
        company.get_filings.return_value = []
        MockCompany.return_value = company

        from backtester.data.edgar_insider import EdgarInsiderSource
        source = EdgarInsiderSource.__new__(EdgarInsiderSource)
        source.user_agent = "test"
        source.max_filings = 50
        result = source.fetch("TEST")

        from backtester.data.edgar_insider import _COLUMNS
        assert list(result.columns) == _COLUMNS
        assert len(result) == 0

    @patch("backtester.data.edgar_insider.Company")
    def test_dataframe_api_parsing(self, MockCompany):
        """When parsed form returns DataFrame (non_derivative_table), parse correctly."""
        txn_df = _make_txn_df([
            ["P", 2000, 100.0, date(2020, 3, 1), "A", 12000, "D"],
            ["S", 500, 105.0, date(2020, 3, 2), "D", 11500, "I"],
        ])
        filing = _make_mock_filing(
            filed_date=date(2020, 3, 5),
            owner_name="Alice Director",
            owner_title="Director",
            issuer_name="ACME Corp",
            issuer_cik=54321,
            transactions_df=txn_df,
        )
        company = MagicMock()
        company.cik = 54321
        company.get_filings.return_value = [filing]
        MockCompany.return_value = company

        from backtester.data.edgar_insider import EdgarInsiderSource
        source = EdgarInsiderSource.__new__(EdgarInsiderSource)
        source.user_agent = "test"
        source.max_filings = 50
        result = source.fetch("ACME")

        assert len(result) == 2
        buy_row = result[result["transaction_type"] == "P"].iloc[0]
        sell_row = result[result["transaction_type"] == "S"].iloc[0]

        assert buy_row["shares"] == 2000.0
        assert buy_row["price"] == 100.0
        assert buy_row["insider_name"] == "Alice Director"
        assert buy_row["is_direct"] == True  # noqa: E712 (numpy bool)

        assert sell_row["shares"] == -500.0
        assert sell_row["price"] == 105.0
        assert sell_row["is_direct"] == False  # noqa: E712 (numpy bool)

    @patch("backtester.data.edgar_insider.Company")
    def test_legacy_iter_parsing_fallback(self, MockCompany):
        """When non_derivative_table is None, fall back to legacy iterable API."""
        filing = MagicMock()
        filing.filing_date = date(2020, 2, 1)

        parsed = MagicMock()
        parsed.owner_name = "Legacy User"
        parsed.owner_title = "VP"
        parsed.issuer = MagicMock()
        parsed.issuer.cik = 11111
        # No DataFrame-based API
        parsed.non_derivative_table = None
        parsed.to_dataframe.side_effect = AttributeError("no method")

        # Legacy iterable
        txn = MagicMock()
        txn.transaction_code = "P"
        txn.transaction_shares = 300
        txn.transaction_price_per_share = 75.0
        txn.shares_owned_following = 1300
        txn.transaction_date = date(2020, 1, 30)
        txn.direct_or_indirect_ownership = "D"
        txn.acquired_disposed_code = "A"
        parsed.transactions = [txn]
        parsed.non_derivative_transactions = None

        filing.obj.return_value = parsed

        company = MagicMock()
        company.cik = 11111
        company.get_filings.return_value = [filing]
        MockCompany.return_value = company

        from backtester.data.edgar_insider import EdgarInsiderSource
        source = EdgarInsiderSource.__new__(EdgarInsiderSource)
        source.user_agent = "test"
        source.max_filings = 50
        result = source.fetch("LEG")

        assert len(result) == 1
        assert result.iloc[0]["shares"] == 300.0
        assert result.iloc[0]["price"] == 75.0
        assert result.iloc[0]["insider_name"] == "Legacy User"

    @patch("backtester.data.edgar_insider.Company")
    def test_none_filings_returns_empty_df(self, MockCompany):
        """get_filings returning None -> empty DataFrame."""
        company = MagicMock()
        company.cik = 12345
        company.get_filings.return_value = None
        MockCompany.return_value = company

        from backtester.data.edgar_insider import EdgarInsiderSource
        source = EdgarInsiderSource.__new__(EdgarInsiderSource)
        source.user_agent = "test"
        source.max_filings = 50
        result = source.fetch("TEST")

        assert len(result) == 0
