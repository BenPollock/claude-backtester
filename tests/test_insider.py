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
