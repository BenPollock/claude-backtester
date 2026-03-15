"""Tests for FundamentalDataManager, EdgarDataManager, EdgarCache, and financial data merging."""

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.data.fundamental import (
    EdgarDataManager,
    FundamentalDataManager,
    _CSVFundamentalData,
)
from backtester.data.fundamental_cache import EdgarCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(tmpdir, filename, content):
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


def make_daily_df(start="2020-01-02", days=252, start_price=100.0):
    """Create a simple daily OHLCV DataFrame for merge tests."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    close = np.linspace(start_price, start_price * 1.2, days)
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(days, 1_000_000),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )
    return df


def make_financial_data(
    quarters=8,
    start_revenue=1e9,
    growth=0.10,
    start_date=date(2019, 3, 31),
    filing_delay_days=35,
):
    """Create synthetic quarterly financial statement data.

    Returns a DataFrame with columns: metric, period_end, filed_date, value, form.
    """
    rows = []
    for q in range(quarters):
        period_end = start_date + timedelta(days=91 * q)
        filed_date = period_end + timedelta(days=filing_delay_days)
        rev = start_revenue * (1 + growth) ** q
        ni = rev * 0.10
        eps = ni / 1e7
        oi = rev * 0.15
        gp = rev * 0.40
        ocf = rev * 0.12
        capex = rev * 0.03

        metrics = {
            "revenue": rev,
            "net_income": ni,
            "eps_diluted": eps,
            "operating_income": oi,
            "gross_profit": gp,
            "operating_cf": ocf,
            "capex": capex,
            "total_assets": rev * 2.0,
            "total_debt": rev * 0.5,
            "current_assets": rev * 0.6,
            "current_liabilities": rev * 0.3,
            "equity": rev * 0.8,
            "shares_outstanding": 1e7,
        }

        for metric, value in metrics.items():
            rows.append(
                {
                    "metric": metric,
                    "period_end": period_end,
                    "filed_date": filed_date,
                    "value": value,
                    "form": "10-Q",
                }
            )

    return pd.DataFrame(rows)


def _make_manager_with_cached_financials(symbol, fin_df):
    """Create EdgarDataManager with pre-cached financials in a temp dir."""
    tmpdir = tempfile.mkdtemp()
    cache = EdgarCache(tmpdir, "financials")
    cache.save(symbol, fin_df)
    mgr = EdgarDataManager(
        cache_dir=tmpdir,
        use_edgar=False,
        enable_financials=True,
        enable_insider=False,
        enable_institutional=False,
        enable_events=False,
    )
    # Manually attach cache since use_edgar=False skips _init_edgar_sources
    mgr._financials_cache = cache
    return mgr


# ===========================================================================
# 1. Legacy CSV FundamentalDataManager tests (preserved from original)
# ===========================================================================


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
        val = mgr.get("AAPL", "pe_ratio", date(2020, 3, 1))
        assert val == 25.0
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
        val = mgr.get("AAPL", "pe_ratio", date(2020, 2, 1))
        assert val is None
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
        csv_content = (
            "date,symbol,field,value\n"
            "2020-07-15,AAPL,pe_ratio,28.0\n"
            "2020-01-15,AAPL,pe_ratio,25.0\n"
            "2020-04-15,AAPL,pe_ratio,22.5\n"
        )
        path = _write_csv(str(tmp_path), "fundamentals.csv", csv_content)
        mgr = FundamentalDataManager(path)
        assert mgr.get("AAPL", "pe_ratio", date(2020, 3, 1)) == 25.0
        assert mgr.get("AAPL", "pe_ratio", date(2020, 5, 1)) == 22.5
        assert mgr.get("AAPL", "pe_ratio", date(2020, 8, 1)) == 28.0


# ===========================================================================
# 2. EdgarDataManager CSV mode
# ===========================================================================


class TestEdgarDataManagerCSVMode:
    """EdgarDataManager with csv_path for backward compatibility."""

    def test_csv_get(self, tmp_path):
        """EdgarDataManager.get() works through CSV path."""
        csv_content = (
            "date,symbol,field,value\n"
            "2020-03-01,SPY,revenue,5000000\n"
            "2020-06-01,SPY,revenue,5500000\n"
        )
        path = _write_csv(str(tmp_path), "fund.csv", csv_content)
        mgr = EdgarDataManager(csv_path=path)
        val = mgr.get("SPY", "revenue", date(2020, 4, 1))
        assert val == 5_000_000.0

    def test_csv_mode_load_financials_empty(self, tmp_path):
        """In CSV-only mode, load_financials returns empty DataFrame."""
        csv_content = "date,symbol,field,value\n2020-03-01,X,eps,1.0\n"
        path = _write_csv(str(tmp_path), "fund.csv", csv_content)
        mgr = EdgarDataManager(csv_path=path)
        result = mgr.load_financials("X")
        assert result.empty


# ===========================================================================
# 3. merge_all_onto_daily with financials
# ===========================================================================


class TestMergeFinancials:
    """Test financial statement data merging onto daily DataFrames."""

    def test_merge_creates_fund_columns(self):
        """After merge, daily DataFrame should contain fund_ prefixed columns."""
        fin = make_financial_data(quarters=4, start_date=date(2019, 6, 30))
        daily = make_daily_df(start="2020-01-02", days=100)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert len(result) == len(daily), "Row count must be preserved"
        fund_cols = [c for c in result.columns if c.startswith("fund_")]
        assert len(fund_cols) > 0, "Should have fund_ columns after merge"

    def test_ttm_computation(self):
        """TTM of revenue should be sum of last 4 quarterly values."""
        rows = []
        base = date(2019, 3, 31)
        for q in range(8):
            period_end = base + timedelta(days=91 * q)
            filed_date = period_end + timedelta(days=35)
            rev = (q + 1) * 100.0
            rows.append(
                {
                    "metric": "revenue",
                    "period_end": period_end,
                    "filed_date": filed_date,
                    "value": rev,
                    "form": "10-Q",
                }
            )
        fin = pd.DataFrame(rows)

        # Daily data must start after the last filing date (2021-01-31)
        # so merge_asof can see all 8 quarterly filings
        daily = make_daily_df(start="2021-02-01", days=10)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert "fund_revenue_ttm" in result.columns
        ttm_val = result["fund_revenue_ttm"].iloc[-1]
        assert ttm_val == pytest.approx(2600.0), f"Expected TTM=2600, got {ttm_val}"

    def test_pe_ratio_computation(self):
        """P/E ratio = Close / eps_diluted_ttm."""
        rows = []
        base = date(2019, 3, 31)
        for q in range(4):
            period_end = base + timedelta(days=91 * q)
            filed_date = period_end + timedelta(days=35)
            rows.append(
                {
                    "metric": "eps_diluted",
                    "period_end": period_end,
                    "filed_date": filed_date,
                    "value": 2.5,
                    "form": "10-Q",
                }
            )
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-06-01", days=10, start_price=100.0)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert "fund_pe_ratio" in result.columns
        pe = result["fund_pe_ratio"].iloc[0]
        assert not pd.isna(pe), "P/E ratio should not be NaN"
        assert 9.0 < pe < 11.0, f"P/E should be ~10, got {pe}"

    def test_pb_ratio_computation(self):
        """P/B ratio = Close / (equity / shares_outstanding)."""
        pe_d = date(2019, 6, 30)
        fd = date(2019, 8, 4)
        rows = [
            {"metric": "equity", "period_end": pe_d, "filed_date": fd, "value": 1e9, "form": "10-Q"},
            {"metric": "shares_outstanding", "period_end": pe_d, "filed_date": fd, "value": 1e7, "form": "10-Q"},
        ]
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-01-02", days=10, start_price=200.0)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert "fund_pb_ratio" in result.columns
        pb = result["fund_pb_ratio"].iloc[0]
        assert not pd.isna(pb)
        assert 1.8 < pb < 2.2, f"P/B should be ~2.0, got {pb}"

    def test_point_in_time_correctness(self):
        """Data should not be visible before its filing date."""
        rows = [
            {
                "metric": "revenue",
                "period_end": date(2020, 3, 31),
                "filed_date": date(2020, 5, 5),
                "value": 1e9,
                "form": "10-Q",
            }
        ]
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-04-01", days=30)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        if "fund_revenue" in result.columns:
            before_filing = result.loc[result.index < pd.Timestamp(2020, 5, 5)]
            assert before_filing["fund_revenue"].isna().all(), (
                "Revenue should be NaN before filing date"
            )
            after_filing = result.loc[result.index >= pd.Timestamp(2020, 5, 5)]
            if not after_filing.empty:
                assert after_filing["fund_revenue"].notna().any(), (
                    "Revenue should be available on/after filing date"
                )

    def test_forward_fill_between_filings(self):
        """Quarterly data should carry forward between filing dates."""
        rows = [
            {"metric": "revenue", "period_end": date(2019, 3, 31),
             "filed_date": date(2019, 5, 5), "value": 1e9, "form": "10-Q"},
            {"metric": "revenue", "period_end": date(2019, 6, 30),
             "filed_date": date(2019, 8, 4), "value": 1.1e9, "form": "10-Q"},
        ]
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2019-05-06", days=80)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        if "fund_revenue" in result.columns:
            mid_row = result.loc[result.index <= pd.Timestamp(2019, 7, 1)]
            if not mid_row.empty:
                assert mid_row["fund_revenue"].iloc[-1] == pytest.approx(1e9)

    def test_margin_calculations(self):
        """Known revenue + operating_income + net_income -> correct margins."""
        pe_d = date(2019, 6, 30)
        fd = date(2019, 8, 4)
        rows = [
            {"metric": "revenue", "period_end": pe_d, "filed_date": fd, "value": 1000.0, "form": "10-Q"},
            {"metric": "operating_income", "period_end": pe_d, "filed_date": fd, "value": 150.0, "form": "10-Q"},
            {"metric": "net_income", "period_end": pe_d, "filed_date": fd, "value": 100.0, "form": "10-Q"},
            {"metric": "gross_profit", "period_end": pe_d, "filed_date": fd, "value": 400.0, "form": "10-Q"},
        ]
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-01-02", days=10)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert result["fund_gross_margin"].iloc[-1] == pytest.approx(0.40)
        assert result["fund_operating_margin"].iloc[-1] == pytest.approx(0.15)
        assert result["fund_net_margin"].iloc[-1] == pytest.approx(0.10)

    def test_fcf_calculation(self):
        """FCF = operating_cf - capex."""
        pe_d = date(2019, 6, 30)
        fd = date(2019, 8, 4)
        rows = [
            {"metric": "operating_cf", "period_end": pe_d, "filed_date": fd, "value": 500.0, "form": "10-Q"},
            {"metric": "capex", "period_end": pe_d, "filed_date": fd, "value": 100.0, "form": "10-Q"},
        ]
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-01-02", days=10)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert "fund_free_cash_flow" in result.columns
        fcf = result["fund_free_cash_flow"].iloc[-1]
        assert fcf == pytest.approx(400.0)

    def test_current_ratio(self):
        """Current ratio = current_assets / current_liabilities."""
        pe_d = date(2019, 6, 30)
        fd = date(2019, 8, 4)
        rows = [
            {"metric": "current_assets", "period_end": pe_d, "filed_date": fd, "value": 600.0, "form": "10-Q"},
            {"metric": "current_liabilities", "period_end": pe_d, "filed_date": fd, "value": 300.0, "form": "10-Q"},
        ]
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-01-02", days=10)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert result["fund_current_ratio"].iloc[-1] == pytest.approx(2.0)

    def test_debt_to_equity(self):
        """D/E = total_debt / equity."""
        pe_d = date(2019, 6, 30)
        fd = date(2019, 8, 4)
        rows = [
            {"metric": "total_debt", "period_end": pe_d, "filed_date": fd, "value": 500.0, "form": "10-Q"},
            {"metric": "equity", "period_end": pe_d, "filed_date": fd, "value": 1000.0, "form": "10-Q"},
        ]
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-01-02", days=10)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert result["fund_debt_to_equity"].iloc[-1] == pytest.approx(0.5)

    def test_roa_roe(self):
        """ROA = net_income_ttm / total_assets, ROE = net_income_ttm / equity."""
        rows = []
        for q in range(4):
            pe_d = date(2019, 3, 31) + timedelta(days=91 * q)
            fd = pe_d + timedelta(days=35)
            rows.append({"metric": "net_income", "period_end": pe_d, "filed_date": fd, "value": 25.0, "form": "10-Q"})
            rows.append({"metric": "total_assets", "period_end": pe_d, "filed_date": fd, "value": 1000.0, "form": "10-Q"})
            rows.append({"metric": "equity", "period_end": pe_d, "filed_date": fd, "value": 500.0, "form": "10-Q"})
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-06-01", days=10)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert result["fund_roa"].iloc[-1] == pytest.approx(100.0 / 1000.0)
        assert result["fund_roe"].iloc[-1] == pytest.approx(100.0 / 500.0)

    def test_ps_ratio(self):
        """P/S = (Close * shares) / revenue_ttm."""
        rows = []
        for q in range(4):
            pe_d = date(2019, 3, 31) + timedelta(days=91 * q)
            fd = pe_d + timedelta(days=35)
            rows.append({"metric": "revenue", "period_end": pe_d, "filed_date": fd, "value": 250.0, "form": "10-Q"})
            rows.append({"metric": "shares_outstanding", "period_end": pe_d, "filed_date": fd, "value": 10.0, "form": "10-Q"})
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-06-01", days=10, start_price=100.0)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        ps = result["fund_ps_ratio"].iloc[0]
        assert not pd.isna(ps)
        assert 0.8 < ps < 1.2, f"P/S should be ~1.0, got {ps}"

    def test_fcf_yield(self):
        """FCF yield = fcf_ttm / mkt_cap."""
        rows = []
        for q in range(4):
            pe_d = date(2019, 3, 31) + timedelta(days=91 * q)
            fd = pe_d + timedelta(days=35)
            rows.append({"metric": "operating_cf", "period_end": pe_d, "filed_date": fd, "value": 50.0, "form": "10-Q"})
            rows.append({"metric": "capex", "period_end": pe_d, "filed_date": fd, "value": 10.0, "form": "10-Q"})
            rows.append({"metric": "shares_outstanding", "period_end": pe_d, "filed_date": fd, "value": 10.0, "form": "10-Q"})
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-06-01", days=10, start_price=100.0)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        if "fund_fcf_yield" in result.columns:
            fcf_y = result["fund_fcf_yield"].iloc[0]
            if not pd.isna(fcf_y):
                assert 0.1 < fcf_y < 0.2, f"FCF yield should be ~0.16, got {fcf_y}"

    def test_growth_rates_require_252_days(self):
        """Revenue and earnings growth YoY use shift(252), so short data yields NaN."""
        fin = make_financial_data(quarters=4)
        daily = make_daily_df(start="2020-01-02", days=50)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert result["fund_revenue_growth_yoy"].isna().all()
        assert result["fund_earnings_growth_yoy"].isna().all()

    def test_empty_data_returns_unmodified(self):
        """Empty financial data -> DataFrame unchanged (no fund_ columns added)."""
        mgr = EdgarDataManager(use_edgar=False)
        daily = make_daily_df(days=10)
        result = mgr.merge_all_onto_daily("TEST", daily)

        fund_cols = [c for c in result.columns if c.startswith("fund_")]
        assert len(fund_cols) == 0, "No fund_ columns with empty data"

    def test_missing_some_metrics_produce_nan(self):
        """Only partial financial data -> missing derived columns are NaN."""
        rows = [
            {
                "metric": "revenue",
                "period_end": date(2019, 6, 30),
                "filed_date": date(2019, 8, 4),
                "value": 1e9,
                "form": "10-Q",
            }
        ]
        fin = pd.DataFrame(rows)
        daily = make_daily_df(start="2020-01-02", days=10)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        assert result["fund_pe_ratio"].isna().all()
        assert result["fund_pb_ratio"].isna().all()

    def test_does_not_mutate_input(self):
        """merge_all_onto_daily should not mutate the input DataFrame."""
        fin = make_financial_data(quarters=4)
        daily = make_daily_df(days=10)
        original_cols = list(daily.columns)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        mgr.merge_all_onto_daily("TEST", daily)

        assert list(daily.columns) == original_cols

    def test_non_datetime_index_returns_unchanged(self):
        """If index is not DatetimeIndex, return unchanged."""
        mgr = EdgarDataManager(use_edgar=False, enable_financials=True)
        df = pd.DataFrame({"Close": [100, 101]}, index=[0, 1])
        result = mgr.merge_all_onto_daily("TEST", df)
        assert list(result.columns) == ["Close"]

    def test_ttm_with_fewer_than_4_quarters(self):
        """TTM with only 2 quarters sums those 2 (partial TTM)."""
        rows = []
        base = date(2019, 3, 31)
        for q in range(2):
            pe_d = base + timedelta(days=91 * q)
            fd = pe_d + timedelta(days=35)
            rows.append({"metric": "revenue", "period_end": pe_d,
                         "filed_date": fd, "value": 100.0, "form": "10-Q"})
        fin = pd.DataFrame(rows)

        daily = make_daily_df(start="2020-01-02", days=10)
        mgr = _make_manager_with_cached_financials("TEST", fin)
        result = mgr.merge_all_onto_daily("TEST", daily)

        ttm = result["fund_revenue_ttm"].iloc[-1]
        assert ttm == pytest.approx(200.0)


# ===========================================================================
# 4. EdgarCache tests
# ===========================================================================


class TestEdgarCache:
    """Test the EdgarCache Parquet cache."""

    def test_path_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            path = cache._path("AAPL")
            assert path == Path(tmpdir) / "edgar" / "financials" / "AAPL.parquet"

    def test_path_uppercase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "insider")
            assert cache._path("aapl").name == "AAPL.parquet"

    def test_has_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            assert cache.has("TEST") is False

    def test_has_after_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            cache.save("TEST", pd.DataFrame({"a": [1]}))
            assert cache.has("TEST") is True

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            df = pd.DataFrame(
                {"metric": ["revenue"], "period_end": ["2020-01-01"],
                 "filed_date": ["2020-02-01"], "value": [1e9]}
            )
            cache.save("TEST", df)
            loaded = cache.load("TEST")
            assert loaded is not None
            pd.testing.assert_frame_equal(loaded, df)

    def test_merge_and_save_combines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            df1 = pd.DataFrame(
                {"metric": ["revenue"], "period_end": ["2020-01-01"],
                 "filed_date": ["2020-02-01"], "value": [1e9]}
            )
            cache.save("TEST", df1)
            df2 = pd.DataFrame(
                {"metric": ["revenue"], "period_end": ["2020-04-01"],
                 "filed_date": ["2020-05-01"], "value": [1.1e9]}
            )
            cache.merge_and_save("TEST", df2)
            loaded = cache.load("TEST")
            assert len(loaded) == 2

    def test_merge_and_save_dedup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            df = pd.DataFrame(
                {"metric": ["revenue", "revenue"],
                 "period_end": ["2020-01-01", "2020-01-01"],
                 "filed_date": ["2020-02-01", "2020-02-01"],
                 "value": [1e9, 1.1e9]}
            )
            cache.merge_and_save("TEST", df)
            loaded = cache.load("TEST")
            assert len(loaded) == 1
            assert loaded["value"].iloc[0] == 1.1e9

    def test_date_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            df = pd.DataFrame(
                {"filed_date": ["2020-01-01", "2020-06-01"],
                 "metric": ["a", "b"], "value": [1, 2]}
            )
            cache.save("TEST", df)
            dr = cache.date_range("TEST")
            assert dr is not None
            assert dr[0] == date(2020, 1, 1)
            assert dr[1] == date(2020, 6, 1)

    def test_date_range_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            assert cache.date_range("NONE") is None

    def test_invalid_data_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="data_type must be one of"):
                EdgarCache(tmpdir, "invalid")

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            assert cache.load("NONEXISTENT") is None

    def test_insider_dedup_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "insider")
            assert cache._dedup_keys() == ["filed_date", "insider_name", "shares"]

    def test_institutional_dedup_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "institutional")
            assert cache._dedup_keys() == ["report_date"]

    def test_events_dedup_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "events")
            assert cache._dedup_keys() == ["filed_date", "event_date"]

    def test_merge_and_save_empty_into_empty(self):
        """Merging empty data into empty cache does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            cache.merge_and_save("TEST", pd.DataFrame())
            loaded = cache.load("TEST")
            assert loaded is not None
            assert loaded.empty


# ===========================================================================
# 5. EdgarDataManager source injection
# ===========================================================================


class TestEdgarDataManagerSources:
    """Test source injection for testability."""

    def test_source_injection(self):
        class FakeSource:
            def fetch(self, symbol):
                return pd.DataFrame(
                    {"metric": ["revenue"], "period_end": [date(2020, 3, 31)],
                     "filed_date": [date(2020, 5, 5)], "value": [1e9], "form": ["10-Q"]}
                )

        mgr = EdgarDataManager(sources={"financials": FakeSource()})
        result = mgr.load_financials("TEST")
        assert len(result) == 1
        assert result["value"].iloc[0] == 1e9

    def test_graceful_source_failure(self):
        class FailSource:
            def fetch(self, symbol):
                raise RuntimeError("Network error")

        mgr = EdgarDataManager(sources={"financials": FailSource()})
        result = mgr.load_financials("TEST")
        assert result.empty

    def test_get_from_edgar_cache(self):
        """get() falls back to EDGAR financials cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            df = pd.DataFrame(
                {"metric": ["revenue"], "period_end": [date(2020, 3, 31)],
                 "filed_date": [date(2020, 5, 5)], "value": [1e9], "form": ["10-Q"]}
            )
            cache.save("TEST", df)
            mgr = EdgarDataManager(use_edgar=False)
            mgr._financials_cache = cache
            val = mgr.get("TEST", "revenue", date(2020, 6, 1))
            assert val == 1e9

    def test_cache_used_before_source(self):
        """Cached data is returned without calling source."""
        class NeverCallSource:
            def fetch(self, symbol):
                raise AssertionError("Source should not be called when cache has data")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            cache.save("TEST", pd.DataFrame({"metric": ["rev"], "value": [1]}))
            mgr = EdgarDataManager(sources={"financials": NeverCallSource()})
            mgr._financials_cache = cache
            result = mgr.load_financials("TEST")
            assert len(result) == 1

    def test_source_result_cached(self):
        """Source result is saved to cache for future loads."""
        call_count = [0]

        class CountingSource:
            def fetch(self, symbol):
                call_count[0] += 1
                return pd.DataFrame(
                    {"metric": ["rev"], "period_end": [date(2020, 1, 1)],
                     "filed_date": [date(2020, 2, 1)], "value": [42], "form": ["10-Q"]}
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EdgarCache(tmpdir, "financials")
            mgr = EdgarDataManager(sources={"financials": CountingSource()})
            mgr._financials_cache = cache

            mgr.load_financials("TEST")
            assert call_count[0] == 1

            # Second load should hit cache
            mgr.load_financials("TEST")
            assert call_count[0] == 1
