"""Unit tests for EdgarDataManager._compute_dividend_growth."""

import numpy as np
import pandas as pd
import pytest


def _make_mgr():
    """Create a minimal EdgarDataManager without invoking __init__."""
    from backtester.data.fundamental import EdgarDataManager

    return EdgarDataManager.__new__(EdgarDataManager)


def _make_df(n_rows, dps=None, div_paid=None, ni_ttm=None, start="2019-01-02"):
    """Build a DataFrame with business-day index and optional fundamental columns.

    Parameters
    ----------
    n_rows : int
        Number of business-day rows.
    dps : array-like or None
        Values for fund_dividends_per_share. Column omitted when None.
    div_paid : array-like or None
        Values for fund_dividends_paid_ttm. Column omitted when None.
    ni_ttm : array-like or None
        Values for fund_net_income_ttm. Column omitted when None.
    """
    dates = pd.bdate_range(start=start, periods=n_rows)
    df = pd.DataFrame(index=dates)
    if dps is not None:
        df["fund_dividends_per_share"] = dps
    if div_paid is not None:
        df["fund_dividends_paid_ttm"] = div_paid
    if ni_ttm is not None:
        df["fund_net_income_ttm"] = ni_ttm
    return df


# -----------------------------------------------------------------------
# 1-5, 13, 16: YoY dividend growth tests (require 260+ rows)
# -----------------------------------------------------------------------


class TestDividendGrowthYoY:
    """Tests for fund_div_growth_yoy = (dps - dps.shift(252)) / abs(dps.shift(252))."""

    def test_positive_dividend_growth(self):
        """1. DPS increased YoY produces positive growth."""
        mgr = _make_mgr()
        n = 260
        dps = np.empty(n)
        dps[:] = 1.0
        dps[252:] = 1.5  # 50% increase after 252 days
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        growth_values = result["fund_div_growth_yoy"].iloc[252:]
        assert (growth_values.dropna() > 0).all()
        assert growth_values.iloc[0] == pytest.approx(0.5)

    def test_negative_dividend_growth(self):
        """2. DPS decreased YoY produces negative growth."""
        mgr = _make_mgr()
        n = 260
        dps = np.empty(n)
        dps[:] = 2.0
        dps[252:] = 1.0  # 50% decrease
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        growth_values = result["fund_div_growth_yoy"].iloc[252:]
        assert (growth_values.dropna() < 0).all()
        assert growth_values.iloc[0] == pytest.approx(-0.5)

    def test_zero_growth(self):
        """3. DPS unchanged YoY produces zero growth."""
        mgr = _make_mgr()
        n = 260
        dps = np.full(n, 1.5)
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        growth_values = result["fund_div_growth_yoy"].iloc[252:]
        assert np.allclose(growth_values.dropna().values, 0.0)

    def test_dps_from_zero_gives_nan(self):
        """4. DPS going from 0 to positive produces NaN (division by zero guard)."""
        mgr = _make_mgr()
        n = 260
        dps = np.empty(n)
        dps[:] = 0.0
        dps[252:] = 1.0
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        # At row 252: dps_prev = dps[0] = 0 -> NaN
        assert np.isnan(result["fund_div_growth_yoy"].iloc[252])

    def test_known_calculation(self):
        """5. DPS from 1.0 to 1.1 produces exactly 10% growth."""
        mgr = _make_mgr()
        n = 260
        dps = np.empty(n)
        dps[:] = 1.0
        dps[252:] = 1.1
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        assert result["fund_div_growth_yoy"].iloc[252] == pytest.approx(0.10)

    def test_first_252_rows_are_nan(self):
        """13. First 252 rows lack YoY history so growth should be NaN."""
        mgr = _make_mgr()
        n = 260
        dps = np.full(n, 2.0)
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        assert result["fund_div_growth_yoy"].iloc[:252].isna().all()

    def test_multiple_rows_varying_dividends(self):
        """16. Multiple rows with varying DPS produce correct per-row growth."""
        mgr = _make_mgr()
        n = 260
        dps = np.full(n, 1.0, dtype=float)
        # Row 252: dps=2.0, prev(shift 252)=dps[0]=1.0 -> growth = 1.0
        dps[252] = 2.0
        # Row 253: dps=3.0, prev=dps[1]=1.0 -> growth = 2.0
        dps[253] = 3.0
        # Row 254: dps=0.5, prev=dps[2]=1.0 -> growth = -0.5
        dps[254] = 0.5
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        assert result["fund_div_growth_yoy"].iloc[252] == pytest.approx(1.0)
        assert result["fund_div_growth_yoy"].iloc[253] == pytest.approx(2.0)
        assert result["fund_div_growth_yoy"].iloc[254] == pytest.approx(-0.5)


# -----------------------------------------------------------------------
# 6-9: Payout ratio tests (shorter DataFrames are fine)
# -----------------------------------------------------------------------


class TestPayoutRatio:
    """Tests for fund_payout_ratio = dividends_paid_ttm / net_income_ttm."""

    def test_payout_ratio_basic(self):
        """6. Payout ratio = div_paid / net_income."""
        mgr = _make_mgr()
        n = 10
        div_paid = np.full(n, 50.0)
        ni_ttm = np.full(n, 100.0)
        df = _make_df(n, div_paid=div_paid, ni_ttm=ni_ttm)
        result = mgr._compute_dividend_growth(df)
        assert np.allclose(result["fund_payout_ratio"].values, 0.5)

    def test_payout_ratio_greater_than_one(self):
        """7. Payout ratio > 1.0 when paying more than earning."""
        mgr = _make_mgr()
        n = 10
        div_paid = np.full(n, 150.0)
        ni_ttm = np.full(n, 100.0)
        df = _make_df(n, div_paid=div_paid, ni_ttm=ni_ttm)
        result = mgr._compute_dividend_growth(df)
        assert np.allclose(result["fund_payout_ratio"].values, 1.5)

    def test_payout_ratio_negative_net_income(self):
        """8. Negative net income produces a negative payout ratio."""
        mgr = _make_mgr()
        n = 10
        div_paid = np.full(n, 50.0)
        ni_ttm = np.full(n, -100.0)
        df = _make_df(n, div_paid=div_paid, ni_ttm=ni_ttm)
        result = mgr._compute_dividend_growth(df)
        assert np.allclose(result["fund_payout_ratio"].values, -0.5)

    def test_zero_net_income_gives_nan(self):
        """9. Zero net income produces NaN payout ratio (division by zero guard)."""
        mgr = _make_mgr()
        n = 10
        div_paid = np.full(n, 50.0)
        ni_ttm = np.full(n, 0.0)
        df = _make_df(n, div_paid=div_paid, ni_ttm=ni_ttm)
        result = mgr._compute_dividend_growth(df)
        assert result["fund_payout_ratio"].isna().all()


# -----------------------------------------------------------------------
# 10-12: Missing column tests
# -----------------------------------------------------------------------


class TestMissingColumns:
    """Tests for graceful handling when input columns are absent."""

    def test_missing_dps_gives_nan_growth(self):
        """10. Missing fund_dividends_per_share -> NaN growth."""
        mgr = _make_mgr()
        df = _make_df(260)  # no dps column
        result = mgr._compute_dividend_growth(df)
        assert "fund_div_growth_yoy" in result.columns
        assert result["fund_div_growth_yoy"].isna().all()

    def test_missing_div_paid_gives_nan_payout(self):
        """11. Missing fund_dividends_paid_ttm -> NaN payout ratio."""
        mgr = _make_mgr()
        ni_ttm = np.full(10, 100.0)
        df = _make_df(10, ni_ttm=ni_ttm)  # no div_paid
        result = mgr._compute_dividend_growth(df)
        assert "fund_payout_ratio" in result.columns
        assert result["fund_payout_ratio"].isna().all()

    def test_missing_ni_ttm_gives_nan_payout(self):
        """12. Missing fund_net_income_ttm -> NaN payout ratio."""
        mgr = _make_mgr()
        div_paid = np.full(10, 50.0)
        df = _make_df(10, div_paid=div_paid)  # no ni_ttm
        result = mgr._compute_dividend_growth(df)
        assert "fund_payout_ratio" in result.columns
        assert result["fund_payout_ratio"].isna().all()


# -----------------------------------------------------------------------
# 14-15: Column naming and all-NaN inputs
# -----------------------------------------------------------------------


class TestOutputColumns:
    """Tests for correct output column names and all-NaN edge case."""

    def test_correct_column_names(self):
        """14. Both fund_div_growth_yoy and fund_payout_ratio columns are created."""
        mgr = _make_mgr()
        n = 260
        dps = np.full(n, 1.0)
        div_paid = np.full(n, 50.0)
        ni_ttm = np.full(n, 100.0)
        df = _make_df(n, dps=dps, div_paid=div_paid, ni_ttm=ni_ttm)
        result = mgr._compute_dividend_growth(df)
        assert "fund_div_growth_yoy" in result.columns
        assert "fund_payout_ratio" in result.columns

    def test_all_nan_inputs_give_nan_outputs(self):
        """15. All NaN inputs produce NaN for both growth and payout ratio."""
        mgr = _make_mgr()
        n = 260
        dps = np.full(n, np.nan)
        div_paid = np.full(n, np.nan)
        ni_ttm = np.full(n, np.nan)
        df = _make_df(n, dps=dps, div_paid=div_paid, ni_ttm=ni_ttm)
        result = mgr._compute_dividend_growth(df)
        assert result["fund_div_growth_yoy"].isna().all()
        assert result["fund_payout_ratio"].isna().all()


# -----------------------------------------------------------------------
# 17-20: Additional edge cases
# -----------------------------------------------------------------------


class TestDividendGrowthEdgeCases:
    """Additional edge cases for dividend growth calculations."""

    def test_negative_dps_to_positive(self):
        """Negative DPS (unusual) transitioning to positive."""
        mgr = _make_mgr()
        n = 260
        dps = np.full(n, -1.0, dtype=float)
        dps[252:] = 1.0
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        # (1.0 - (-1.0)) / abs(-1.0) = 2.0
        assert result["fund_div_growth_yoy"].iloc[252] == pytest.approx(2.0)

    def test_dps_to_zero(self):
        """DPS declining from positive to zero (dividend cut)."""
        mgr = _make_mgr()
        n = 260
        dps = np.full(n, 2.0, dtype=float)
        dps[252:] = 0.0
        df = _make_df(n, dps=dps)
        result = mgr._compute_dividend_growth(df)
        # (0.0 - 2.0) / abs(2.0) = -1.0
        assert result["fund_div_growth_yoy"].iloc[252] == pytest.approx(-1.0)

    def test_payout_ratio_100_percent(self):
        """Payout ratio is exactly 1.0 when paying all earnings as dividends."""
        mgr = _make_mgr()
        n = 10
        df = _make_df(n, div_paid=np.full(n, 100.0), ni_ttm=np.full(n, 100.0))
        result = mgr._compute_dividend_growth(df)
        np.testing.assert_allclose(result["fund_payout_ratio"].values, 1.0)

    def test_returns_same_dataframe(self):
        """Method returns the same DataFrame (mutated in place)."""
        mgr = _make_mgr()
        df = _make_df(10, dps=np.full(10, 1.0))
        result = mgr._compute_dividend_growth(df)
        assert result is df
        assert "fund_div_growth_yoy" in result.columns
        assert "fund_payout_ratio" in result.columns
