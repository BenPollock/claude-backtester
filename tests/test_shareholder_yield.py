"""Tests for EdgarDataManager._compute_shareholder_yield."""

import numpy as np
import pandas as pd
import pytest


def _make_mgr():
    from backtester.data.fundamental import EdgarDataManager

    return EdgarDataManager.__new__(EdgarDataManager)


def _make_df(
    rows=3,
    close=100.0,
    shares=1_000_000.0,
    repurchased=None,
    issued=None,
    div_paid=None,
    stock_comp=None,
    include_close=True,
    include_shares=True,
):
    """Build a minimal DataFrame with the columns _compute_shareholder_yield expects.

    Scalar values are broadcast to every row.  Pass None to omit a column entirely.
    Pass a list/array to set per-row values.
    """
    idx = pd.bdate_range("2024-01-02", periods=rows, freq="B")
    df = pd.DataFrame(index=idx)

    if include_close:
        df["Close"] = close
    if include_shares:
        df["fund_shares_outstanding"] = shares

    if repurchased is not None:
        df["fund_stock_repurchased_ttm"] = repurchased
    if issued is not None:
        df["fund_stock_issued_proceeds_ttm"] = issued
    if div_paid is not None:
        df["fund_dividends_paid_ttm"] = div_paid
    if stock_comp is not None:
        df["fund_stock_comp_ttm"] = stock_comp

    return df


# ── 1. Positive buyback yield (company repurchasing net) ──────────────────

def test_positive_buyback_yield():
    """When repurchased > issued, buyback yield is positive."""
    mgr = _make_mgr()
    df = _make_df(repurchased=50_000, issued=10_000)
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 100.0 * 1_000_000  # 100M
    expected = (50_000 - 10_000) / mkt_cap
    np.testing.assert_allclose(result["fund_buyback_yield"].values, expected)


# ── 2. Negative buyback yield (company issuing net) ───────────────────────

def test_negative_buyback_yield():
    """When issued > repurchased, buyback yield is negative."""
    mgr = _make_mgr()
    df = _make_df(repurchased=10_000, issued=60_000)
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 100.0 * 1_000_000
    expected = (10_000 - 60_000) / mkt_cap
    assert expected < 0
    np.testing.assert_allclose(result["fund_buyback_yield"].values, expected)


# ── 3. Zero buyback (repurchased == issued) ───────────────────────────────

def test_zero_buyback_yield():
    """When repurchased equals issued, buyback yield is zero."""
    mgr = _make_mgr()
    df = _make_df(repurchased=25_000, issued=25_000)
    result = mgr._compute_shareholder_yield(df)

    np.testing.assert_allclose(result["fund_buyback_yield"].values, 0.0)


# ── 4. Shareholder yield with all components ──────────────────────────────

def test_shareholder_yield_all_components():
    """shareholder_yield = (net_buyback + dividends_paid - stock_comp) / mkt_cap."""
    mgr = _make_mgr()
    df = _make_df(
        repurchased=80_000,
        issued=20_000,
        div_paid=15_000,
        stock_comp=5_000,
    )
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 100.0 * 1_000_000
    net_buyback = 80_000 - 20_000
    expected_sh = (net_buyback + 15_000 - 5_000) / mkt_cap
    np.testing.assert_allclose(result["fund_shareholder_yield"].values, expected_sh)


# ── 5. Missing repurchased → treated as 0 ────────────────────────────────

def test_missing_repurchased_treated_as_zero():
    """Without fund_stock_repurchased_ttm column, repurchased defaults to 0."""
    mgr = _make_mgr()
    df = _make_df(issued=30_000)  # repurchased omitted
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 100.0 * 1_000_000
    expected = (0 - 30_000) / mkt_cap
    np.testing.assert_allclose(result["fund_buyback_yield"].values, expected)


# ── 6. Missing issued → treated as 0 ─────────────────────────────────────

def test_missing_issued_treated_as_zero():
    """Without fund_stock_issued_proceeds_ttm column, issued defaults to 0."""
    mgr = _make_mgr()
    df = _make_df(repurchased=40_000)  # issued omitted
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 100.0 * 1_000_000
    expected = 40_000 / mkt_cap
    np.testing.assert_allclose(result["fund_buyback_yield"].values, expected)


# ── 7. Missing dividends_paid → treated as 0 ─────────────────────────────

def test_missing_dividends_paid_treated_as_zero():
    """Missing dividends column means div contribution is 0 in shareholder yield."""
    mgr = _make_mgr()
    df = _make_df(repurchased=50_000, issued=10_000, stock_comp=3_000)
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 100.0 * 1_000_000
    net_buyback = 50_000 - 10_000
    expected_sh = (net_buyback + 0 - 3_000) / mkt_cap
    np.testing.assert_allclose(result["fund_shareholder_yield"].values, expected_sh)


# ── 8. Missing stock_comp → treated as 0 ─────────────────────────────────

def test_missing_stock_comp_treated_as_zero():
    """Missing stock comp column means sc contribution is 0 in shareholder yield."""
    mgr = _make_mgr()
    df = _make_df(repurchased=50_000, issued=10_000, div_paid=12_000)
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 100.0 * 1_000_000
    net_buyback = 50_000 - 10_000
    expected_sh = (net_buyback + 12_000 - 0) / mkt_cap
    np.testing.assert_allclose(result["fund_shareholder_yield"].values, expected_sh)


# ── 9. Missing Close → all NaN output ────────────────────────────────────

def test_missing_close_yields_nan():
    """If the Close column is absent, both yields are NaN."""
    mgr = _make_mgr()
    df = _make_df(repurchased=50_000, issued=10_000, include_close=False)
    result = mgr._compute_shareholder_yield(df)

    assert result["fund_buyback_yield"].isna().all()
    assert result["fund_shareholder_yield"].isna().all()


# ── 10. Missing shares_outstanding → all NaN output ──────────────────────

def test_missing_shares_outstanding_yields_nan():
    """If fund_shares_outstanding column is absent, both yields are NaN."""
    mgr = _make_mgr()
    df = _make_df(repurchased=50_000, issued=10_000, include_shares=False)
    result = mgr._compute_shareholder_yield(df)

    assert result["fund_buyback_yield"].isna().all()
    assert result["fund_shareholder_yield"].isna().all()


# ── 11. Zero market cap → NaN (division guard) ───────────────────────────

def test_zero_market_cap_close_zero():
    """Close=0 means mkt_cap=0; yields should be NaN (no division by zero)."""
    mgr = _make_mgr()
    df = _make_df(close=0.0, repurchased=50_000, issued=10_000)
    result = mgr._compute_shareholder_yield(df)

    assert result["fund_buyback_yield"].isna().all()
    assert result["fund_shareholder_yield"].isna().all()


def test_zero_market_cap_shares_zero():
    """shares_outstanding=0 means mkt_cap=0; yields should be NaN."""
    mgr = _make_mgr()
    df = _make_df(shares=0.0, repurchased=50_000, issued=10_000)
    result = mgr._compute_shareholder_yield(df)

    assert result["fund_buyback_yield"].isna().all()
    assert result["fund_shareholder_yield"].isna().all()


# ── 12. All NaN inputs → NaN output ──────────────────────────────────────

def test_all_financial_nan_yields_zero():
    """When Close and shares are present but all financial columns have NaN values,
    fillna(0) makes every component 0, so yields are 0 (not NaN)."""
    mgr = _make_mgr()
    idx = pd.bdate_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "Close": [100.0, 100.0, 100.0],
            "fund_shares_outstanding": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            "fund_stock_repurchased_ttm": [np.nan, np.nan, np.nan],
            "fund_stock_issued_proceeds_ttm": [np.nan, np.nan, np.nan],
            "fund_dividends_paid_ttm": [np.nan, np.nan, np.nan],
            "fund_stock_comp_ttm": [np.nan, np.nan, np.nan],
        },
        index=idx,
    )
    result = mgr._compute_shareholder_yield(df)

    # fillna(0) on every component → numerator=0, mkt_cap valid → yields = 0
    np.testing.assert_allclose(result["fund_buyback_yield"].values, 0.0)
    np.testing.assert_allclose(result["fund_shareholder_yield"].values, 0.0)


def test_nan_close_and_shares_values():
    """When Close and shares columns exist but hold NaN, mkt_cap is NaN → yields NaN."""
    mgr = _make_mgr()
    idx = pd.bdate_range("2024-01-02", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "Close": [np.nan, np.nan],
            "fund_shares_outstanding": [np.nan, np.nan],
            "fund_stock_repurchased_ttm": [50_000, 50_000],
        },
        index=idx,
    )
    result = mgr._compute_shareholder_yield(df)

    assert result["fund_buyback_yield"].isna().all()
    assert result["fund_shareholder_yield"].isna().all()


# ── 13. Correct column names ─────────────────────────────────────────────

def test_output_column_names():
    """Method must create fund_buyback_yield and fund_shareholder_yield columns."""
    mgr = _make_mgr()
    df = _make_df()
    result = mgr._compute_shareholder_yield(df)

    assert "fund_buyback_yield" in result.columns
    assert "fund_shareholder_yield" in result.columns


def test_returns_same_dataframe():
    """Method returns the same DataFrame (mutated in place)."""
    mgr = _make_mgr()
    df = _make_df()
    result = mgr._compute_shareholder_yield(df)
    assert result is df


# ── 14. Multiple rows with varying values ─────────────────────────────────

def test_multiple_rows_varying_values():
    """Each row should be computed independently with its own inputs."""
    mgr = _make_mgr()
    idx = pd.bdate_range("2024-01-02", periods=4, freq="B")
    df = pd.DataFrame(
        {
            "Close": [50.0, 100.0, 200.0, 150.0],
            "fund_shares_outstanding": [1_000_000, 2_000_000, 500_000, 1_000_000],
            "fund_stock_repurchased_ttm": [10_000, 0, 100_000, 50_000],
            "fund_stock_issued_proceeds_ttm": [5_000, 20_000, 0, 50_000],
            "fund_dividends_paid_ttm": [2_000, 3_000, 5_000, 0],
            "fund_stock_comp_ttm": [1_000, 1_000, 2_000, 0],
        },
        index=idx,
    )
    result = mgr._compute_shareholder_yield(df)

    mkt_caps = np.array([50e6, 200e6, 100e6, 150e6])
    net_bb = np.array([10_000 - 5_000, 0 - 20_000, 100_000 - 0, 50_000 - 50_000])
    divs = np.array([2_000, 3_000, 5_000, 0])
    sc = np.array([1_000, 1_000, 2_000, 0])

    expected_bb = net_bb / mkt_caps
    expected_sh = (net_bb + divs - sc) / mkt_caps

    np.testing.assert_allclose(result["fund_buyback_yield"].values, expected_bb)
    np.testing.assert_allclose(result["fund_shareholder_yield"].values, expected_sh)


# ── 15. Large values (realistic magnitudes) ──────────────────────────────

def test_large_realistic_values():
    """Apple-scale: ~3T market cap, billions in buybacks."""
    mgr = _make_mgr()
    df = _make_df(
        rows=1,
        close=190.0,
        shares=15_500_000_000,           # 15.5B shares
        repurchased=77_000_000_000,      # $77B repurchased (TTM)
        issued=1_000_000_000,            # $1B issued
        div_paid=15_000_000_000,         # $15B dividends
        stock_comp=10_000_000_000,       # $10B stock comp
    )
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 190.0 * 15_500_000_000  # 2.945T
    net_bb = 77e9 - 1e9              # 76B
    expected_bb = net_bb / mkt_cap
    expected_sh = (net_bb + 15e9 - 10e9) / mkt_cap

    np.testing.assert_allclose(
        result["fund_buyback_yield"].values, expected_bb, rtol=1e-10
    )
    np.testing.assert_allclose(
        result["fund_shareholder_yield"].values, expected_sh, rtol=1e-10
    )


# ── Additional edge cases ────────────────────────────────────────────────

def test_mixed_nan_and_valid_rows():
    """Rows where mkt_cap is valid get a yield; rows with NaN mkt_cap get NaN."""
    mgr = _make_mgr()
    idx = pd.bdate_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "Close": [100.0, np.nan, 100.0],
            "fund_shares_outstanding": [1_000_000, 1_000_000, np.nan],
            "fund_stock_repurchased_ttm": [50_000, 50_000, 50_000],
        },
        index=idx,
    )
    result = mgr._compute_shareholder_yield(df)

    # Row 0: valid mkt_cap → valid yield
    assert not np.isnan(result["fund_buyback_yield"].iloc[0])
    expected_0 = 50_000 / (100.0 * 1_000_000)
    np.testing.assert_allclose(result["fund_buyback_yield"].iloc[0], expected_0)

    # Row 1: NaN Close → NaN mkt_cap → NaN yield
    assert np.isnan(result["fund_buyback_yield"].iloc[1])

    # Row 2: NaN shares → NaN mkt_cap → NaN yield
    assert np.isnan(result["fund_buyback_yield"].iloc[2])


# ── 16. Negative repurchased values ──────────────────────────────────────

def test_negative_repurchased_value():
    """Negative repurchased (unusual accounting) should still compute correctly."""
    mgr = _make_mgr()
    df = _make_df(repurchased=-10_000, issued=20_000)
    result = mgr._compute_shareholder_yield(df)

    mkt_cap = 100.0 * 1_000_000
    expected = (-10_000 - 20_000) / mkt_cap
    np.testing.assert_allclose(result["fund_buyback_yield"].values, expected)


# ── 17. Output dtypes are float ──────────────────────────────────────────

def test_output_dtypes_are_float():
    """Both yield columns should be float64 dtype."""
    mgr = _make_mgr()
    df = _make_df(repurchased=50_000, issued=10_000, div_paid=5_000, stock_comp=2_000)
    result = mgr._compute_shareholder_yield(df)

    assert result["fund_buyback_yield"].dtype == np.float64
    assert result["fund_shareholder_yield"].dtype == np.float64


# ── 18. All components zero → zero yields ────────────────────────────────

def test_all_components_zero():
    """When all financial components are 0, yields should be 0 (not NaN)."""
    mgr = _make_mgr()
    df = _make_df(repurchased=0, issued=0, div_paid=0, stock_comp=0)
    result = mgr._compute_shareholder_yield(df)

    np.testing.assert_allclose(result["fund_buyback_yield"].values, 0.0)
    np.testing.assert_allclose(result["fund_shareholder_yield"].values, 0.0)
