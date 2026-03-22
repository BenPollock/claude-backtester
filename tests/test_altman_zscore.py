"""Comprehensive unit tests for EdgarDataManager._compute_altman_z."""

import numpy as np
import pandas as pd
import pytest


def _make_mgr():
    """Create a minimal EdgarDataManager without calling __init__."""
    from backtester.data.fundamental import EdgarDataManager

    return EdgarDataManager.__new__(EdgarDataManager)


def _base_df(n=1, **overrides):
    """Build a DataFrame with all columns needed by _compute_altman_z.

    Default values produce a known, easy-to-verify Z-score in the safe zone.
    Override any column by passing keyword arguments.
    """
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    data = {
        "Close": 50.0,
        "fund_total_assets": 1000.0,
        "fund_retained_earnings": 200.0,
        "fund_operating_income": 100.0,
        "fund_total_liabilities": 500.0,
        "fund_revenue": 800.0,
        "fund_current_assets": 400.0,
        "fund_current_liabilities": 200.0,
        "fund_shares_outstanding": 100.0,
    }
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


def _expected_z(
    ca=400.0,
    cl=200.0,
    ta=1000.0,
    re=200.0,
    oi=100.0,
    close=50.0,
    shares=100.0,
    tl=500.0,
    rev=800.0,
):
    """Manually compute the Altman Z-Score from raw inputs."""
    wc = ca - cl
    mkt_cap = close * shares
    x1 = 1.2 * wc / ta
    x2 = 1.4 * re / ta
    x3 = 3.3 * oi / ta
    x4 = 0.6 * mkt_cap / tl
    x5 = 1.0 * rev / ta
    return x1 + x2 + x3 + x4 + x5


# ── 1. Safe zone (Z > 2.99) with known values ────────────────────────

class TestSafeZone:
    def test_safe_zone_default_values(self):
        mgr = _make_mgr()
        df = _base_df()
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        assert z > 2.99
        assert result["fund_altman_zone"].iloc[0] == "safe"


# ── 2. Grey zone (1.8 <= Z <= 2.99) ──────────────────────────────────

class TestGreyZone:
    def test_grey_zone(self):
        """Tune inputs so Z falls between 1.8 and 2.99.

        x1 = 1.2*(250-200)/1000 = 0.06
        x2 = 1.4*50/1000        = 0.07
        x3 = 3.3*20/1000        = 0.066
        x4 = 0.6*(10*5)/500     = 0.06
        x5 = 1.0*2000/1000      = 2.0
        Z  = 2.256
        """
        mgr = _make_mgr()
        df = _base_df(
            fund_revenue=2000.0,
            fund_operating_income=20.0,
            fund_retained_earnings=50.0,
            fund_current_assets=250.0,
            fund_current_liabilities=200.0,
            Close=10.0,
            fund_shares_outstanding=5.0,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        assert 1.8 <= z <= 2.99, f"Expected grey zone, got Z={z}"
        assert result["fund_altman_zone"].iloc[0] == "grey"


# ── 3. Distress zone (Z < 1.8) ───────────────────────────────────────

class TestDistressZone:
    def test_distress_zone(self):
        """Create a weak company profile: negative WC, low earnings."""
        mgr = _make_mgr()
        df = _base_df(
            fund_current_assets=100.0,
            fund_current_liabilities=500.0,
            fund_retained_earnings=-200.0,
            fund_operating_income=-50.0,
            fund_revenue=200.0,
            fund_total_liabilities=2000.0,
            fund_shares_outstanding=10.0,
            Close=10.0,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        assert z < 1.8, f"Expected distress zone, got Z={z}"
        assert result["fund_altman_zone"].iloc[0] == "distress"


# ── 4. Known manual calculation verification ─────────────────────────

class TestManualCalculation:
    def test_matches_hand_calculation(self):
        mgr = _make_mgr()
        df = _base_df()
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        expected = _expected_z()
        np.testing.assert_almost_equal(z, expected, decimal=10)


# ── 5. All NaN data → NaN Z and NaN zone ─────────────────────────────

class TestAllNaN:
    def test_all_nan_yields_nan(self):
        """When ta=0 and tl=0, every component gets NaN → overall NaN."""
        mgr = _make_mgr()
        df = _base_df(
            fund_total_assets=0.0,
            fund_total_liabilities=0.0,
        )
        result = mgr._compute_altman_z(df)
        assert np.isnan(result["fund_altman_z"].iloc[0])
        assert pd.isna(result["fund_altman_zone"].iloc[0])


# ── 6. Missing Close → returns NaN (early return) ────────────────────

class TestMissingClose:
    def test_missing_close_returns_nan(self):
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=1, freq="B")
        df = pd.DataFrame(
            {"fund_total_assets": [1000.0]},
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        assert "fund_altman_z" in result.columns
        assert np.isnan(result["fund_altman_z"].iloc[0])
        assert pd.isna(result["fund_altman_zone"].iloc[0])


# ── 7. Missing total_assets → returns NaN (early return) ─────────────

class TestMissingTotalAssets:
    def test_missing_total_assets_returns_nan(self):
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=1, freq="B")
        df = pd.DataFrame(
            {"Close": [50.0]},
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        assert "fund_altman_z" in result.columns
        assert np.isnan(result["fund_altman_z"].iloc[0])
        assert pd.isna(result["fund_altman_zone"].iloc[0])


# ── 8. Division by zero: total_assets=0 → NaN for x1/x2/x3/x5 ──────

class TestDivByZeroTotalAssets:
    def test_total_assets_zero_x4_survives(self):
        """When total_assets=0, x1/x2/x3/x5 are NaN but x4 still computes."""
        mgr = _make_mgr()
        df = _base_df(fund_total_assets=0.0)
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        # x4 = 0.6 * (50*100) / 500 = 6.0
        expected_x4 = 0.6 * (50.0 * 100.0) / 500.0
        np.testing.assert_almost_equal(z, expected_x4, decimal=10)


# ── 9. Division by zero: total_liabilities=0 → NaN for x4 ────────────

class TestDivByZeroTotalLiabilities:
    def test_total_liabilities_zero(self):
        """When total_liabilities=0, x4 is NaN; other components survive."""
        mgr = _make_mgr()
        df = _base_df(fund_total_liabilities=0.0)
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        ta = 1000.0
        x1 = 1.2 * (400.0 - 200.0) / ta
        x2 = 1.4 * 200.0 / ta
        x3 = 3.3 * 100.0 / ta
        x5 = 1.0 * 800.0 / ta
        expected = x1 + x2 + x3 + x5
        np.testing.assert_almost_equal(z, expected, decimal=10)


# ── 10. Partial components (some None) → sum of available ─────────────

class TestPartialComponents:
    def test_only_revenue_and_total_assets(self):
        """Only revenue and total_assets present; only x5 contributes."""
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=1, freq="B")
        df = pd.DataFrame(
            {
                "Close": [50.0],
                "fund_total_assets": [1000.0],
                "fund_revenue": [800.0],
            },
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        expected_x5 = 1.0 * 800.0 / 1000.0
        np.testing.assert_almost_equal(z, expected_x5, decimal=10)

    def test_missing_current_assets_and_liabilities(self):
        """When current_assets/current_liabilities missing, x1 is NaN."""
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=1, freq="B")
        df = pd.DataFrame(
            {
                "Close": [50.0],
                "fund_total_assets": [1000.0],
                "fund_retained_earnings": [200.0],
                "fund_operating_income": [100.0],
                "fund_total_liabilities": [500.0],
                "fund_revenue": [800.0],
                "fund_shares_outstanding": [100.0],
            },
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        ta = 1000.0
        x2 = 1.4 * 200.0 / ta
        x3 = 3.3 * 100.0 / ta
        x4 = 0.6 * (50.0 * 100.0) / 500.0
        x5 = 1.0 * 800.0 / ta
        expected = x2 + x3 + x4 + x5
        np.testing.assert_almost_equal(z, expected, decimal=10)


# ── 11. Zone boundary at exactly 2.99 → "grey" ───────────────────────

class TestBoundary299:
    def test_exactly_2_99_is_grey(self):
        """Z == 2.99 should be classified as 'grey' (<= 2.99)."""
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=1, freq="B")
        # Only x5 survives: Z = 1.0 * rev / ta = 2990/1000 = 2.99
        df = pd.DataFrame(
            {
                "Close": [50.0],
                "fund_total_assets": [1000.0],
                "fund_revenue": [2990.0],
            },
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        np.testing.assert_almost_equal(z, 2.99, decimal=10)
        assert result["fund_altman_zone"].iloc[0] == "grey"


# ── 12. Zone boundary at exactly 1.8 → "grey" ────────────────────────

class TestBoundary18:
    def test_exactly_1_8_is_grey(self):
        """Z == 1.8 should be classified as 'grey' (>= 1.8)."""
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=1, freq="B")
        # Only x5 survives: Z = 1.0 * rev / ta = 1800/1000 = 1.8
        df = pd.DataFrame(
            {
                "Close": [50.0],
                "fund_total_assets": [1000.0],
                "fund_revenue": [1800.0],
            },
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        np.testing.assert_almost_equal(z, 1.8, decimal=10)
        assert result["fund_altman_zone"].iloc[0] == "grey"


# ── 13. Negative working capital (distress signal) ───────────────────

class TestNegativeWorkingCapital:
    def test_negative_wc_pushes_z_down(self):
        """Large negative WC should drag Z into distress."""
        mgr = _make_mgr()
        df = _base_df(
            fund_current_assets=100.0,
            fund_current_liabilities=900.0,
            fund_retained_earnings=0.0,
            fund_operating_income=0.0,
            fund_revenue=0.0,
            fund_total_liabilities=2000.0,
            fund_shares_outstanding=10.0,
            Close=10.0,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        # x1 = 1.2 * (100-900)/1000 = -0.96
        # x2 = 0, x3 = 0
        # x4 = 0.6 * (10*10) / 2000 = 0.03
        # x5 = 0
        expected = 1.2 * (-800.0 / 1000.0) + 0.6 * (100.0 / 2000.0)
        np.testing.assert_almost_equal(z, expected, decimal=10)
        assert z < 0, "Negative WC with zero income should give negative Z"
        assert result["fund_altman_zone"].iloc[0] == "distress"


# ── 14. Large positive Z (healthy company) ───────────────────────────

class TestLargePositiveZ:
    def test_very_healthy_company(self):
        """A company with strong fundamentals should have Z >> 2.99."""
        mgr = _make_mgr()
        df = _base_df(
            fund_current_assets=5000.0,
            fund_current_liabilities=500.0,
            fund_retained_earnings=3000.0,
            fund_operating_income=2000.0,
            fund_revenue=5000.0,
            fund_total_liabilities=200.0,
            fund_shares_outstanding=1000.0,
            Close=100.0,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        expected = _expected_z(
            ca=5000.0,
            cl=500.0,
            ta=1000.0,
            re=3000.0,
            oi=2000.0,
            close=100.0,
            shares=1000.0,
            tl=200.0,
            rev=5000.0,
        )
        np.testing.assert_almost_equal(z, expected, decimal=10)
        assert z > 10, f"Expected Z >> 2.99, got {z}"
        assert result["fund_altman_zone"].iloc[0] == "safe"


# ── 15. Output column dtypes ─────────────────────────────────────────

class TestOutputDtypes:
    def test_fund_altman_z_is_float(self):
        mgr = _make_mgr()
        df = _base_df()
        result = mgr._compute_altman_z(df)
        assert result["fund_altman_z"].dtype == np.float64

    def test_fund_altman_zone_is_object(self):
        mgr = _make_mgr()
        df = _base_df()
        result = mgr._compute_altman_z(df)
        assert result["fund_altman_zone"].dtype == object


# ── 16. Multiple rows with different zones ────────────────────────────

class TestMultipleRows:
    def test_three_rows_three_zones(self):
        """Three rows engineered to land in safe, grey, and distress zones.

        Row 0 (safe):    Z = standard _base_df defaults -> ~7.45
        Row 1 (grey):    Z = 0.06 + 0.07 + 0.066 + 0.06 + 2.0 = 2.256
        Row 2 (distress): negative WC, negative earnings -> < 1.8
        """
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "Close": [50.0, 10.0, 10.0],
                "fund_total_assets": [1000.0, 1000.0, 1000.0],
                "fund_revenue": [800.0, 2000.0, 0.0],
                "fund_operating_income": [100.0, 20.0, -100.0],
                "fund_retained_earnings": [200.0, 50.0, -500.0],
                "fund_current_assets": [400.0, 250.0, 100.0],
                "fund_current_liabilities": [200.0, 200.0, 900.0],
                "fund_total_liabilities": [500.0, 500.0, 2000.0],
                "fund_shares_outstanding": [100.0, 5.0, 10.0],
            },
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        zones = result["fund_altman_zone"].tolist()
        zs = result["fund_altman_z"].tolist()

        assert zones[0] == "safe", f"Row 0: Z={zs[0]}, zone={zones[0]}"
        assert zones[1] == "grey", f"Row 1: Z={zs[1]}, zone={zones[1]}"
        assert zones[2] == "distress", f"Row 2: Z={zs[2]}, zone={zones[2]}"


# ── 17. Does not mutate input DataFrame ───────────────────────────────

class TestNoMutation:
    def test_returns_same_df(self):
        """_compute_altman_z returns the same DataFrame it was given."""
        mgr = _make_mgr()
        df = _base_df()
        result = mgr._compute_altman_z(df)
        assert result is df
        assert "fund_altman_z" in result.columns
        assert "fund_altman_zone" in result.columns


# ── 18. Just above distress boundary (1.7999) ─────────────────────────

class TestBoundaryJustBelow18:
    def test_just_below_1_8_is_distress(self):
        """Z = 1.799 should be 'distress' (< 1.8)."""
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=1, freq="B")
        # x5 = 1.0 * 1799 / 1000 = 1.799
        df = pd.DataFrame(
            {
                "Close": [50.0],
                "fund_total_assets": [1000.0],
                "fund_revenue": [1799.0],
            },
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        assert z < 1.8
        assert result["fund_altman_zone"].iloc[0] == "distress"


# ── 19. Just above safe boundary (3.0) ────────────────────────────────

class TestBoundaryJustAbove299:
    def test_just_above_2_99_is_safe(self):
        """Z = 3.0 should be 'safe' (> 2.99)."""
        mgr = _make_mgr()
        idx = pd.bdate_range("2024-01-02", periods=1, freq="B")
        # x5 = 1.0 * 3000 / 1000 = 3.0
        df = pd.DataFrame(
            {
                "Close": [50.0],
                "fund_total_assets": [1000.0],
                "fund_revenue": [3000.0],
            },
            index=idx,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        assert z > 2.99
        assert result["fund_altman_zone"].iloc[0] == "safe"


# ── 20. Negative Z-Score ──────────────────────────────────────────────

class TestNegativeZ:
    def test_negative_z_is_distress(self):
        """A negative Z-Score should be classified as 'distress'."""
        mgr = _make_mgr()
        df = _base_df(
            fund_current_assets=50.0,
            fund_current_liabilities=900.0,
            fund_retained_earnings=-500.0,
            fund_operating_income=-200.0,
            fund_revenue=100.0,
            fund_total_liabilities=2000.0,
            fund_shares_outstanding=1.0,
            Close=1.0,
        )
        result = mgr._compute_altman_z(df)
        z = result["fund_altman_z"].iloc[0]
        assert z < 0
        assert result["fund_altman_zone"].iloc[0] == "distress"
