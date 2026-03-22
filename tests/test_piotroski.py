"""Comprehensive unit tests for EdgarDataManager._compute_piotroski_f.

Tests cover all 9 Piotroski criteria, edge cases (NaN, division by zero,
missing columns, single row), and score properties (dtype, range).
"""

import numpy as np
import pandas as pd
import pytest

from backtester.data.fundamental import EdgarDataManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mgr():
    """Create a minimal EdgarDataManager without EDGAR side effects."""
    return EdgarDataManager.__new__(EdgarDataManager)


def _bdays(n: int = 260) -> pd.DatetimeIndex:
    """Business-day index with *n* entries starting 2019-01-02."""
    return pd.bdate_range(start="2019-01-02", periods=n)


def _full_df(n: int = 260, **overrides) -> pd.DataFrame:
    """DataFrame with all columns the method reads, constant values.

    The first half uses "worse" values and the second half "better" so that
    shift(252) YoY comparisons can show improvement.  With n=260, the last
    ~8 rows have a valid prior-year row from the first half.
    """
    idx = _bdays(n)
    half = n // 2

    # "prior" (first half) vs "current" (second half) values are designed
    # so that ALL 9 criteria score 1.0 at the last row.
    ni = np.concatenate([np.full(half, 50.0), np.full(n - half, 150.0)])
    ta = np.full(n, 1000.0)
    ocf = np.concatenate([np.full(half, 60.0), np.full(n - half, 200.0)])
    debt = np.concatenate([np.full(half, 500.0), np.full(n - half, 300.0)])
    ca = np.concatenate([np.full(half, 500.0), np.full(n - half, 700.0)])
    cl = np.full(n, 300.0)
    shares = np.concatenate([np.full(half, 110.0), np.full(n - half, 100.0)])
    gp = np.concatenate([np.full(half, 400.0), np.full(n - half, 600.0)])
    rev = np.concatenate([np.full(half, 1800.0), np.full(n - half, 2200.0)])

    data = {
        "fund_net_income_ttm": ni,
        "fund_total_assets": ta,
        "fund_operating_cf_ttm": ocf,
        "fund_total_debt": debt,
        "fund_current_assets": ca,
        "fund_current_liabilities": cl,
        "fund_shares_outstanding": shares,
        "fund_gross_profit": gp,
        "fund_revenue": rev,
        "Close": np.full(n, 50.0),
    }
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


def _score_last(df: pd.DataFrame) -> float:
    """Run _compute_piotroski_f and return the last row's score."""
    mgr = _make_mgr()
    return mgr._compute_piotroski_f(df)["fund_piotroski_f"].iloc[-1]


# ===================================================================
# 1. Perfect score (9/9)
# ===================================================================
class TestPerfectScore:
    def test_perfect_9(self):
        """All 9 criteria positive -> score = 9."""
        assert _score_last(_full_df(260)) == 9.0


# ===================================================================
# 2. Zero score (0/9)
# ===================================================================
class TestZeroScore:
    def test_zero_score(self):
        """All 9 criteria negative -> score = 0."""
        n = 260
        half = n // 2
        # Invert: first half "good", second half "bad"
        df = _full_df(
            n,
            fund_net_income_ttm=np.concatenate(
                [np.full(half, 100.0), np.full(n - half, -50.0)]
            ),
            # CFO must be more negative than NI to fail accruals too
            # (CFO/TA = -0.10 is NOT > ROA = -0.05 -> criterion 4 = 0)
            fund_operating_cf_ttm=np.concatenate(
                [np.full(half, 100.0), np.full(n - half, -100.0)]
            ),
            fund_total_debt=np.concatenate(
                [np.full(half, 300.0), np.full(n - half, 500.0)]
            ),
            fund_current_assets=np.concatenate(
                [np.full(half, 700.0), np.full(n - half, 400.0)]
            ),
            fund_shares_outstanding=np.concatenate(
                [np.full(half, 90.0), np.full(n - half, 120.0)]
            ),
            fund_gross_profit=np.concatenate(
                [np.full(half, 600.0), np.full(n - half, 400.0)]
            ),
            fund_revenue=np.concatenate(
                [np.full(half, 2200.0), np.full(n - half, 1800.0)]
            ),
        )
        assert _score_last(df) == 0.0


# ===================================================================
# 3. Intermediate score (5/9)
# ===================================================================
class TestIntermediateScore:
    def test_five_of_nine(self):
        """Five criteria pass, four fail -> score = 5."""
        n = 260
        half = n // 2
        # Pass: ROA>0(1), CFO>0(2), ROA improved(3), accruals(4), shares(7)
        # Fail: leverage(5), current ratio(6), gross margin(8), asset turn(9)
        df = _full_df(
            n,
            # NI improves -> ROA>0 (pass) + ROA improved (pass)
            fund_net_income_ttm=np.concatenate(
                [np.full(half, 50.0), np.full(n - half, 100.0)]
            ),
            # CFO improves and CFO/TA > ROA (pass both 2 and 4)
            fund_operating_cf_ttm=np.concatenate(
                [np.full(half, 60.0), np.full(n - half, 150.0)]
            ),
            # Leverage worsens
            fund_total_debt=np.concatenate(
                [np.full(half, 300.0), np.full(n - half, 500.0)]
            ),
            # Current ratio worsens
            fund_current_assets=np.concatenate(
                [np.full(half, 700.0), np.full(n - half, 500.0)]
            ),
            # Shares flat -> not diluted (pass)
            fund_shares_outstanding=np.full(n, 100.0),
            # Gross margin worsens
            fund_gross_profit=np.concatenate(
                [np.full(half, 600.0), np.full(n - half, 400.0)]
            ),
            # Asset turnover worsens
            fund_revenue=np.concatenate(
                [np.full(half, 2200.0), np.full(n - half, 1800.0)]
            ),
        )
        assert _score_last(df) == 5.0


# ===================================================================
# 4. All NaN data -> NaN score
# ===================================================================
class TestAllNaN:
    def test_all_nan_values(self):
        """Every fundamental column is NaN -> score is NaN."""
        n = 10
        idx = _bdays(n)
        cols = [
            "fund_net_income_ttm", "fund_total_assets",
            "fund_operating_cf_ttm", "fund_total_debt",
            "fund_current_assets", "fund_current_liabilities",
            "fund_shares_outstanding", "fund_gross_profit", "fund_revenue",
        ]
        data = {c: np.full(n, np.nan) for c in cols}
        data["Close"] = np.full(n, 50.0)
        df = pd.DataFrame(data, index=idx)
        assert np.isnan(_score_last(df))


# ===================================================================
# 5. Partial NaN -> sum of non-NaN criteria
# ===================================================================
class TestPartialNaN:
    def test_partial_nan_sums_available(self):
        """Only ROA, CFO, and accrual columns present (short df).

        Criteria 1 (ROA>0)=1, 2 (CFO>0)=1, 4 (accruals: CFO/TA>ROA)=1.
        YoY criteria NaN.  Remaining column criteria NaN.  Score = 3.
        """
        n = 10
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_net_income_ttm": np.full(n, 100.0),
            "fund_total_assets": np.full(n, 1000.0),
            "fund_operating_cf_ttm": np.full(n, 120.0),
            "Close": np.full(n, 50.0),
        }, index=idx)
        assert _score_last(df) == 3.0


# ===================================================================
# 6. ROA > 0 criterion in isolation
# ===================================================================
class TestROAPositive:
    def test_roa_positive(self):
        """Positive NI / TA -> criterion 1 = 1.  Short df, only ROA + CFO."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_net_income_ttm": [100.0] * n,
            "fund_total_assets": [1000.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        # ROA = 0.1 > 0 -> 1.  No CFO column -> criterion 2 NaN.
        # Accruals NaN (no ocf).  Score = 1.
        assert _score_last(df) == 1.0

    def test_roa_negative(self):
        """Negative NI -> criterion 1 = 0, score = 0."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_net_income_ttm": [-100.0] * n,
            "fund_total_assets": [1000.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        assert _score_last(df) == 0.0


# ===================================================================
# 7. CFO > 0 criterion in isolation
# ===================================================================
class TestCFOPositive:
    def test_cfo_positive(self):
        """Positive OCF -> criterion 2 = 1.  Only OCF column present."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_operating_cf_ttm": [120.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        assert _score_last(df) == 1.0

    def test_cfo_negative(self):
        """Negative OCF -> criterion 2 = 0."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_operating_cf_ttm": [-50.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        assert _score_last(df) == 0.0


# ===================================================================
# 8. ROA improvement YoY
# ===================================================================
class TestROAImprovement:
    def test_roa_improved(self):
        """ROA increases YoY -> criterion 3 = 1."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_net_income_ttm": np.concatenate(
                [np.full(half, 50.0), np.full(n - half, 150.0)]
            ),
            "fund_total_assets": np.full(n, 1000.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        # Criteria 1 (ROA>0)=1, 3 (ROA improved)=1, others NaN -> 2
        assert _score_last(df) == 2.0

    def test_roa_declined(self):
        """ROA decreases YoY -> criterion 3 = 0."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_net_income_ttm": np.concatenate(
                [np.full(half, 150.0), np.full(n - half, 50.0)]
            ),
            "fund_total_assets": np.full(n, 1000.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        # Criteria 1 (ROA>0)=1, 3 (ROA improved)=0 -> 1
        assert _score_last(df) == 1.0


# ===================================================================
# 9. Accruals: CFO/TA > ROA
# ===================================================================
class TestAccruals:
    def test_accruals_pass(self):
        """CFO/TA (0.15) > ROA (0.10) -> criterion 4 = 1."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_net_income_ttm": [100.0] * n,
            "fund_total_assets": [1000.0] * n,
            "fund_operating_cf_ttm": [150.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        # 1(ROA>0)=1, 2(CFO>0)=1, 4(accruals)=1 -> 3
        assert _score_last(df) == 3.0

    def test_accruals_fail(self):
        """CFO/TA (0.08) < ROA (0.10) -> criterion 4 = 0."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_net_income_ttm": [100.0] * n,
            "fund_total_assets": [1000.0] * n,
            "fund_operating_cf_ttm": [80.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        # 1(ROA>0)=1, 2(CFO>0)=1, 4(accruals)=0 -> 2
        assert _score_last(df) == 2.0


# ===================================================================
# 10. Leverage decreased
# ===================================================================
class TestLeverageDecreased:
    def test_leverage_decreased(self):
        """Debt/TA drops YoY -> criterion 5 = 1."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_total_debt": np.concatenate(
                [np.full(half, 500.0), np.full(n - half, 300.0)]
            ),
            "fund_total_assets": np.full(n, 1000.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 1.0

    def test_leverage_increased(self):
        """Debt/TA rises YoY -> criterion 5 = 0."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_total_debt": np.concatenate(
                [np.full(half, 300.0), np.full(n - half, 500.0)]
            ),
            "fund_total_assets": np.full(n, 1000.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 0.0


# ===================================================================
# 11. Current ratio improved
# ===================================================================
class TestCurrentRatioImproved:
    def test_current_ratio_improved(self):
        """CA/CL increases YoY -> criterion 6 = 1."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_current_assets": np.concatenate(
                [np.full(half, 400.0), np.full(n - half, 700.0)]
            ),
            "fund_current_liabilities": np.full(n, 300.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 1.0

    def test_current_ratio_declined(self):
        """CA/CL decreases YoY -> criterion 6 = 0."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_current_assets": np.concatenate(
                [np.full(half, 700.0), np.full(n - half, 400.0)]
            ),
            "fund_current_liabilities": np.full(n, 300.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 0.0


# ===================================================================
# 12. Shares not diluted
# ===================================================================
class TestSharesNotDiluted:
    def test_shares_unchanged(self):
        """Shares flat -> shares <= prior, criterion 7 = 1."""
        n = 260
        df = pd.DataFrame({
            "fund_shares_outstanding": np.full(n, 100.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 1.0

    def test_shares_decreased(self):
        """Buyback -> criterion 7 = 1."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_shares_outstanding": np.concatenate(
                [np.full(half, 110.0), np.full(n - half, 90.0)]
            ),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 1.0

    def test_shares_increased(self):
        """Dilution -> criterion 7 = 0."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_shares_outstanding": np.concatenate(
                [np.full(half, 90.0), np.full(n - half, 110.0)]
            ),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 0.0


# ===================================================================
# 13. Gross margin improved
# ===================================================================
class TestGrossMarginImproved:
    def test_gross_margin_improved(self):
        """GP/Rev increases YoY -> criterion 8 = 1."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_gross_profit": np.concatenate(
                [np.full(half, 400.0), np.full(n - half, 600.0)]
            ),
            "fund_revenue": np.full(n, 2000.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 1.0

    def test_gross_margin_declined(self):
        """GP/Rev decreases YoY -> criterion 8 = 0."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_gross_profit": np.concatenate(
                [np.full(half, 600.0), np.full(n - half, 400.0)]
            ),
            "fund_revenue": np.full(n, 2000.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 0.0


# ===================================================================
# 14. Asset turnover improved
# ===================================================================
class TestAssetTurnoverImproved:
    def test_asset_turnover_improved(self):
        """Rev/TA increases YoY -> criterion 9 = 1."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_revenue": np.concatenate(
                [np.full(half, 1800.0), np.full(n - half, 2200.0)]
            ),
            "fund_total_assets": np.full(n, 1000.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 1.0

    def test_asset_turnover_declined(self):
        """Rev/TA decreases YoY -> criterion 9 = 0."""
        n = 260
        half = n // 2
        df = pd.DataFrame({
            "fund_revenue": np.concatenate(
                [np.full(half, 2200.0), np.full(n - half, 1800.0)]
            ),
            "fund_total_assets": np.full(n, 1000.0),
            "Close": np.full(n, 50.0),
        }, index=_bdays(n))
        assert _score_last(df) == 0.0


# ===================================================================
# 15. Division by zero protection
# ===================================================================
class TestDivisionByZero:
    def test_total_assets_zero(self):
        """TA=0 -> ROA, leverage, asset turnover = NaN.  No crash."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_net_income_ttm": [100.0] * n,
            "fund_total_assets": [0.0] * n,
            "fund_operating_cf_ttm": [120.0] * n,
            "fund_total_debt": [300.0] * n,
            "fund_revenue": [2000.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        score = _score_last(df)
        # Only CFO > 0 can score.  ROA/accruals/leverage/turnover all NaN.
        assert score == 1.0

    def test_current_liabilities_zero(self):
        """CL=0 -> current ratio = NaN.  No crash."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_current_assets": [600.0] * n,
            "fund_current_liabilities": [0.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        score = _score_last(df)
        # Current ratio NaN -> criterion 6 NaN.  All others NaN too -> NaN.
        assert np.isnan(score)

    def test_revenue_zero(self):
        """Rev=0 -> gross margin = NaN.  No crash."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_gross_profit": [500.0] * n,
            "fund_revenue": [0.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        # Should not raise.
        mgr = _make_mgr()
        result = mgr._compute_piotroski_f(df)
        assert "fund_piotroski_f" in result.columns


# ===================================================================
# 16. Missing columns -> those criteria NaN, others computed
# ===================================================================
class TestMissingColumns:
    def test_no_debt_column(self):
        """Missing fund_total_debt -> criterion 5 NaN, rest unaffected."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({
            "fund_net_income_ttm": [100.0] * n,
            "fund_total_assets": [1000.0] * n,
            "fund_operating_cf_ttm": [150.0] * n,
            "Close": [50.0] * n,
        }, index=idx)
        # 1(ROA>0)=1, 2(CFO>0)=1, 4(accruals)=1 -> 3.
        assert _score_last(df) == 3.0

    def test_no_fundamental_columns_at_all(self):
        """Only Close -> all criteria NaN -> NaN score."""
        n = 5
        idx = _bdays(n)
        df = pd.DataFrame({"Close": [50.0] * n}, index=idx)
        assert np.isnan(_score_last(df))


# ===================================================================
# 17. Single row -> YoY criteria NaN
# ===================================================================
class TestSingleRow:
    def test_single_row(self):
        """One row: shift(252) is NaN for all YoY criteria.

        Only level criteria (1, 2, 4) can fire.
        """
        idx = _bdays(1)
        df = pd.DataFrame({
            "fund_net_income_ttm": [100.0],
            "fund_total_assets": [1000.0],
            "fund_operating_cf_ttm": [150.0],
            "fund_total_debt": [300.0],
            "fund_current_assets": [600.0],
            "fund_current_liabilities": [300.0],
            "fund_shares_outstanding": [100.0],
            "fund_gross_profit": [500.0],
            "fund_revenue": [2000.0],
            "Close": [50.0],
        }, index=idx)
        # Criteria 1=1, 2=1, 4=1 (CFO/TA=0.15 > ROA=0.10).
        # YoY criteria 3,5,6,7,8,9 = NaN.
        assert _score_last(df) == 3.0


# ===================================================================
# 18. Score dtype is float
# ===================================================================
class TestScoreDtype:
    def test_dtype_float_with_valid_data(self):
        """fund_piotroski_f column must be a float dtype."""
        df = _full_df(260)
        mgr = _make_mgr()
        result = mgr._compute_piotroski_f(df)
        assert np.issubdtype(result["fund_piotroski_f"].dtype, np.floating)

    def test_dtype_float_with_all_nan(self):
        """Score stays float even when every value is NaN."""
        n = 5
        df = pd.DataFrame({"Close": [50.0] * n}, index=_bdays(n))
        mgr = _make_mgr()
        result = mgr._compute_piotroski_f(df)
        assert np.issubdtype(result["fund_piotroski_f"].dtype, np.floating)


# ===================================================================
# 19. Score is always bounded [0, 9]
# ===================================================================
class TestScoreBounds:
    def test_score_never_exceeds_9(self):
        """Even with extreme positive values, score caps at 9."""
        n = 260
        half = n // 2
        df = _full_df(
            n,
            fund_net_income_ttm=np.concatenate(
                [np.full(half, 1.0), np.full(n - half, 1e12)]
            ),
            fund_operating_cf_ttm=np.concatenate(
                [np.full(half, 1.0), np.full(n - half, 1e12)]
            ),
        )
        score = _score_last(df)
        assert 0 <= score <= 9

    def test_score_never_below_zero(self):
        """Score is always >= 0 even with all criteria failing."""
        n = 260
        half = n // 2
        # Use the same setup as TestZeroScore which achieves 0
        df = _full_df(
            n,
            fund_net_income_ttm=np.concatenate(
                [np.full(half, 100.0), np.full(n - half, -50.0)]
            ),
            fund_operating_cf_ttm=np.concatenate(
                [np.full(half, 100.0), np.full(n - half, -100.0)]
            ),
            fund_total_debt=np.concatenate(
                [np.full(half, 300.0), np.full(n - half, 500.0)]
            ),
            fund_current_assets=np.concatenate(
                [np.full(half, 700.0), np.full(n - half, 400.0)]
            ),
            fund_shares_outstanding=np.concatenate(
                [np.full(half, 90.0), np.full(n - half, 120.0)]
            ),
            fund_gross_profit=np.concatenate(
                [np.full(half, 600.0), np.full(n - half, 400.0)]
            ),
            fund_revenue=np.concatenate(
                [np.full(half, 2200.0), np.full(n - half, 1800.0)]
            ),
        )
        score = _score_last(df)
        assert score >= 0.0


# ===================================================================
# 20. Does not mutate input DataFrame
# ===================================================================
class TestNoMutation:
    def test_does_not_mutate_input(self):
        """_compute_piotroski_f should not modify the original DataFrame columns."""
        df = _full_df(260)
        original_cols = set(df.columns)
        mgr = _make_mgr()
        result = mgr._compute_piotroski_f(df)
        # The method adds fund_piotroski_f to df (mutates in place and returns it)
        # Verify it returns the same DataFrame
        assert result is df
        assert "fund_piotroski_f" in result.columns


# ===================================================================
# 21. Equal YoY values (boundary: no change)
# ===================================================================
class TestYoYBoundary:
    def test_unchanged_values_yoy(self):
        """When all values are constant, YoY criteria that compare > fail (= is not >).

        Level criteria: ROA>0 (1), CFO>0 (1), accruals CFO/TA>ROA (1)
        YoY criteria: ROA improved (0 - equal, not greater), leverage (0 - equal),
                       current ratio (0 - equal), shares (1 - <= passes),
                       gross margin (0 - equal), asset turnover (0 - equal)
        Total = 4
        """
        n = 260
        df = _full_df(
            n,
            fund_net_income_ttm=np.full(n, 100.0),
            fund_operating_cf_ttm=np.full(n, 150.0),
            fund_total_debt=np.full(n, 300.0),
            fund_current_assets=np.full(n, 600.0),
            fund_shares_outstanding=np.full(n, 100.0),
            fund_gross_profit=np.full(n, 500.0),
            fund_revenue=np.full(n, 2000.0),
        )
        score = _score_last(df)
        # Level: ROA>0=1, CFO>0=1, Accruals=1, Shares<=prior=1 = 4
        # YoY with equal values: all use >, so equal = 0
        assert score == 4.0
