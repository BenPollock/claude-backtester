"""Tests for calendar-based analytics (analytics/calendar.py)."""

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.calendar import (
    monthly_returns,
    drawdown_periods,
    yearly_summary,
    print_calendar_report,
    plot_monthly_heatmap,
)


def make_equity(values, start="2020-01-02"):
    """Build an equity Series from a list of values."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(
        values,
        index=pd.DatetimeIndex(dates.date, name="Date"),
        name="Equity",
    )


def make_long_equity(days=504, start="2020-01-02", start_price=10000.0):
    """Build a ~2-year equity curve with a realistic drawdown embedded.

    The curve rises for the first half, drops sharply, then recovers.
    This guarantees at least one completed drawdown period.
    """
    dates = pd.bdate_range(start=start, periods=days)
    rng = np.random.default_rng(99)
    prices = [start_price]
    for i in range(1, days):
        if i < days // 3:
            # Rising phase
            ret = 0.001 + rng.normal(0, 0.005)
        elif i < days // 2:
            # Drawdown phase
            ret = -0.003 + rng.normal(0, 0.005)
        else:
            # Recovery phase
            ret = 0.002 + rng.normal(0, 0.005)
        prices.append(prices[-1] * (1 + ret))
    return pd.Series(
        prices,
        index=pd.DatetimeIndex(dates.date, name="Date"),
        name="Equity",
    )


class TestMonthlyReturns:
    def test_shape_year_rows_month_columns(self):
        """monthly_returns should have Year rows, month columns 1-12, plus YTD."""
        equity = make_long_equity()
        mr = monthly_returns(equity)
        # Should have at least 2 years of data
        assert len(mr) >= 2
        # Columns should include months and YTD
        assert "YTD" in mr.columns

    def test_month_columns_are_integers_1_to_12(self):
        """Month columns should be numbered 1-12."""
        equity = make_long_equity()
        mr = monthly_returns(equity)
        month_cols = [c for c in mr.columns if c != "YTD"]
        assert month_cols == list(range(1, len(month_cols) + 1))
        # The first year (2020) should have up to 12 month columns
        assert len(month_cols) <= 12

    def test_values_correct_for_simple_curve(self):
        """For a known constant-growth equity, verify a monthly return is positive."""
        # 100 -> 110 linearly over ~63 bdays (about 3 months starting Jan 2020)
        values = np.linspace(100, 110, 63)
        equity = make_equity(values, start="2020-01-02")
        mr = monthly_returns(equity)
        # Year 2020 should exist as the only row
        assert 2020 in mr.index
        # January return should be positive (price went up)
        jan_ret = mr.loc[2020, 1]
        assert jan_ret > 0

    def test_ytd_is_compounded(self):
        """YTD should be the compounded product of monthly returns."""
        equity = make_long_equity()
        mr = monthly_returns(equity)
        for yr in mr.index:
            row = mr.loc[yr].drop("YTD").dropna()
            expected_ytd = (1 + row).prod() - 1
            assert abs(mr.loc[yr, "YTD"] - expected_ytd) < 1e-10


class TestDrawdownPeriods:
    def test_identifies_at_least_one_drawdown(self):
        """The synthetic curve with an embedded dip should produce drawdowns."""
        equity = make_long_equity()
        dp = drawdown_periods(equity)
        assert len(dp) >= 1

    def test_correct_columns(self):
        """Returned DataFrame should have the expected column names."""
        equity = make_long_equity()
        dp = drawdown_periods(equity)
        expected_cols = {"start", "trough", "recovery", "depth", "duration_days"}
        assert set(dp.columns) == expected_cols

    def test_top_n_limits_rows(self):
        """top_n=1 should return exactly one row."""
        equity = make_long_equity()
        dp = drawdown_periods(equity, top_n=1)
        assert len(dp) == 1

    def test_depth_is_negative(self):
        """Drawdown depth values should be negative (price fell below peak)."""
        equity = make_long_equity()
        dp = drawdown_periods(equity)
        for depth in dp["depth"]:
            assert depth < 0

    def test_duration_days_positive(self):
        """Duration should be a positive integer for any identified drawdown."""
        equity = make_long_equity()
        dp = drawdown_periods(equity)
        for d in dp["duration_days"]:
            assert d > 0

    def test_no_drawdown_on_monotonic_up(self):
        """A strictly monotonic equity curve should produce no drawdowns."""
        values = np.linspace(100, 200, 252)
        equity = make_equity(values)
        dp = drawdown_periods(equity)
        assert len(dp) == 0


class TestYearlySummary:
    def test_one_row_per_year(self):
        """yearly_summary should return one row per calendar year."""
        equity = make_long_equity()
        ys = yearly_summary(equity)
        years_in_data = pd.DatetimeIndex(equity.index).year.unique()
        assert len(ys) == len(years_in_data)

    def test_correct_column_names(self):
        """Columns should be: year, return, max_drawdown, sharpe, trading_days."""
        equity = make_long_equity()
        ys = yearly_summary(equity)
        expected = {"year", "return", "max_drawdown", "sharpe", "trading_days"}
        assert set(ys.columns) == expected

    def test_trading_days_reasonable(self):
        """Each year should have between 1 and 262 trading days.

        bdate_range generates all Mon-Fri days (no holiday exclusions),
        so a full calendar year can have up to ~261 business days.
        """
        equity = make_long_equity()
        ys = yearly_summary(equity)
        for _, row in ys.iterrows():
            assert 1 <= row["trading_days"] <= 262

    def test_max_drawdown_is_non_positive(self):
        """Max drawdown for any year should be <= 0."""
        equity = make_long_equity()
        ys = yearly_summary(equity)
        for _, row in ys.iterrows():
            assert row["max_drawdown"] <= 0


class TestPrintCalendarReport:
    def test_runs_without_error(self, capsys):
        """print_calendar_report should produce output containing 'Monthly Returns'."""
        equity = make_long_equity()
        print_calendar_report(equity)
        captured = capsys.readouterr()
        assert "Monthly Returns" in captured.out

    def test_output_contains_yearly_summary(self, capsys):
        """Output should include the yearly summary section."""
        equity = make_long_equity()
        print_calendar_report(equity)
        captured = capsys.readouterr()
        assert "Yearly Summary" in captured.out

    def test_output_contains_drawdown_section(self, capsys):
        """Output should include the drawdown periods section."""
        equity = make_long_equity()
        print_calendar_report(equity)
        captured = capsys.readouterr()
        assert "Drawdown Periods" in captured.out


class TestDrawdownPeriodsOngoing:
    """Test drawdown_periods with an ongoing (unrecovered) drawdown."""

    def test_ongoing_drawdown_has_none_recovery(self):
        """An unrecovered drawdown should have recovery=None."""
        # Price rises then falls — never recovers
        values = [100, 110, 105, 95, 90, 85]
        equity = make_equity(values)
        dp = drawdown_periods(equity)
        assert len(dp) >= 1
        # The last drawdown should have recovery=None since we end underwater
        last = dp.iloc[-1]
        assert last["recovery"] is None

    def test_single_day_no_crash(self):
        """drawdown_periods with a single-value equity should not crash."""
        equity = make_equity([100])
        dp = drawdown_periods(equity)
        assert dp.empty

    def test_two_values_with_decline(self):
        """Two values: peak then decline = one drawdown period."""
        equity = make_equity([100, 90])
        dp = drawdown_periods(equity)
        assert len(dp) == 1
        assert dp.iloc[0]["depth"] == pytest.approx(-0.1, rel=1e-6)


class TestYearlySummaryEdgeCases:
    """Edge cases for yearly_summary."""

    def test_single_year_data(self):
        """Data within a single year should produce exactly 1 row."""
        values = np.linspace(100, 120, 50)
        equity = make_equity(values, start="2020-06-01")
        ys = yearly_summary(equity)
        assert len(ys) == 1
        assert ys.iloc[0]["year"] == 2020

    def test_sharpe_sign_matches_return(self):
        """Positive year return should have positive Sharpe."""
        values = np.linspace(100, 130, 252)
        equity = make_equity(values, start="2020-01-02")
        ys = yearly_summary(equity)
        for _, row in ys.iterrows():
            if row["return"] > 0:
                assert row["sharpe"] > 0


class TestMonthlyReturnsEdgeCases:
    """Edge cases for monthly_returns."""

    def test_flat_equity_zero_returns(self):
        """Flat equity produces 0% monthly returns."""
        values = [100.0] * 50
        equity = make_equity(values, start="2020-01-02")
        mr = monthly_returns(equity)
        if not mr.empty:
            month_cols = [c for c in mr.columns if c != "YTD"]
            for c in month_cols:
                v = mr.iloc[0].get(c)
                if pd.notna(v):
                    assert v == pytest.approx(0.0, abs=1e-10)

    def test_empty_input_returns_empty(self):
        """A 1-value equity series (no returns) produces empty table."""
        equity = make_equity([100])
        mr = monthly_returns(equity)
        assert mr.empty


class TestPlotMonthlyHeatmap:
    def test_does_not_raise(self):
        """plot_monthly_heatmap should not raise with Agg backend."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        equity = make_long_equity()
        plot_monthly_heatmap(equity)
        plt.close("all")

    def test_empty_equity(self):
        """Plotting with a very short series should not crash."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        equity = make_equity([100, 100])
        # monthly_returns may return a small table; just ensure no exception
        plot_monthly_heatmap(equity)
        plt.close("all")
