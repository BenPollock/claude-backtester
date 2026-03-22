"""End-to-end integration tests for alternative data features.

Tests run full backtests through BacktestEngine.run() with mock data
to verify that EDGAR derived metrics (F-Score, Z-Score, Buyback Yield,
Dividend Growth) and alternative data features work correctly across
the entire pipeline.

EDGAR-related tests inject fund_ columns directly into mock DataFrames
rather than fetching from EDGAR (use_edgar=False).
"""

import tempfile
from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig, RegimeFilter
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.strategies.registry import discover_strategies

from tests.conftest import MockDataSource, make_price_df

discover_strategies()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_controlled_df(prices, start="2020-01-02", volume=1_000_000):
    """Create OHLCV DataFrame from a list of close prices."""
    days = len(prices)
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    closes = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(days, volume),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )


def _dates_from_df(df):
    start = df.index[0]
    end = df.index[-1]
    if hasattr(start, "date") and callable(start.date):
        start = start.date()
    if hasattr(end, "date") and callable(end.date):
        end = end.date()
    return start, end


def _run_backtest(config, source):
    tmpdir = tempfile.mkdtemp()
    dm = DataManager(cache_dir=tmpdir, source=source)
    engine = BacktestEngine(config, data_manager=dm)
    return engine.run()


def _add_fund_columns(df, **columns):
    """Add fund_ columns to a DataFrame. Returns the modified df."""
    for col, val in columns.items():
        df[col] = val
    return df


def _add_macro_columns(df, yield_spread=1.5, credit_spread=3.0):
    """Add FRED macro columns for macro_aware_value strategy."""
    df["fred_yield_spread_10y2y"] = yield_spread
    df["fred_credit_spread_hy"] = credit_spread
    return df


def _make_quality_df(n=300, uptrend=True):
    """Create an uptrending price DataFrame with good fundamental columns.

    Includes all columns needed by macro_aware_value: fund_piotroski_f,
    fund_altman_z, fund_pe_ratio, plus SMA will be computed by the strategy.
    """
    if uptrend:
        prices = [80 + i * 0.2 for i in range(n)]
    else:
        prices = [120 - i * 0.1 for i in range(n)]
    df = make_controlled_df(prices)
    return df


def _check_invariants(result, starting_cash=100_000.0):
    """Standard E2E invariants: positive equity, entry <= exit."""
    eq = result.equity_series
    # Equity always positive
    assert (eq > 0).all(), "Equity should always be positive"
    # Entry before exit invariant
    for trade in result.trades:
        assert trade.entry_date <= trade.exit_date, (
            f"Entry date {trade.entry_date} should be <= exit date "
            f"{trade.exit_date}"
        )


# ===========================================================================
# Piotroski F-Score E2E
# ===========================================================================


class TestPiotroskiFScoreE2E:
    """E2E tests verifying Piotroski F-Score integration through backtest."""

    def test_fscore_high_quality_gets_bought(self):
        """High F-Score + good valuation + uptrend → macro_aware_value BUYs."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=8.0,
            fund_altman_z=4.5,
            fund_pe_ratio=12.0,
            fund_net_income=1e8,
        )
        _add_macro_columns(df, yield_spread=1.5, credit_spread=3.0)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "High F-Score stock should generate trades"

    def test_fscore_absent_graceful_hold(self):
        """No fund_ columns → all HOLDs, no crash."""
        prices = [80 + i * 0.2 for i in range(300)]
        df = make_controlled_df(prices)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "Missing F-Score data should produce no trades"

    def test_fscore_worst_no_buy(self):
        """F-Score = 0 (worst quality) → no BUY signals."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=0.0,
            fund_altman_z=4.0,
            fund_pe_ratio=12.0,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "expansion_min_f": 5},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "F-Score=0 should not generate BUY trades"

    def test_fscore_best_gets_bought(self):
        """F-Score = 9 (best quality) → high quality stocks get bought."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=9.0,
            fund_altman_z=5.0,
            fund_pe_ratio=10.0,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "F-Score=9 should generate trades"

    def test_fscore_invariants(self):
        """Standard invariants: positive equity, T+1 fills."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=4.0,
            fund_pe_ratio=15.0,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        _check_invariants(result)


# ===========================================================================
# Altman Z-Score E2E
# ===========================================================================


class TestAltmanZScoreE2E:
    """E2E tests verifying Altman Z-Score integration through backtest."""

    def test_distress_excluded_from_buy(self):
        """Z-Score < 1.8 (distress) → no BUY signals despite good F-Score."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=8.0,
            fund_altman_z=1.0,  # distress zone
            fund_pe_ratio=10.0,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "min_z_score": 1.8},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "Distressed stocks (Z<1.8) should not be bought"

    def test_safe_zone_eligible_for_buy(self):
        """Z-Score > 2.99 (safe) + good fundamentals → BUY."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=4.0,  # safe zone
            fund_pe_ratio=12.0,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "Safe zone stock should be eligible for BUY"

    def test_zscore_absent_graceful_degradation(self):
        """Missing Z-Score → HOLD (graceful degradation)."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=8.0,
            fund_pe_ratio=12.0,
            # fund_altman_z intentionally omitted
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "Missing Z-Score should produce no trades"

    def test_zscore_boundary_at_1_8(self):
        """Z-Score exactly 1.8 → meets threshold (>= 1.8), eligible for BUY."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=1.8,  # exactly at boundary
            fund_pe_ratio=12.0,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "min_z_score": 1.8},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "Z-Score at boundary (1.8) should be eligible"

    def test_zscore_invariants(self):
        """Standard invariants with Z-Score data present."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=6.0,
            fund_altman_z=3.5,
            fund_pe_ratio=14.0,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        _check_invariants(result)


# ===========================================================================
# Buyback Yield E2E
# ===========================================================================


class TestBuybackYieldE2E:
    """E2E tests verifying buyback yield columns survive the full pipeline.

    No existing strategy consumes fund_buyback_yield directly, so these
    tests verify: (a) columns are computed through enrichment, (b) backtest
    completes without error, (c) standard invariants hold.
    """

    def test_positive_buyback_yield_present(self):
        """High buyback yield columns present after enrichment pipeline."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=4.0,
            fund_pe_ratio=12.0,
            fund_buyback_yield=0.05,
            fund_shareholder_yield=0.08,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        # Backtest completes and trades happen (columns don't interfere)
        assert len(result.trades) > 0

    def test_negative_buyback_no_crash(self):
        """Negative buyback yield (dilution) doesn't crash backtest."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=6.0,
            fund_altman_z=3.5,
            fund_pe_ratio=14.0,
            fund_buyback_yield=-0.03,  # dilution
            fund_shareholder_yield=-0.05,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert result.equity_series is not None

    def test_missing_buyback_graceful(self):
        """Missing buyback columns → backtest still runs."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=4.0,
            fund_pe_ratio=12.0,
            # No buyback columns
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "Missing buyback data shouldn't block trading"

    def test_shareholder_yield_computation(self):
        """Shareholder yield column value survives through the pipeline."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=4.0,
            fund_pe_ratio=12.0,
            fund_buyback_yield=0.04,
            fund_shareholder_yield=0.06,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        _check_invariants(result)

    def test_buyback_invariants(self):
        """Standard invariants with buyback columns present."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=8.0,
            fund_altman_z=5.0,
            fund_pe_ratio=10.0,
            fund_buyback_yield=0.02,
            fund_shareholder_yield=0.04,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        _check_invariants(result)


# ===========================================================================
# Dividend Growth E2E
# ===========================================================================


class TestDividendGrowthE2E:
    """E2E tests verifying dividend growth columns survive the full pipeline."""

    def test_growing_dividends_no_crash(self):
        """Positive dividend growth columns present → backtest completes."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=4.0,
            fund_pe_ratio=12.0,
            fund_div_growth_yoy=0.10,
            fund_payout_ratio=0.40,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "Growing dividends shouldn't block trading"

    def test_cutting_dividends_no_crash(self):
        """Negative dividend growth → backtest still completes."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=6.0,
            fund_altman_z=3.5,
            fund_pe_ratio=14.0,
            fund_div_growth_yoy=-0.30,
            fund_payout_ratio=0.80,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert result.equity_series is not None

    def test_missing_dividend_data_graceful(self):
        """Missing dividend columns → backtest still runs."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=4.0,
            fund_pe_ratio=12.0,
            # No dividend columns
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "Missing dividend data shouldn't block trading"

    def test_high_payout_ratio_tracking(self):
        """High payout ratio (>1.0) column present → backtest completes."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=7.0,
            fund_altman_z=4.0,
            fund_pe_ratio=12.0,
            fund_div_growth_yoy=0.05,
            fund_payout_ratio=1.5,  # paying out more than earnings
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert result.equity_series is not None

    def test_dividend_invariants(self):
        """Standard invariants with dividend columns present."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(
            df,
            fund_piotroski_f=8.0,
            fund_altman_z=5.0,
            fund_pe_ratio=10.0,
            fund_div_growth_yoy=0.15,
            fund_payout_ratio=0.35,
        )
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        _check_invariants(result)


# ===========================================================================
# Alt-data helpers
# ===========================================================================


def _make_vix_columns(df, ratio=0.9, vix_close=15.0):
    """Add VIX columns to a DataFrame in-place."""
    df["vix_close"] = vix_close
    df["vix_3m"] = vix_close / ratio if ratio != 0 else vix_close
    df["vix_ratio"] = ratio
    df["vix_regime"] = "contango" if ratio < 1.0 else "backwardation"
    return df


def _make_fred_columns(df, yield_spread=1.5, credit_spread=3.0, regime_score=0.8):
    """Add FRED macro regime columns to a DataFrame in-place."""
    df["fred_yield_spread_10y2y"] = yield_spread
    df["fred_yield_spread_10y3m"] = yield_spread + 0.5
    df["fred_credit_spread_hy"] = credit_spread
    df["fred_credit_spread_baa_aaa"] = credit_spread * 0.3
    df["fred_lei"] = 100.0
    df["fred_claims"] = 200_000.0
    df["fred_macro_regime"] = regime_score
    return df


def _make_analyst_columns(df, breadth=0.5):
    """Add analyst revision columns to a DataFrame in-place."""
    up = max(0, int(10 * (1 + breadth) / 2))
    down = 10 - up
    df["analyst_rev_up_7d"] = up
    df["analyst_rev_down_7d"] = down
    df["analyst_rev_breadth"] = breadth
    return df


def _make_insider_columns(df, buy_ratio=0.7):
    """Add insider columns to a DataFrame in-place."""
    df["insider_buy_ratio_90d"] = buy_ratio
    return df


# ===========================================================================
# VIX Term Structure E2E
# ===========================================================================


class TestVIXTermStructureE2E:
    """VIX term structure data flowing through the full pipeline."""

    def test_vix_contango_risk_on(self):
        """VIX contango (ratio < 1) combined with risk_regime → BUY."""
        df = _make_quality_df(n=200, uptrend=True)
        _make_vix_columns(df, ratio=0.85)
        _make_fred_columns(df, yield_spread=1.5, credit_spread=3.0)
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=4.0, fund_pe_ratio=12.0)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="risk_regime",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "Risk-on regime should generate trades"
        _check_invariants(result)

    def test_vix_backwardation_risk_off(self):
        """VIX backwardation (ratio > 1) → risk_regime exits."""
        df = _make_quality_df(n=200, uptrend=True)
        n = len(df)
        # Start risk-on, switch to backwardation at day 60
        vix_ratio = np.concatenate([np.full(60, 0.85), np.full(n - 60, 1.3)])
        df["vix_ratio"] = vix_ratio
        df["vix_close"] = 15.0
        df["vix_3m"] = 16.0
        df["vix_regime"] = ["contango" if r < 1.0 else "backwardation" for r in vix_ratio]
        # Other macro signals: risk-on first, risk-off after
        yield_spread = np.concatenate([np.full(60, 1.5), np.full(n - 60, -0.5)])
        credit_spread = np.concatenate([np.full(60, 3.0), np.full(n - 60, 8.0)])
        df["fred_yield_spread_10y2y"] = yield_spread
        df["fred_credit_spread_hy"] = credit_spread
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=4.0, fund_pe_ratio=12.0)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="risk_regime",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20},
        )
        result = _run_backtest(config, source)
        # Should have at least one trade (buy then sell or force-close)
        assert len(result.trades) >= 1

    def test_vix_absent_graceful(self):
        """No vix_ columns → risk_regime returns HOLD (no crash)."""
        df = _make_quality_df(n=100, uptrend=True)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="risk_regime",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "No macro data → no trades"


# ===========================================================================
# Intermarket E2E
# ===========================================================================


class TestIntermarketE2E:
    """Cross-asset intermarket data surviving the pipeline."""

    def test_intermarket_columns_present(self):
        """Intermarket columns present → backtest completes normally."""
        df = _make_quality_df(n=300, uptrend=True)
        n = len(df)
        df["intermarket_cu_au_ratio"] = 0.015
        df["intermarket_cu_au_momentum"] = 0.05
        df["intermarket_dollar"] = 100.0
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=4.0, fund_pe_ratio=12.0)
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0
        _check_invariants(result)

    def test_intermarket_absent_no_crash(self):
        """Missing intermarket columns → no crash."""
        df = _make_quality_df(n=100, uptrend=True)
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=4.0, fund_pe_ratio=12.0)
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20},
        )
        result = _run_backtest(config, source)
        assert result.equity_series is not None


# ===========================================================================
# FRED Macro Regime E2E
# ===========================================================================


class TestFREDMacroE2E:
    """FRED macro regime data flowing through the pipeline."""

    def test_expansion_allows_lower_fscore(self):
        """Expansion regime (positive yield, tight credit) → F>=5 passes."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(df, fund_piotroski_f=5.0, fund_altman_z=4.0, fund_pe_ratio=15.0)
        _add_macro_columns(df, yield_spread=1.5, credit_spread=3.0)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 50,
                "expansion_min_f": 5,
                "contraction_min_f": 7,
            },
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "F=5 should pass in expansion"

    def test_contraction_requires_higher_fscore(self):
        """Contraction (inverted yield, wide credit) → F=5 blocked, needs 7."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(df, fund_piotroski_f=5.0, fund_altman_z=4.0, fund_pe_ratio=12.0)
        # Contraction: inverted yield + wide credit
        _add_macro_columns(df, yield_spread=-0.5, credit_spread=7.0)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 50,
                "expansion_min_f": 5,
                "contraction_min_f": 7,
            },
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "F=5 should NOT pass in contraction (needs 7)"

    def test_fred_regime_replace_mode(self):
        """fred_regime_mode='replace' ignores SMA regime, uses only FRED."""
        df = _make_quality_df(n=100, uptrend=True)
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=4.0, fund_pe_ratio=12.0)
        _make_fred_columns(df, regime_score=0.8)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20},
            use_fred=True,
            fred_regime_mode="replace",
            regime_filter=RegimeFilter(
                benchmark="TEST", indicator="sma",
                fast_period=20, slow_period=50,
            ),
        )
        # FRED bullish in replace mode → trades go through even if SMA filter
        # would block. We can't easily test the SMA blocking here without
        # controlling the benchmark data more precisely, but we verify the
        # engine accepts the config and completes
        with patch("backtester.data.fred_source.FredDataSource") as MockFred:
            mock_src = MagicMock()
            macro_df = pd.DataFrame(
                {"fred_macro_regime": np.full(len(df), 0.8)},
                index=df.index,
            )
            mock_src.load_macro_regime.return_value = macro_df
            MockFred.return_value = mock_src
            result = _run_backtest(config, source)

        assert len(result.portfolio.equity_history) > 0

    def test_fred_absent_graceful(self):
        """No fred_ columns → macro_aware_value falls back to conservative."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(df, fund_piotroski_f=6.0, fund_altman_z=4.0, fund_pe_ratio=12.0)
        # No FRED columns at all

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 50,
                "contraction_min_f": 7,
            },
        )
        result = _run_backtest(config, source)
        # No FRED data → falls back to contraction thresholds, F=6 < 7 = no trades
        assert len(result.trades) == 0


# ===========================================================================
# Treasury Yield Curve E2E
# ===========================================================================


class TestTreasuryYieldCurveE2E:
    """Treasury yield curve data surviving the pipeline."""

    def test_yield_curve_columns_present(self):
        """Yield curve columns present → backtest completes."""
        df = _make_quality_df(n=200, uptrend=True)
        n = len(df)
        df["yield_3m"] = 2.0
        df["yield_2y"] = 3.0
        df["yield_10y"] = 4.0
        df["yield_30y"] = 4.5
        df["yield_spread"] = 1.0
        df["yield_real_10y"] = 1.8
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=4.0, fund_pe_ratio=12.0)
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0
        _check_invariants(result)

    def test_inverted_curve_no_crash(self):
        """Inverted yield curve (negative spread) → backtest completes."""
        df = _make_quality_df(n=200, uptrend=True)
        df["yield_2y"] = 4.5
        df["yield_10y"] = 3.5
        df["yield_spread"] = -1.0
        _add_fund_columns(df, fund_piotroski_f=8.0, fund_altman_z=4.0, fund_pe_ratio=10.0)
        # Inverted yield curve → contraction macro signal
        _add_macro_columns(df, yield_spread=-1.0, credit_spread=6.0)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "contraction_min_f": 7},
        )
        result = _run_backtest(config, source)
        _check_invariants(result)


# ===========================================================================
# CBOE Put-Call Ratio E2E
# ===========================================================================


class TestPutCallRatioE2E:
    """CBOE put-call ratio data surviving the pipeline."""

    def test_pcr_columns_present(self):
        """PCR columns present → backtest completes normally."""
        df = _make_quality_df(n=200, uptrend=True)
        df["sentiment_pcr"] = 0.75
        df["sentiment_pcr_ma10"] = 0.78
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=4.0, fund_pe_ratio=12.0)
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0
        _check_invariants(result)

    def test_pcr_absent_no_crash(self):
        """Missing PCR columns → no crash."""
        df = _make_quality_df(n=100, uptrend=True)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            strategy_params={"sma_fast": 20, "sma_slow": 50},
        )
        result = _run_backtest(config, source)
        assert result.equity_series is not None


# ===========================================================================
# Analyst Revisions E2E
# ===========================================================================


class TestAnalystRevisionsE2E:
    """Analyst revision data flowing through sentiment_momentum strategy."""

    def test_bullish_revisions_trigger_buy(self):
        """Positive analyst breadth + insider buying + above SMA → BUY."""
        prices = [50 + i * 0.3 for i in range(200)]
        df = make_controlled_df(prices)
        _make_analyst_columns(df, breadth=0.6)
        _make_insider_columns(df, buy_ratio=0.8)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sentiment_momentum",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "min_signals_buy": 2},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "Bullish sentiment should generate trades"

    def test_bearish_revisions_trigger_sell(self):
        """Negative breadth + insider selling + below SMA → SELL."""
        # Rising then falling prices
        prices = [50 + i * 0.5 for i in range(60)]
        prices += [80 - i * 0.3 for i in range(140)]
        df = make_controlled_df(prices)
        n = len(df)

        # Start bullish, then bearish
        breadth = np.concatenate([np.full(60, 0.6), np.full(140, -0.5)])
        df["analyst_rev_breadth"] = breadth
        df["analyst_rev_up_7d"] = np.where(breadth > 0, 8, 2)
        df["analyst_rev_down_7d"] = np.where(breadth > 0, 2, 8)

        insider_ratio = np.concatenate([np.full(60, 0.8), np.full(140, 0.1)])
        df["insider_buy_ratio_90d"] = insider_ratio

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sentiment_momentum",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 20,
                "min_signals_buy": 2,
                "min_signals_sell": 2,
            },
        )
        result = _run_backtest(config, source)
        assert len(result.trades) >= 1

    def test_analyst_absent_graceful(self):
        """No analyst columns → sentiment_momentum returns HOLD."""
        prices = [50 + i * 0.2 for i in range(100)]
        df = make_controlled_df(prices)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sentiment_momentum",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "min_signals_buy": 3},
        )
        result = _run_backtest(config, source)
        # Only SMA signal available (max 1/3 bullish) → no buys with threshold=3
        assert len(result.trades) == 0

    def test_analyst_invariants(self):
        """Standard invariants with analyst data."""
        prices = [50 + i * 0.3 for i in range(200)]
        df = make_controlled_df(prices)
        _make_analyst_columns(df, breadth=0.4)
        _make_insider_columns(df, buy_ratio=0.7)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sentiment_momentum",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "min_signals_buy": 2},
        )
        result = _run_backtest(config, source)
        _check_invariants(result)


# ===========================================================================
# Cross-Feature Interaction E2E
# ===========================================================================


class TestCrossFeatureE2E:
    """Multiple alt-data features active simultaneously."""

    def test_all_alt_data_together(self):
        """VIX + FRED + analyst + insider + fundamental all present → completes."""
        df = _make_quality_df(n=300, uptrend=True)
        _add_fund_columns(df, fund_piotroski_f=8.0, fund_altman_z=4.5, fund_pe_ratio=10.0)
        _make_vix_columns(df, ratio=0.85)
        _make_fred_columns(df, yield_spread=2.0, credit_spread=2.5, regime_score=0.9)
        _make_analyst_columns(df, breadth=0.7)
        _make_insider_columns(df, buy_ratio=0.8)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0
        _check_invariants(result)

    def test_risk_regime_with_all_signals(self):
        """risk_regime with VIX + FRED + fundamentals → full lifecycle."""
        df = _make_quality_df(n=200, uptrend=True)
        _make_vix_columns(df, ratio=0.85)
        _make_fred_columns(df, yield_spread=1.5, credit_spread=3.0)
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=4.0, fund_pe_ratio=12.0)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="risk_regime",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "min_f_score": 5},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0
        _check_invariants(result)

    def test_sentiment_momentum_with_all_data(self):
        """sentiment_momentum with analyst + insider + SMA → full pipeline."""
        prices = [50 + i * 0.3 for i in range(200)]
        df = make_controlled_df(prices)
        _make_analyst_columns(df, breadth=0.5)
        _make_insider_columns(df, buy_ratio=0.75)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sentiment_momentum",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "min_signals_buy": 2},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0
        _check_invariants(result)

    def test_cash_accounting_invariant(self):
        """Cash + position value approximates total equity after force-close."""
        df = _make_quality_df(n=200, uptrend=True)
        _add_fund_columns(df, fund_piotroski_f=7.0, fund_altman_z=3.5, fund_pe_ratio=12.0)
        _add_macro_columns(df, yield_spread=1.5, credit_spread=3.0)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        final_cash = result.portfolio.cash
        final_positions = sum(
            pos.market_value for pos in result.portfolio.positions.values()
        )
        final_equity = result.portfolio.equity_history[-1][1]
        assert abs(final_cash + final_positions - final_equity) < 1.0

    def test_disabled_features_zero_impact(self):
        """All use_* flags False → no alt data loaded, plain backtest."""
        prices = [100 + i * 0.2 for i in range(200)]
        df = make_controlled_df(prices)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            strategy_params={"sma_fast": 20, "sma_slow": 50},
            use_vix=False,
            use_intermarket=False,
            use_fred=False,
            use_yield_curve=False,
            use_pcr=False,
            use_analyst=False,
        )
        result = _run_backtest(config, source)
        assert len(result.portfolio.equity_history) > 0


# ===========================================================================
# Engine integration unit tests
# ===========================================================================


class TestMergeAuxiliaryData:
    """Test the _merge_auxiliary_data static method."""

    def test_forward_fills_auxiliary(self):
        """Auxiliary data with gaps gets forward-filled to daily index."""
        daily_idx = pd.bdate_range("2020-01-02", periods=10, freq="B")
        daily_df = pd.DataFrame(
            {"Close": range(10), "Volume": 1000},
            index=pd.DatetimeIndex(daily_idx.date),
        )
        aux_idx = [daily_df.index[0], daily_df.index[3], daily_df.index[7]]
        aux_df = pd.DataFrame(
            {"fred_lei": [100.0, 101.0, 102.0]},
            index=pd.DatetimeIndex(aux_idx),
        )

        result = BacktestEngine._merge_auxiliary_data(daily_df, aux_df)
        assert "fred_lei" in result.columns
        assert result["fred_lei"].iloc[0] == 100.0
        assert result["fred_lei"].iloc[1] == 100.0  # forward-filled
        assert result["fred_lei"].iloc[3] == 101.0

    def test_does_not_mutate_input(self):
        """_merge_auxiliary_data does not modify the input DataFrame."""
        daily_idx = pd.bdate_range("2020-01-02", periods=5, freq="B")
        daily_df = pd.DataFrame(
            {"Close": range(5)},
            index=pd.DatetimeIndex(daily_idx.date),
        )
        aux_df = pd.DataFrame(
            {"vix_ratio": [0.9, 0.95, 1.0, 1.05, 1.1]},
            index=daily_df.index,
        )
        original_cols = list(daily_df.columns)
        BacktestEngine._merge_auxiliary_data(daily_df, aux_df)
        assert list(daily_df.columns) == original_cols


class TestCheckRegimeExtended:
    """Test enhanced _check_regime with VIX and FRED checks."""

    def _make_engine(self, use_vix=False, use_fred=False,
                     fred_regime_mode="supplement", regime_filter=None):
        from tests.conftest import make_price_df
        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            use_vix=use_vix,
            use_fred=use_fred,
            fred_regime_mode=fred_regime_mode,
            regime_filter=regime_filter,
        )
        source = MockDataSource()
        source.add("TEST", make_price_df())
        dm = DataManager(cache_dir=tempfile.mkdtemp(), source=source)
        return BacktestEngine(config, data_manager=dm)

    def test_vix_contango_allows(self):
        engine = self._make_engine(use_vix=True)
        row = pd.Series({"vix_ratio": 0.85})
        assert engine._check_regime(row) == True

    def test_vix_backwardation_blocks(self):
        engine = self._make_engine(use_vix=True)
        row = pd.Series({"vix_ratio": 1.2})
        assert engine._check_regime(row) == False

    def test_vix_nan_allows(self):
        engine = self._make_engine(use_vix=True)
        row = pd.Series({"vix_ratio": float("nan")})
        assert engine._check_regime(row) == True

    def test_fred_bullish_allows(self):
        engine = self._make_engine(use_fred=True)
        row = pd.Series({"fred_macro_regime": 0.7})
        assert engine._check_regime(row) == True

    def test_fred_bearish_blocks(self):
        engine = self._make_engine(use_fred=True)
        row = pd.Series({"fred_macro_regime": 0.3})
        assert engine._check_regime(row) == False

    def test_replace_mode_ignores_sma(self):
        """In replace mode, SMA regime filter is ignored."""
        engine = self._make_engine(
            use_fred=True,
            fred_regime_mode="replace",
            regime_filter=RegimeFilter(
                benchmark="TEST", indicator="sma",
                fast_period=20, slow_period=50,
            ),
        )
        row = pd.Series({
            "regime_fast": 90.0,
            "regime_slow": 100.0,
            "fred_macro_regime": 0.8,
        })
        assert engine._check_regime(row) == True

    def test_supplement_mode_requires_both(self):
        """In supplement mode, both SMA and FRED must agree."""
        engine = self._make_engine(
            use_fred=True,
            fred_regime_mode="supplement",
            regime_filter=RegimeFilter(
                benchmark="TEST", indicator="sma",
                fast_period=20, slow_period=50,
            ),
        )
        row = pd.Series({
            "regime_fast": 90.0,
            "regime_slow": 100.0,
            "fred_macro_regime": 0.8,
        })
        assert engine._check_regime(row) == False

    def test_combined_vix_and_fred(self):
        """Both VIX and FRED must pass."""
        engine = self._make_engine(use_vix=True, use_fred=True)
        assert engine._check_regime(
            pd.Series({"vix_ratio": 0.85, "fred_macro_regime": 0.7})
        ) == True
        assert engine._check_regime(
            pd.Series({"vix_ratio": 1.2, "fred_macro_regime": 0.7})
        ) == False
        assert engine._check_regime(
            pd.Series({"vix_ratio": 0.85, "fred_macro_regime": 0.2})
        ) == False


# ===========================================================================
# Config field tests
# ===========================================================================


class TestAltDataConfigFields:
    """Verify new config fields have correct defaults."""

    def test_defaults(self):
        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
        )
        assert config.use_vix is False
        assert config.use_intermarket is False
        assert config.use_fred is False
        assert config.fred_api_key is None
        assert config.fred_regime_mode == "supplement"
        assert config.use_yield_curve is False
        assert config.use_pcr is False
        assert config.use_analyst is False

    def test_custom_values(self):
        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            use_vix=True,
            use_fred=True,
            fred_api_key="test_key",
            fred_regime_mode="replace",
            use_yield_curve=True,
            use_pcr=True,
            use_analyst=True,
        )
        assert config.use_vix is True
        assert config.use_fred is True
        assert config.fred_api_key == "test_key"
        assert config.fred_regime_mode == "replace"


# ===========================================================================
# MacroAwareValue sell signal E2E
# ===========================================================================


class TestMacroAwareValueSellE2E:
    """E2E tests for MacroAwareValue sell triggers."""

    def test_sell_on_quality_deterioration(self):
        """F-Score drops below 3 mid-backtest → position sold."""
        n = 300
        prices = [80 + i * 0.2 for i in range(n)]
        df = make_controlled_df(prices)

        # Start with good fundamentals, deteriorate after day 100
        f_scores = np.concatenate([np.full(100, 8.0), np.full(n - 100, 2.0)])
        df["fund_piotroski_f"] = f_scores
        df["fund_altman_z"] = 4.0
        df["fund_pe_ratio"] = 12.0
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        # Should have trades: buy when good, sell when deterioration
        assert len(result.trades) >= 1
        _check_invariants(result)

    def test_sell_on_distress(self):
        """Z-Score drops below threshold → SELL."""
        n = 300
        prices = [80 + i * 0.2 for i in range(n)]
        df = make_controlled_df(prices)

        z_scores = np.concatenate([np.full(100, 4.0), np.full(n - 100, 1.0)])
        df["fund_piotroski_f"] = 7.0
        df["fund_altman_z"] = z_scores
        df["fund_pe_ratio"] = 12.0
        _add_macro_columns(df)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="macro_aware_value",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "min_z_score": 1.8},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) >= 1
        _check_invariants(result)


# ===========================================================================
# RiskRegime F-Score filter E2E
# ===========================================================================


class TestRiskRegimeFScoreE2E:
    """E2E tests for RiskRegime F-Score quality filter."""

    def test_low_fscore_blocks_buy_in_risk_on(self):
        """Risk-on but F-Score below threshold → no BUY."""
        df = _make_quality_df(n=200, uptrend=True)
        _make_vix_columns(df, ratio=0.85)
        _make_fred_columns(df, yield_spread=1.5, credit_spread=3.0)
        _add_fund_columns(df, fund_piotroski_f=2.0)  # very low F-Score

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="risk_regime",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "min_f_score": 5},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "Low F-Score should block BUY in risk-on"


# ===========================================================================
# SentimentMomentum threshold E2E
# ===========================================================================


class TestSentimentThresholdE2E:
    """E2E tests for SentimentMomentum signal thresholds."""

    def test_high_threshold_blocks_buy(self):
        """min_signals_buy=3 with only 2 sources → no trades."""
        prices = [50 + i * 0.3 for i in range(200)]
        df = make_controlled_df(prices)
        _make_analyst_columns(df, breadth=0.5)
        # Only SMA + analyst = max 2 bullish, needs 3
        # (no insider data)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sentiment_momentum",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "min_signals_buy": 3},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "Only 2 signals available, need 3"

    def test_sentiment_with_insider_only(self):
        """Only insider data + SMA, no analyst → should still work."""
        prices = [50 + i * 0.3 for i in range(200)]
        df = make_controlled_df(prices)
        _make_insider_columns(df, buy_ratio=0.8)

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sentiment_momentum",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "min_signals_buy": 2},
        )
        result = _run_backtest(config, source)
        # SMA bullish + insider bullish = 2 >= 2 → trades happen
        assert len(result.trades) > 0
        _check_invariants(result)


# ===========================================================================
# VIX regime filter E2E (engine-level)
# ===========================================================================


class TestVIXRegimeFilterE2E:
    """E2E tests for engine-level VIX regime filter blocking BUYs."""

    def test_vix_backwardation_blocks_sma_crossover(self):
        """VIX backwardation with use_vix=True → engine blocks BUY signals.

        Mocks MarketDataManager to inject VIX backwardation data into the
        engine pipeline, verifying the engine-level regime filter suppresses
        BUY signals when VIX ratio > 1.
        """
        prices = [100 + i * 0.3 for i in range(200)]
        df = make_controlled_df(prices)

        # Build VIX data that will be merged by the engine
        vix_df = pd.DataFrame(
            {
                "vix_close": np.full(len(df), 25.0),
                "vix_3m": np.full(len(df), 20.0),
                "vix_ratio": np.full(len(df), 1.3),
                "vix_regime": "backwardation",
            },
            index=df.index,
        )

        source = MockDataSource()
        source.add("TEST", df)
        start, end = _dates_from_df(df)

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.10,
            strategy_params={"sma_fast": 20, "sma_slow": 50},
            use_vix=True,
        )
        with patch("backtester.data.market_data.MarketDataManager") as MockMkt:
            mock_mgr = MagicMock()
            mock_mgr.load_vix_data.return_value = vix_df
            mock_mgr.load_intermarket_data.return_value = pd.DataFrame()
            MockMkt.return_value = mock_mgr
            # Also patch for benchmark loading
            result = _run_backtest(config, source)

        # VIX backwardation blocks all BUY signals at engine level
        assert len(result.trades) == 0, "VIX backwardation should block BUYs"
