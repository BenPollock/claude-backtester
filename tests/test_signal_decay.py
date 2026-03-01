"""Tests for signal decay analysis (analytics/signal_decay.py).

All tests use synthetic price data with known values so that expected
returns can be calculated by hand.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.signal_decay import (
    average_signal_decay,
    compute_signal_returns,
    optimal_holding_period,
    signal_decay_summary,
)
from backtester.portfolio.order import Trade


# ── Helpers ──────────────────────────────────────────────────────────


def _make_price_df(closes: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of Close prices.

    Uses business-day frequency so the index represents trading days.
    Open/High/Low are set equal to Close for simplicity; Volume is constant.
    """
    dates = pd.bdate_range(start=start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=dates,
    )


def _make_trade(symbol: str, entry_date: date, exit_date: date,
                entry_price: float, exit_price: float,
                quantity: int = 100) -> Trade:
    """Create a Trade with sensible defaults for fields we don't test."""
    pnl = (exit_price - entry_price) * quantity
    pnl_pct = (exit_price - entry_price) / entry_price
    holding_days = (exit_date - entry_date).days
    return Trade(
        symbol=symbol,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_pct=pnl_pct,
        holding_days=holding_days,
        fees_total=0.0,
    )


# ── compute_signal_returns tests ─────────────────────────────────────


class TestComputeSignalReturns:
    """Tests for compute_signal_returns."""

    def test_basic_returns_known_prices(self):
        """Verify T+1 through T+5 returns against hand-calculated values."""
        # Prices: 100, 102, 105, 103, 108, 110, ...
        closes = [100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 112.0]
        df = _make_price_df(closes)
        dates = df.index

        trade = _make_trade(
            symbol="AAPL",
            entry_date=dates[0].date(),
            exit_date=dates[6].date(),  # held through all 7 days
            entry_price=100.0,
            exit_price=112.0,
        )

        result = compute_signal_returns([trade], {"AAPL": df}, max_horizon=5)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["ticker"] == "AAPL"
        assert row["signal_side"] == "BUY"

        # Expected cumulative returns from Close[0]=100
        assert pytest.approx(row["T+1"], abs=1e-9) == 0.02    # (102-100)/100
        assert pytest.approx(row["T+2"], abs=1e-9) == 0.05    # (105-100)/100
        assert pytest.approx(row["T+3"], abs=1e-9) == 0.03    # (103-100)/100
        assert pytest.approx(row["T+4"], abs=1e-9) == 0.08    # (108-100)/100
        assert pytest.approx(row["T+5"], abs=1e-9) == 0.10    # (110-100)/100

    def test_trade_exited_before_max_horizon(self):
        """Post-exit periods should be NaN."""
        closes = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
        df = _make_price_df(closes)
        dates = df.index

        # Trade exits on day 2 (index 2), so T+3 onward should be NaN
        trade = _make_trade(
            symbol="MSFT",
            entry_date=dates[0].date(),
            exit_date=dates[2].date(),
            entry_price=100.0,
            exit_price=110.0,
        )

        result = compute_signal_returns([trade], {"MSFT": df}, max_horizon=5)
        row = result.iloc[0]

        # T+1 and T+2 are within the trade's lifetime
        assert pytest.approx(row["T+1"], abs=1e-9) == 0.05  # (105-100)/100
        assert pytest.approx(row["T+2"], abs=1e-9) == 0.10  # (110-100)/100

        # T+3 through T+5 are after exit -- should be NaN
        assert np.isnan(row["T+3"])
        assert np.isnan(row["T+4"])
        assert np.isnan(row["T+5"])

    def test_empty_trades_returns_empty_dataframe(self):
        """Empty trades list should return an empty DataFrame with correct columns."""
        result = compute_signal_returns([], {}, max_horizon=3)

        assert result.empty
        expected_cols = ["trade_id", "ticker", "entry_date", "signal_side",
                         "T+1", "T+2", "T+3"]
        assert list(result.columns) == expected_cols

    def test_ticker_not_in_price_data(self):
        """Trade for a ticker not in price_data should produce all NaN returns."""
        trade = _make_trade(
            symbol="MISSING",
            entry_date=date(2024, 1, 2),
            exit_date=date(2024, 1, 10),
            entry_price=50.0,
            exit_price=55.0,
        )

        result = compute_signal_returns([trade], {}, max_horizon=3)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["ticker"] == "MISSING"
        assert np.isnan(row["T+1"])
        assert np.isnan(row["T+2"])
        assert np.isnan(row["T+3"])

    def test_single_trade(self):
        """Single trade should produce a single-row DataFrame."""
        closes = [50.0, 52.0, 48.0]
        df = _make_price_df(closes)
        dates = df.index

        trade = _make_trade(
            symbol="SPY",
            entry_date=dates[0].date(),
            exit_date=dates[2].date(),
            entry_price=50.0,
            exit_price=48.0,
        )

        result = compute_signal_returns([trade], {"SPY": df}, max_horizon=2)

        assert len(result) == 1
        row = result.iloc[0]
        assert pytest.approx(row["T+1"], abs=1e-9) == 0.04   # (52-50)/50
        assert pytest.approx(row["T+2"], abs=1e-9) == -0.04  # (48-50)/50

    def test_horizon_beyond_data(self):
        """When max_horizon extends past available data, later horizons are NaN."""
        closes = [100.0, 110.0]
        df = _make_price_df(closes)
        dates = df.index

        trade = _make_trade(
            symbol="XYZ",
            entry_date=dates[0].date(),
            exit_date=dates[1].date(),
            entry_price=100.0,
            exit_price=110.0,
        )

        result = compute_signal_returns([trade], {"XYZ": df}, max_horizon=5)
        row = result.iloc[0]

        assert pytest.approx(row["T+1"], abs=1e-9) == 0.10
        # T+2 through T+5 are beyond available data
        for h in range(2, 6):
            assert np.isnan(row[f"T+{h}"])

    def test_multiple_trades_same_ticker(self):
        """Multiple trades on the same ticker produce separate rows."""
        closes = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0]
        df = _make_price_df(closes)
        dates = df.index

        trades = [
            _make_trade("AAPL", dates[0].date(), dates[3].date(), 100.0, 106.0),
            _make_trade("AAPL", dates[3].date(), dates[6].date(), 106.0, 112.0),
        ]

        result = compute_signal_returns(trades, {"AAPL": df}, max_horizon=3)
        assert len(result) == 2

        # First trade: entry at 100
        assert pytest.approx(result.iloc[0]["T+1"], abs=1e-9) == 0.02  # (102-100)/100

        # Second trade: entry at 106
        assert pytest.approx(result.iloc[1]["T+1"], abs=1e-9) == (108 - 106) / 106


# ── average_signal_decay tests ───────────────────────────────────────


class TestAverageSignalDecay:
    """Tests for average_signal_decay."""

    def test_mean_median_std_across_trades(self):
        """Verify mean, median, and std with two trades having known returns."""
        # Build a DataFrame manually rather than going through compute_signal_returns
        data = {
            "trade_id": [0, 1],
            "ticker": ["A", "A"],
            "entry_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "signal_side": ["BUY", "BUY"],
            "T+1": [0.10, 0.20],   # mean=0.15, median=0.15, std known
            "T+2": [0.20, 0.10],   # mean=0.15, median=0.15
            "T+3": [0.30, 0.00],   # mean=0.15, median=0.15
        }
        df = pd.DataFrame(data)

        mean_s, median_s, std_s = average_signal_decay(df)

        assert pytest.approx(mean_s["T+1"], abs=1e-9) == 0.15
        assert pytest.approx(mean_s["T+2"], abs=1e-9) == 0.15
        assert pytest.approx(mean_s["T+3"], abs=1e-9) == 0.15

        assert pytest.approx(median_s["T+1"], abs=1e-9) == 0.15
        assert pytest.approx(median_s["T+2"], abs=1e-9) == 0.15
        assert pytest.approx(median_s["T+3"], abs=1e-9) == 0.15

        # std of [0.10, 0.20] with ddof=1 = 0.070710...
        expected_std = pd.Series([0.10, 0.20]).std()
        assert pytest.approx(std_s["T+1"], abs=1e-6) == expected_std

    def test_empty_dataframe(self):
        """Empty input returns empty Series for all three outputs."""
        cols = ["trade_id", "ticker", "entry_date", "signal_side", "T+1", "T+2"]
        df = pd.DataFrame(columns=cols)
        mean_s, median_s, std_s = average_signal_decay(df)

        assert mean_s.empty
        assert median_s.empty
        assert std_s.empty

    def test_nan_values_excluded_from_mean(self):
        """NaN values should be excluded (pandas default skipna=True)."""
        data = {
            "trade_id": [0, 1],
            "ticker": ["A", "A"],
            "entry_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "signal_side": ["BUY", "BUY"],
            "T+1": [0.10, 0.20],
            "T+2": [0.20, np.nan],  # only one value for T+2
        }
        df = pd.DataFrame(data)
        mean_s, median_s, _ = average_signal_decay(df)

        assert pytest.approx(mean_s["T+1"], abs=1e-9) == 0.15
        assert pytest.approx(mean_s["T+2"], abs=1e-9) == 0.20  # single value


# ── optimal_holding_period tests ─────────────────────────────────────


class TestOptimalHoldingPeriod:
    """Tests for optimal_holding_period."""

    def test_peak_in_middle(self):
        """Returns peak at horizon where average return is highest."""
        # Construct returns that peak at T+3
        data = {
            "trade_id": [0],
            "ticker": ["A"],
            "entry_date": [date(2024, 1, 2)],
            "signal_side": ["BUY"],
            "T+1": [0.02],
            "T+2": [0.05],
            "T+3": [0.08],  # peak
            "T+4": [0.06],
            "T+5": [0.03],
        }
        df = pd.DataFrame(data)
        result = optimal_holding_period(df)

        assert result["optimal_days"] == 3
        assert pytest.approx(result["peak_return"], abs=1e-9) == 0.08
        assert pytest.approx(result["return_at_max_horizon"], abs=1e-9) == 0.03

    def test_monotonically_increasing(self):
        """When returns only increase, optimal = max_horizon."""
        data = {
            "trade_id": [0],
            "ticker": ["A"],
            "entry_date": [date(2024, 1, 2)],
            "signal_side": ["BUY"],
            "T+1": [0.01],
            "T+2": [0.02],
            "T+3": [0.03],
            "T+4": [0.04],
        }
        df = pd.DataFrame(data)
        result = optimal_holding_period(df)

        assert result["optimal_days"] == 4
        assert pytest.approx(result["peak_return"], abs=1e-9) == 0.04
        assert pytest.approx(result["return_at_max_horizon"], abs=1e-9) == 0.04

    def test_monotonically_decreasing(self):
        """When returns only decrease, optimal = 1."""
        data = {
            "trade_id": [0],
            "ticker": ["A"],
            "entry_date": [date(2024, 1, 2)],
            "signal_side": ["BUY"],
            "T+1": [0.04],
            "T+2": [0.03],
            "T+3": [0.02],
            "T+4": [0.01],
        }
        df = pd.DataFrame(data)
        result = optimal_holding_period(df)

        assert result["optimal_days"] == 1
        assert pytest.approx(result["peak_return"], abs=1e-9) == 0.04
        assert pytest.approx(result["return_at_max_horizon"], abs=1e-9) == 0.01

    def test_empty_input(self):
        """Empty DataFrame returns zero defaults."""
        cols = ["trade_id", "ticker", "entry_date", "signal_side", "T+1"]
        df = pd.DataFrame(columns=cols)
        result = optimal_holding_period(df)

        assert result["optimal_days"] == 0
        assert result["peak_return"] == 0.0
        assert result["return_at_max_horizon"] == 0.0


# ── signal_decay_summary tests ───────────────────────────────────────


class TestSignalDecaySummary:
    """Tests for signal_decay_summary end-to-end."""

    def test_end_to_end_integration(self):
        """Full pipeline: trades + price data -> complete summary dict."""
        # Two tickers, rising prices
        closes_a = [100.0, 103.0, 106.0, 104.0, 109.0, 112.0]
        closes_b = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0]

        df_a = _make_price_df(closes_a)
        df_b = _make_price_df(closes_b)
        dates_a = df_a.index
        dates_b = df_b.index

        trades = [
            _make_trade("A", dates_a[0].date(), dates_a[4].date(), 100.0, 109.0),
            _make_trade("B", dates_b[0].date(), dates_b[3].date(), 50.0, 53.0),
        ]

        price_data = {"A": df_a, "B": df_b}
        result = signal_decay_summary(trades, price_data, max_horizon=5)

        # Check structure
        assert "per_trade_returns" in result
        assert "avg_decay" in result
        assert "median_decay" in result
        assert "std_decay" in result
        assert "optimal_holding" in result
        assert result["total_signals"] == 2

        # Per-trade returns DataFrame has 2 rows
        ptr = result["per_trade_returns"]
        assert len(ptr) == 2

        # avg_decay is a Series indexed by T+1..T+5
        avg = result["avg_decay"]
        assert len(avg) == 5
        assert "T+1" in avg.index

        # optimal_holding has the expected keys
        oh = result["optimal_holding"]
        assert "optimal_days" in oh
        assert "peak_return" in oh
        assert "return_at_max_horizon" in oh

    def test_no_trades(self):
        """No trades yields empty returns and zero counts."""
        result = signal_decay_summary([], {}, max_horizon=3)

        assert result["total_signals"] == 0
        assert result["per_trade_returns"].empty
        assert result["avg_decay"].empty
        assert result["optimal_holding"]["optimal_days"] == 0

    def test_single_trade_summary(self):
        """Single trade produces valid summary without errors."""
        closes = [200.0, 210.0, 205.0, 215.0]
        df = _make_price_df(closes)
        dates = df.index

        trade = _make_trade("QQQ", dates[0].date(), dates[3].date(), 200.0, 215.0)
        result = signal_decay_summary([trade], {"QQQ": df}, max_horizon=3)

        assert result["total_signals"] == 1
        avg = result["avg_decay"]
        # With a single trade, mean equals the trade's return at each horizon
        assert pytest.approx(avg["T+1"], abs=1e-9) == 0.05   # (210-200)/200
        assert pytest.approx(avg["T+2"], abs=1e-9) == 0.025  # (205-200)/200
        assert pytest.approx(avg["T+3"], abs=1e-9) == 0.075  # (215-200)/200
