"""Tests for analytics metrics."""

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import (
    cagr, sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_duration,
    total_return, win_rate, profit_factor, compute_all_metrics,
)
from backtester.portfolio.order import Trade
from datetime import date


def make_equity(values, start="2020-01-02"):
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=pd.DatetimeIndex(dates.date, name="Date"), name="Equity")


class TestMetrics:
    def test_total_return(self):
        equity = make_equity([100, 110, 120, 130])
        assert abs(total_return(equity) - 0.30) < 0.001

    def test_cagr_one_year(self):
        # 100 -> 110 over ~1 year (252 bdays ≈ 365 days)
        values = np.linspace(100, 110, 252)
        equity = make_equity(values)
        result = cagr(equity)
        assert abs(result - 0.10) < 0.02  # approximately 10%

    def test_cagr_flat(self):
        equity = make_equity([100, 100, 100])
        assert cagr(equity) == 0.0

    def test_sharpe_positive(self):
        # Steadily increasing equity should have positive Sharpe
        values = np.linspace(100, 120, 252)
        equity = make_equity(values)
        assert sharpe_ratio(equity) > 0

    def test_sharpe_flat(self):
        equity = make_equity([100, 100, 100, 100])
        assert sharpe_ratio(equity) == 0.0

    def test_max_drawdown(self):
        equity = make_equity([100, 110, 90, 95, 105])
        dd = max_drawdown(equity)
        # Peak was 110, trough was 90 -> dd = (90-110)/110 = -0.1818
        assert abs(dd - (-20 / 110)) < 0.001

    def test_max_drawdown_no_drawdown(self):
        equity = make_equity([100, 110, 120])
        assert max_drawdown(equity) == 0.0

    def test_win_rate(self):
        trades = [
            Trade("A", date(2020,1,1), date(2020,2,1), 100, 110, 10, 100, 0.10, 30, 0),
            Trade("B", date(2020,1,1), date(2020,2,1), 100, 90, 10, -100, -0.10, 30, 0),
            Trade("C", date(2020,1,1), date(2020,2,1), 100, 120, 10, 200, 0.20, 30, 0),
        ]
        assert abs(win_rate(trades) - 2/3) < 0.001

    def test_profit_factor(self):
        trades = [
            Trade("A", date(2020,1,1), date(2020,2,1), 100, 110, 10, 100, 0.10, 30, 0),
            Trade("B", date(2020,1,1), date(2020,2,1), 100, 90, 10, -50, -0.05, 30, 0),
        ]
        assert abs(profit_factor(trades) - 2.0) < 0.001

    def test_sortino_positive_returns(self):
        # Equity with upward trend but some down days to create downside deviation
        rng = np.random.default_rng(42)
        values = [100.0]
        for _ in range(251):
            values.append(values[-1] * (1 + 0.001 + rng.normal(0, 0.01)))
        equity = make_equity(values)
        result = sortino_ratio(equity)
        assert result > 0
        assert np.isfinite(result)

    def test_sortino_no_downside_returns_inf(self):
        # Monotonically increasing — no negative returns
        equity = make_equity([100, 101, 102, 103, 104, 105])
        result = sortino_ratio(equity)
        assert result == float("inf")

    def test_sortino_few_returns(self):
        # Fewer than 2 data points → 0.0
        equity = make_equity([100])
        assert sortino_ratio(equity) == 0.0

    def test_max_drawdown_duration_basic(self):
        # Peak at 110, drops to 90, recovers at index 4 (105 still below 110),
        # full recovery implied by 120
        equity = make_equity([100, 110, 90, 95, 105, 120])
        duration = max_drawdown_duration(equity)
        assert duration > 0

    def test_max_drawdown_duration_still_underwater(self):
        # Never recovers from drawdown — should include final underwater stretch
        equity = make_equity([100, 110, 90, 85, 80])
        duration = max_drawdown_duration(equity)
        # Drawdown starts after day 2 (peak 110), ends at last day
        assert duration > 0

    def test_max_drawdown_duration_no_drawdown(self):
        # Monotonically rising → duration should be 0
        equity = make_equity([100, 101, 102, 103])
        assert max_drawdown_duration(equity) == 0

    def test_profit_factor_no_losses_returns_inf(self):
        trades = [
            Trade("A", date(2020,1,1), date(2020,2,1), 100, 110, 10, 100, 0.10, 30, 0),
            Trade("B", date(2020,1,1), date(2020,2,1), 100, 120, 10, 200, 0.20, 30, 0),
        ]
        assert profit_factor(trades) == float("inf")

    def test_profit_factor_no_trades_returns_zero(self):
        assert profit_factor([]) == 0.0

    def test_win_rate_no_trades_returns_zero(self):
        assert win_rate([]) == 0.0

    def test_compute_all_metrics(self):
        equity = make_equity(np.linspace(100, 120, 50))
        trades = []
        m = compute_all_metrics(equity, trades)
        assert "cagr" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m
        assert m["total_trades"] == 0
