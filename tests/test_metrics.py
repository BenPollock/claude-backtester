"""Tests for analytics metrics."""

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import (
    cagr, sharpe_ratio, max_drawdown, total_return, win_rate, profit_factor,
    compute_all_metrics,
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

    def test_compute_all_metrics(self):
        equity = make_equity(np.linspace(100, 120, 50))
        trades = []
        m = compute_all_metrics(equity, trades)
        assert "cagr" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m
        assert m["total_trades"] == 0
