"""Tests for analytics metrics."""

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import (
    cagr, sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_duration,
    total_return, win_rate, profit_factor, compute_all_metrics,
    calmar_ratio, beta, alpha, information_ratio, tracking_error, capture_ratio,
    trade_expectancy, avg_win_loss, holding_period_stats, max_consecutive,
    exposure_time,
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


# ---------------------------------------------------------------------------
# Benchmark-relative metrics
# ---------------------------------------------------------------------------

def _make_trades():
    """Helper: 3 trades — 2 winners, 1 loser."""
    return [
        Trade("A", date(2020, 1, 2), date(2020, 2, 3), 100, 120, 10, 200, 0.20, 32, 0),
        Trade("B", date(2020, 3, 2), date(2020, 4, 1), 100, 90, 10, -100, -0.10, 30, 0),
        Trade("C", date(2020, 5, 1), date(2020, 6, 1), 100, 115, 10, 150, 0.15, 31, 0),
    ]


class TestCalmarRatio:
    def test_positive(self):
        equity = make_equity([100, 110, 105, 115, 120])
        result = calmar_ratio(equity)
        assert result > 0

    def test_no_drawdown(self):
        equity = make_equity([100, 110, 120])
        result = calmar_ratio(equity)
        assert result == float("inf")

    def test_flat(self):
        equity = make_equity([100, 100, 100])
        assert calmar_ratio(equity) == 0.0


class TestBenchmarkRelative:
    def setup_method(self):
        self.equity = make_equity(np.linspace(100, 130, 252))
        self.benchmark = make_equity(np.linspace(100, 120, 252))

    def test_beta_positive(self):
        result = beta(self.equity, self.benchmark)
        assert result > 0

    def test_beta_identical(self):
        # Beta of a series vs itself should be ~1.0
        result = beta(self.equity, self.equity)
        assert abs(result - 1.0) < 0.01

    def test_alpha_outperformance(self):
        # Use noisy returns where strategy clearly outperforms to get positive alpha
        rng = np.random.default_rng(99)
        bm_vals = [100.0]
        strat_vals = [100.0]
        for _ in range(251):
            r = rng.normal(0.0003, 0.01)
            bm_vals.append(bm_vals[-1] * (1 + r))
            strat_vals.append(strat_vals[-1] * (1 + r + 0.001))  # consistent excess
        bm = make_equity(bm_vals)
        eq = make_equity(strat_vals)
        result = alpha(eq, bm)
        assert result > 0

    def test_information_ratio(self):
        result = information_ratio(self.equity, self.benchmark)
        assert np.isfinite(result)

    def test_tracking_error(self):
        result = tracking_error(self.equity, self.benchmark)
        assert result >= 0

    def test_tracking_error_identical(self):
        result = tracking_error(self.equity, self.equity)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_capture_ratio_up(self):
        result = capture_ratio(self.equity, self.benchmark, "up")
        assert result > 0

    def test_capture_ratio_down(self):
        result = capture_ratio(self.equity, self.benchmark, "down")
        # Could be any finite number
        assert np.isfinite(result)

    def test_compute_all_includes_benchmark(self):
        trades = []
        m = compute_all_metrics(self.equity, trades, benchmark_series=self.benchmark)
        assert "alpha" in m
        assert "beta" in m
        assert "information_ratio" in m
        assert "tracking_error" in m
        assert "up_capture" in m
        assert "down_capture" in m


class TestTradeLevel:
    def test_trade_expectancy(self):
        trades = _make_trades()
        result = trade_expectancy(trades)
        expected = (200 - 100 + 150) / 3.0
        assert result == pytest.approx(expected)

    def test_trade_expectancy_empty(self):
        assert trade_expectancy([]) == 0.0

    def test_avg_win_loss(self):
        trades = _make_trades()
        result = avg_win_loss(trades)
        assert result["avg_win"] == pytest.approx((200 + 150) / 2.0)
        assert result["avg_loss"] == pytest.approx(-100.0)
        assert result["payoff_ratio"] == pytest.approx(175.0 / 100.0)

    def test_avg_win_loss_no_losers(self):
        trades = [_make_trades()[0]]  # single winner
        result = avg_win_loss(trades)
        assert result["avg_loss"] == 0.0
        assert result["payoff_ratio"] == float("inf")

    def test_holding_period_stats(self):
        trades = _make_trades()
        result = holding_period_stats(trades)
        assert result["avg_days"] == pytest.approx((32 + 30 + 31) / 3.0)
        assert result["avg_days_winners"] == pytest.approx((32 + 31) / 2.0)
        assert result["avg_days_losers"] == pytest.approx(30.0)

    def test_holding_period_empty(self):
        result = holding_period_stats([])
        assert result["avg_days"] == 0

    def test_max_consecutive(self):
        trades = _make_trades()  # win, loss, win
        result = max_consecutive(trades)
        assert result["max_consecutive_wins"] == 1
        assert result["max_consecutive_losses"] == 1

    def test_max_consecutive_streak(self):
        # 3 wins in a row
        t = Trade("A", date(2020, 1, 2), date(2020, 2, 1), 100, 110, 10, 100, 0.10, 30, 0)
        result = max_consecutive([t, t, t])
        assert result["max_consecutive_wins"] == 3
        assert result["max_consecutive_losses"] == 0

    def test_max_consecutive_empty(self):
        result = max_consecutive([])
        assert result["max_consecutive_wins"] == 0

    def test_exposure_time(self):
        equity = make_equity(np.linspace(100, 120, 50))
        trades = _make_trades()
        result = exposure_time(equity, trades)
        assert 0.0 <= result <= 1.0

    def test_exposure_time_no_trades(self):
        equity = make_equity([100, 110, 120])
        assert exposure_time(equity, []) == 0.0
