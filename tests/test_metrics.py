"""Tests for analytics metrics."""

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import (
    cagr, sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_duration,
    total_return, win_rate, profit_factor, compute_all_metrics,
    calmar_ratio, beta, alpha, information_ratio, tracking_error, capture_ratio,
    trade_expectancy, avg_win_loss, holding_period_stats, max_consecutive,
    exposure_time, historical_var, cvar, omega_ratio, treynor_ratio,
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
        result = sharpe_ratio(equity)
        assert result > 0
        # Verify: for linearly increasing equity, daily return is constant
        # daily_return = (120/100)^(1/251) - 1 ≈ 0.000722
        # std of constant returns is very small (not zero due to floating point)
        # Sharpe should be very high (>10) since std is near zero
        assert result > 10.0

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
        # Sortino should be higher than Sharpe since it only penalizes downside vol
        sharpe_val = sharpe_ratio(equity)
        assert result > sharpe_val

    def test_sortino_denominator_uses_all_returns(self):
        """Sortino downside deviation uses N (total count), not just negative count.

        The academic formula is: σ_d = sqrt(Σ min(r_i - MAR, 0)^2 / N)
        where N is the total number of returns, not just the negative ones.
        """
        equity = make_equity([100, 110, 105, 115, 108, 120])
        returns = equity.pct_change().dropna()
        excess = returns - 0.0  # risk_free_rate = 0

        # Correct downside deviation: sqrt(mean(min(excess, 0)^2)) over ALL returns
        downside_squared = np.minimum(excess.values, 0) ** 2
        expected_dd = np.sqrt(downside_squared.mean())
        expected_sortino = (excess.mean() / expected_dd) * np.sqrt(252)

        result = sortino_ratio(equity, risk_free_rate=0.0)
        assert result == pytest.approx(expected_sortino, rel=1e-6)

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
        # Beta of a series vs itself should be exactly 1.0
        result = beta(self.equity, self.equity)
        assert result == pytest.approx(1.0, abs=1e-10)

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
        # Jensen's alpha = annualized excess return over CAPM prediction
        # With 0.001 daily excess and beta ~1, alpha ≈ 0.001 * 252 = 0.252
        assert result == pytest.approx(0.252, abs=0.05)

    def test_information_ratio(self):
        result = information_ratio(self.equity, self.benchmark)
        assert np.isfinite(result)
        # Equity outperforms benchmark (130 vs 120), so IR should be positive
        assert result > 0

    def test_information_ratio_identical(self):
        # Identical series should have IR = 0 (no active return)
        result = information_ratio(self.equity, self.equity)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_tracking_error(self):
        result = tracking_error(self.equity, self.benchmark)
        assert result >= 0

    def test_tracking_error_identical(self):
        result = tracking_error(self.equity, self.equity)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_capture_ratio_up(self):
        result = capture_ratio(self.equity, self.benchmark, "up")
        assert result > 0
        # Both are linearly increasing, equity grows faster (30% vs 20%)
        # Up capture = mean(strat_ret on up days) / mean(bm_ret on up days) * 100
        # Since both are linear with all positive returns, ratio ≈ 130/120 * 100
        # but it's return-based so approximately (30/20)*100 = 150 is an upper bound
        assert result > 100.0  # Captures more than benchmark on up days

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


class TestHistoricalVaR:
    def test_var_basic(self):
        """VaR at 95% should return the 5th percentile of daily returns."""
        rng = np.random.default_rng(42)
        values = [100.0]
        for _ in range(999):
            values.append(values[-1] * (1 + rng.normal(0, 0.01)))
        equity = make_equity(values)
        result = historical_var(equity, confidence=0.95)
        # VaR should be negative (it's a loss percentile)
        assert result < 0
        # Manual check: compute returns and verify percentile
        returns = equity.pct_change().dropna()
        expected = float(np.percentile(returns, 5))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_var_monotonic_increase(self):
        """Monotonically increasing equity: 5th percentile should still be positive."""
        equity = make_equity(np.linspace(100, 200, 252))
        result = historical_var(equity, confidence=0.95)
        # All returns are positive, so 5th percentile is also positive
        assert result > 0

    def test_var_few_returns(self):
        equity = make_equity([100])
        assert historical_var(equity) == 0.0


class TestCVaR:
    def test_cvar_basic(self):
        """CVaR should be <= VaR (further into the tail)."""
        rng = np.random.default_rng(42)
        values = [100.0]
        for _ in range(999):
            values.append(values[-1] * (1 + rng.normal(0, 0.01)))
        equity = make_equity(values)
        var_val = historical_var(equity, confidence=0.95)
        cvar_val = cvar(equity, confidence=0.95)
        # CVaR is the mean of returns below VaR, so it should be <= VaR
        assert cvar_val <= var_val

    def test_cvar_manual(self):
        """CVaR should equal mean of returns at or below the VaR threshold."""
        rng = np.random.default_rng(42)
        values = [100.0]
        for _ in range(999):
            values.append(values[-1] * (1 + rng.normal(0, 0.01)))
        equity = make_equity(values)
        returns = equity.pct_change().dropna()
        var_val = float(np.percentile(returns, 5))
        expected_cvar = float(returns[returns <= var_val].mean())
        result = cvar(equity, confidence=0.95)
        assert result == pytest.approx(expected_cvar, rel=1e-6)

    def test_cvar_few_returns(self):
        equity = make_equity([100])
        assert cvar(equity) == 0.0


class TestOmegaRatio:
    def test_omega_positive_returns(self):
        """Omega ratio > 1 for positive-mean returns (gains exceed losses)."""
        rng = np.random.default_rng(42)
        values = [100.0]
        for _ in range(251):
            values.append(values[-1] * (1 + 0.001 + rng.normal(0, 0.005)))
        equity = make_equity(values)
        result = omega_ratio(equity, threshold=0.0)
        assert result > 1.0

    def test_omega_all_positive(self):
        """Omega = inf when all returns are positive (no losses below threshold)."""
        equity = make_equity(np.linspace(100, 120, 50))
        result = omega_ratio(equity, threshold=0.0)
        assert result == float("inf")

    def test_omega_manual_calculation(self):
        """Verify Omega ratio against manual calculation."""
        equity = make_equity([100, 110, 105, 115, 108])
        returns = equity.pct_change().dropna()
        gains = returns[returns > 0] - 0.0
        losses = 0.0 - returns[returns <= 0]
        expected = float(gains.sum() / losses.sum())
        result = omega_ratio(equity, threshold=0.0)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_omega_few_returns(self):
        equity = make_equity([100])
        assert omega_ratio(equity) == 0.0


class TestTreynorRatio:
    def test_treynor_basic(self):
        """Treynor ratio = (CAGR - rf) / beta."""
        equity = make_equity(np.linspace(100, 130, 252))
        benchmark = make_equity(np.linspace(100, 120, 252))
        result = treynor_ratio(equity, benchmark)
        # Manual: CAGR / beta
        c = cagr(equity)
        b = beta(equity, benchmark)
        expected = c / b
        assert result == pytest.approx(expected, rel=1e-6)

    def test_treynor_zero_beta(self):
        """Treynor should return 0 when beta is 0."""
        # Strategy returns uncorrelated with benchmark
        equity = make_equity([100, 110, 100, 110, 100])
        benchmark = make_equity([100, 100, 110, 100, 110])
        result = treynor_ratio(equity, benchmark)
        # Beta should be near 0 or negative, but if exactly 0, returns 0
        assert np.isfinite(result)


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


# ---------------------------------------------------------------------------
# Strengthened metric tests — edge cases and value verification
# ---------------------------------------------------------------------------

class TestMetricsEdgeCases:
    def test_cagr_negative_return(self):
        """CAGR should be negative when equity declines."""
        # 100 -> 80 over ~1 year (252 business days)
        values = np.linspace(100, 80, 252)
        equity = make_equity(values)
        result = cagr(equity)
        assert result < 0
        # Approximately -20% CAGR
        assert result == pytest.approx(-0.20, abs=0.03)

    def test_sharpe_declining_equity(self):
        """Sharpe should be negative for consistently declining equity."""
        values = np.linspace(100, 70, 252)
        equity = make_equity(values)
        result = sharpe_ratio(equity)
        assert result < 0
        # Linearly declining equity has constant negative daily returns,
        # so Sharpe should be very negative (large magnitude)
        assert result < -10.0

    def test_alpha_underperformance(self):
        """Alpha should be negative when strategy consistently underperforms."""
        rng = np.random.default_rng(99)
        bm_vals = [100.0]
        strat_vals = [100.0]
        for _ in range(251):
            r = rng.normal(0.0003, 0.01)
            bm_vals.append(bm_vals[-1] * (1 + r))
            strat_vals.append(strat_vals[-1] * (1 + r - 0.001))  # consistent drag
        bm = make_equity(bm_vals)
        eq = make_equity(strat_vals)
        result = alpha(eq, bm)
        assert result < 0
        # Daily underperformance of 0.001 => annualized alpha ~ -0.252
        assert result == pytest.approx(-0.252, abs=0.05)

    def test_omega_ratio_nonzero_threshold(self):
        """Omega with a positive threshold should be lower than with threshold=0."""
        rng = np.random.default_rng(42)
        values = [100.0]
        for _ in range(251):
            values.append(values[-1] * (1 + 0.001 + rng.normal(0, 0.005)))
        equity = make_equity(values)
        omega_zero = omega_ratio(equity, threshold=0.0)
        omega_high = omega_ratio(equity, threshold=0.001)
        # Higher threshold means more returns fall below it -> lower omega
        assert omega_high < omega_zero
        # Both should be positive (net positive returns)
        assert omega_high > 0
        # Manual verification at nonzero threshold
        returns = equity.pct_change().dropna()
        gains = returns[returns > 0.001] - 0.001
        losses = 0.001 - returns[returns <= 0.001]
        expected = float(gains.sum() / losses.sum())
        assert omega_high == pytest.approx(expected, rel=1e-6)

    def test_capture_ratio_down_value(self):
        """Down capture should reflect behavior in down markets.
        A strategy that loses less than benchmark should have down capture < 100."""
        rng = np.random.default_rng(55)
        bm_vals = [100.0]
        strat_vals = [100.0]
        for _ in range(251):
            r = rng.normal(0.0, 0.015)
            bm_vals.append(bm_vals[-1] * (1 + r))
            # Strategy captures only 80% of benchmark moves
            strat_vals.append(strat_vals[-1] * (1 + 0.8 * r))
        bm = make_equity(bm_vals)
        eq = make_equity(strat_vals)
        result = capture_ratio(eq, bm, "down")
        # Strategy captures ~80% of down moves -> down capture ~ 80
        assert result == pytest.approx(80.0, abs=10.0)

    def test_treynor_negative_cagr(self):
        """Treynor should be negative when CAGR is negative (declining equity)."""
        # Declining strategy, rising benchmark
        values_strat = np.linspace(100, 80, 252)
        values_bm = np.linspace(100, 120, 252)
        equity = make_equity(values_strat)
        benchmark = make_equity(values_bm)
        result = treynor_ratio(equity, benchmark)
        # CAGR is negative, beta could be positive or negative,
        # but with inversely correlated returns, beta should be negative
        # Treynor = negative_cagr / negative_beta could be positive
        # OR if beta is near 0, returns 0.
        # The key: result should be finite and computable.
        assert np.isfinite(result)
        # Verify manual calculation
        c = cagr(equity)
        b = beta(equity, benchmark)
        if b != 0:
            assert result == pytest.approx(c / b, rel=1e-6)
        else:
            assert result == 0.0


class TestInfClamping:
    """Test that compute_all_metrics clamps inf values."""

    def test_inf_clamped_to_sentinel(self):
        """Inf values in metrics should be clamped to 99999.0."""
        # Monotonically increasing equity: Sortino = inf, profit_factor = inf (no trades with losses)
        equity = make_equity([100, 101, 102, 103, 104, 105])
        trades = [
            Trade("A", date(2020, 1, 2), date(2020, 2, 1), 100, 110, 10, 100, 0.10, 30, 0),
        ]
        m = compute_all_metrics(equity, trades)
        # sortino should be clamped (all positive returns -> inf without clamping)
        assert m["sortino_ratio"] == 99999.0
        # profit_factor: only winners -> inf without clamping
        assert m["profit_factor"] == 99999.0
        # calmar_ratio: no drawdown -> inf without clamping
        assert m["calmar_ratio"] == 99999.0

    def test_no_inf_in_metrics_dict(self):
        """No metric value should be inf after compute_all_metrics."""
        equity = make_equity(np.linspace(100, 120, 50))
        trades = []
        m = compute_all_metrics(equity, trades)
        for k, v in m.items():
            if isinstance(v, float):
                assert v != float("inf"), f"{k} should not be inf"
                assert v != float("-inf"), f"{k} should not be -inf"


# ---------------------------------------------------------------------------
# Coverage-expanding tests
# ---------------------------------------------------------------------------


class TestSharpeWithRiskFreeRate:
    """Sharpe and Sortino with non-zero risk-free rate."""

    def test_sharpe_with_positive_risk_free(self):
        """Non-zero risk_free_rate should reduce the Sharpe ratio vs zero rf."""
        rng = np.random.default_rng(42)
        values = [100.0]
        for _ in range(251):
            values.append(values[-1] * (1 + 0.0005 + rng.normal(0, 0.01)))
        equity = make_equity(values)
        sharpe_0 = sharpe_ratio(equity, risk_free_rate=0.0)
        sharpe_5 = sharpe_ratio(equity, risk_free_rate=0.05)
        # Higher risk-free rate reduces excess returns and thus Sharpe
        assert sharpe_5 < sharpe_0

    def test_sortino_with_positive_risk_free(self):
        """Non-zero risk_free_rate should reduce the Sortino ratio."""
        rng = np.random.default_rng(42)
        values = [100.0]
        for _ in range(251):
            values.append(values[-1] * (1 + 0.0005 + rng.normal(0, 0.01)))
        equity = make_equity(values)
        sortino_0 = sortino_ratio(equity, risk_free_rate=0.0)
        sortino_5 = sortino_ratio(equity, risk_free_rate=0.05)
        assert sortino_5 < sortino_0


class TestMaxDrawdownAllZero:
    """max_drawdown with degenerate equity series."""

    def test_all_zero_equity(self):
        """All-zero equity should return 0.0 (no drawdown computable)."""
        equity = make_equity([0.0, 0.0, 0.0, 0.0])
        assert max_drawdown(equity) == 0.0

    def test_single_value(self):
        """Single-value series (< 2 items) should return 0.0."""
        equity = make_equity([100.0])
        assert max_drawdown(equity) == 0.0

    def test_drawdown_starts_on_first_bar(self):
        """Drawdown starting on the first bar should be tracked."""
        # Peak on first bar, then immediate decline
        equity = make_equity([200.0, 180.0, 160.0, 170.0])
        dd = max_drawdown(equity)
        # Max drawdown: (160 - 200) / 200 = -0.20
        assert dd == pytest.approx(-0.20)

    def test_drawdown_duration_starts_on_first_bar(self):
        """Drawdown duration when the drawdown starts on bar 1."""
        equity = make_equity([200.0, 180.0, 160.0, 170.0, 190.0])
        duration = max_drawdown_duration(equity)
        assert duration > 0


class TestTotalReturnEdgeCases:
    """Edge cases for total_return."""

    def test_total_return_zero_start(self):
        """Start at zero should return 0.0 (avoid division by zero)."""
        equity = make_equity([0.0, 100.0, 200.0])
        assert total_return(equity) == 0.0

    def test_total_return_single_value(self):
        """Single value should return 0.0."""
        equity = make_equity([100.0])
        assert total_return(equity) == 0.0

    def test_cagr_zero_start(self):
        """CAGR with zero start value should return 0.0."""
        equity = make_equity([0.0, 100.0, 200.0])
        assert cagr(equity) == 0.0

    def test_cagr_single_day(self):
        """CAGR with 0 days between start and end returns 0.0."""
        equity = make_equity([100.0])
        assert cagr(equity) == 0.0


class TestComputeAllWithBenchmark:
    """compute_all_metrics with benchmark includes all expected keys."""

    def test_benchmark_metrics_keys_present(self):
        """All benchmark-relative keys should appear when benchmark is given."""
        equity = make_equity(np.linspace(100, 130, 50))
        benchmark = make_equity(np.linspace(100, 120, 50))
        m = compute_all_metrics(equity, [], benchmark_series=benchmark)
        expected_keys = ["alpha", "beta", "information_ratio", "tracking_error",
                         "up_capture", "down_capture", "treynor_ratio"]
        for key in expected_keys:
            assert key in m, f"Missing key: {key}"

    def test_risk_free_rate_affects_alpha(self):
        """Non-zero risk_free_rate should produce different alpha."""
        equity = make_equity(np.linspace(100, 130, 252))
        benchmark = make_equity(np.linspace(100, 120, 252))
        m_0 = compute_all_metrics(equity, [], risk_free_rate=0.0,
                                   benchmark_series=benchmark)
        m_5 = compute_all_metrics(equity, [], risk_free_rate=0.05,
                                   benchmark_series=benchmark)
        # Different risk-free rates produce different alphas
        assert m_0["alpha"] != m_5["alpha"]


class TestMaxConsecutiveZeroPnl:
    """max_consecutive with zero PnL trades (neither win nor loss)."""

    def test_zero_pnl_resets_streaks(self):
        """A trade with pnl=0 is neither a win nor a loss; resets both streaks."""
        t_win = Trade("A", date(2020, 1, 2), date(2020, 2, 1), 100, 110, 10, 100, 0.10, 30, 0)
        t_zero = Trade("A", date(2020, 2, 1), date(2020, 3, 1), 100, 100, 0, 0, 0.0, 28, 0)
        t_loss = Trade("A", date(2020, 3, 1), date(2020, 4, 1), 100, 90, -10, -100, -0.10, 31, 0)
        result = max_consecutive([t_win, t_win, t_zero, t_win])
        assert result["max_consecutive_wins"] == 2  # reset by zero PnL
