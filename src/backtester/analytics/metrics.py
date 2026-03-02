"""Performance metrics: CAGR, Sharpe, Sortino, Max Drawdown, benchmark-relative, etc."""

import numpy as np
import pandas as pd
from statistics import median


def cagr(equity_series: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_series) < 2:
        return 0.0
    start_val = equity_series.iloc[0]
    end_val = equity_series.iloc[-1]
    if start_val <= 0:
        return 0.0
    days = (equity_series.index[-1] - equity_series.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return (end_val / start_val) ** (1.0 / years) - 1.0


def sharpe_ratio(equity_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily equity series."""
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / 252.0
    std = excess.std()
    if std == 0:
        return 0.0
    return (excess.mean() / std) * np.sqrt(252)


def sortino_ratio(equity_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / 252.0
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return float("inf")
    return (excess.mean() / downside_std) * np.sqrt(252)


def max_drawdown(equity_series: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.25 = 25% drawdown)."""
    if len(equity_series) < 2:
        return 0.0
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    return drawdown.min()


def max_drawdown_duration(equity_series: pd.Series) -> int:
    """Maximum drawdown duration in calendar days."""
    if len(equity_series) < 2:
        return 0
    cummax = equity_series.cummax()
    underwater = equity_series < cummax

    max_duration = 0
    current_start = None

    for i, is_underwater in enumerate(underwater):
        if is_underwater:
            if current_start is None:
                current_start = equity_series.index[i]
        else:
            if current_start is not None:
                duration = (equity_series.index[i] - current_start).days
                max_duration = max(max_duration, duration)
                current_start = None

    # Check if still in drawdown at end
    if current_start is not None:
        duration = (equity_series.index[-1] - current_start).days
        max_duration = max(max_duration, duration)

    return max_duration


def total_return(equity_series: pd.Series) -> float:
    """Total return as a fraction."""
    if len(equity_series) < 2:
        return 0.0
    return equity_series.iloc[-1] / equity_series.iloc[0] - 1.0


def win_rate(trades) -> float:
    """Fraction of trades with positive PnL."""
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t.pnl > 0)
    return winners / len(trades)


def profit_factor(trades) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calmar_ratio(equity_series: pd.Series) -> float:
    """Calmar Ratio: CAGR / |MaxDD|."""
    c = cagr(equity_series)
    dd = max_drawdown(equity_series)
    if dd == 0:
        return float("inf") if c > 0 else 0.0
    return c / abs(dd)


def _aligned_returns(
    equity_series: pd.Series, benchmark_series: pd.Series
) -> pd.DataFrame | None:
    """Compute daily returns for strategy and benchmark, inner-join align them.

    Returns a two-column DataFrame (strategy_returns, benchmark_returns) with
    matching dates and no NaNs, or None if fewer than 2 overlapping rows exist.
    """
    strat_ret = equity_series.pct_change().dropna()
    bm_ret = benchmark_series.pct_change().dropna()
    aligned = pd.concat([strat_ret, bm_ret], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return None
    return aligned


def beta(equity_series: pd.Series, benchmark_series: pd.Series) -> float:
    """Beta: covariance of strategy and benchmark returns / variance of benchmark."""
    aligned = _aligned_returns(equity_series, benchmark_series)
    if aligned is None:
        return 0.0
    cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    bm_var = cov_matrix[1, 1]
    if bm_var == 0:
        return 0.0
    return cov_matrix[0, 1] / bm_var


def alpha(equity_series: pd.Series, benchmark_series: pd.Series,
          risk_free_rate: float = 0.0) -> float:
    """Jensen's Alpha: annualized excess return vs CAPM prediction."""
    aligned = _aligned_returns(equity_series, benchmark_series)
    if aligned is None:
        return 0.0
    b = beta(equity_series, benchmark_series)
    daily_rf = risk_free_rate / 252.0
    mean_strat = aligned.iloc[:, 0].mean()
    mean_bm = aligned.iloc[:, 1].mean()
    daily_alpha = mean_strat - daily_rf - b * (mean_bm - daily_rf)
    return daily_alpha * 252.0


def information_ratio(equity_series: pd.Series, benchmark_series: pd.Series) -> float:
    """Information Ratio: annualized active return / tracking error."""
    aligned = _aligned_returns(equity_series, benchmark_series)
    if aligned is None:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std()
    if te == 0:
        return 0.0
    return (active.mean() / te) * np.sqrt(252)


def tracking_error(equity_series: pd.Series, benchmark_series: pd.Series) -> float:
    """Tracking Error: annualized std of active returns."""
    aligned = _aligned_returns(equity_series, benchmark_series)
    if aligned is None:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return active.std() * np.sqrt(252)


def capture_ratio(equity_series: pd.Series, benchmark_series: pd.Series,
                  side: str = "up") -> float:
    """Up or Down capture ratio vs benchmark.

    Up capture > 100% means you gain more than the benchmark in up markets.
    Down capture < 100% means you lose less than the benchmark in down markets.
    """
    aligned = _aligned_returns(equity_series, benchmark_series)
    if aligned is None:
        return 0.0
    s, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    if side == "up":
        mask = b > 0
    else:
        mask = b < 0
    if mask.sum() == 0:
        return 0.0
    return s[mask].mean() / b[mask].mean() * 100.0


# ── Trade-level statistics ─────────────────────────────────────────


def trade_expectancy(trades) -> float:
    """Average PnL per trade (expectancy)."""
    if not trades:
        return 0.0
    return sum(t.pnl for t in trades) / len(trades)


def avg_win_loss(trades) -> dict:
    """Average winner and loser PnL, and payoff ratio."""
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl < 0]
    avg_win = sum(t.pnl for t in winners) / len(winners) if winners else 0.0
    avg_loss = sum(t.pnl for t in losers) / len(losers) if losers else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf") if avg_win > 0 else 0.0
    return {"avg_win": avg_win, "avg_loss": avg_loss, "payoff_ratio": payoff}


def holding_period_stats(trades) -> dict:
    """Avg/median holding period overall, for winners, and for losers."""
    if not trades:
        return {"avg_days": 0, "median_days": 0,
                "avg_days_winners": 0, "avg_days_losers": 0}
    all_days = [t.holding_days for t in trades]
    win_days = [t.holding_days for t in trades if t.pnl > 0]
    loss_days = [t.holding_days for t in trades if t.pnl < 0]
    return {
        "avg_days": sum(all_days) / len(all_days),
        "median_days": median(all_days),
        "avg_days_winners": sum(win_days) / len(win_days) if win_days else 0,
        "avg_days_losers": sum(loss_days) / len(loss_days) if loss_days else 0,
    }


def max_consecutive(trades) -> dict:
    """Max consecutive winners and losers."""
    if not trades:
        return {"max_consecutive_wins": 0, "max_consecutive_losses": 0}
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0
    for t in trades:
        if t.pnl > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        elif t.pnl < 0:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)
        else:
            cur_wins = cur_losses = 0
    return {"max_consecutive_wins": max_wins, "max_consecutive_losses": max_losses}


def exposure_time(equity_series: pd.Series, trades) -> float:
    """Fraction of trading days where at least one position was open."""
    if len(equity_series) < 2 or not trades:
        return 0.0
    total_days = len(equity_series)
    # Build set of dates when positions were held
    invested_dates = set()
    for t in trades:
        # Generate business days between entry and exit
        days = pd.bdate_range(t.entry_date, t.exit_date)
        invested_dates.update(d.date() for d in days)
    # Intersect with actual equity dates
    equity_dates = set(
        d.date() if hasattr(d, 'date') else d for d in equity_series.index
    )
    return len(invested_dates & equity_dates) / total_days


def historical_var(equity_series: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (Gap 6).

    Returns the daily return at the (1-confidence) percentile.
    E.g., VaR at 95% confidence returns the 5th percentile of returns.
    """
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar(equity_series: pd.Series, confidence: float = 0.95) -> float:
    """Conditional Value at Risk (Expected Shortfall) (Gap 6).

    Mean of returns below the VaR threshold.
    """
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    var = historical_var(equity_series, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def omega_ratio(equity_series: pd.Series, threshold: float = 0.0) -> float:
    """Omega Ratio: gains above threshold / losses below threshold (Gap 43)."""
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if losses.sum() == 0:
        return float("inf") if gains.sum() > 0 else 0.0
    return float(gains.sum() / losses.sum())


def treynor_ratio(
    equity_series: pd.Series,
    benchmark_series: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Treynor Ratio: (CAGR - risk_free) / beta (Gap 43)."""
    b = beta(equity_series, benchmark_series)
    if b == 0:
        return 0.0
    c = cagr(equity_series)
    return (c - risk_free_rate) / b


def compute_all_metrics(
    equity_series: pd.Series,
    trades,
    risk_free_rate: float = 0.0,
    benchmark_series: pd.Series | None = None,
) -> dict:
    """Compute all available metrics and return as a dict."""
    m = {
        "total_return": total_return(equity_series),
        "cagr": cagr(equity_series),
        "sharpe_ratio": sharpe_ratio(equity_series, risk_free_rate),
        "sortino_ratio": sortino_ratio(equity_series, risk_free_rate),
        "calmar_ratio": calmar_ratio(equity_series),
        "max_drawdown": max_drawdown(equity_series),
        "max_drawdown_duration_days": max_drawdown_duration(equity_series),
        "total_trades": len(trades),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "trade_expectancy": trade_expectancy(trades),
        "exposure_time": exposure_time(equity_series, trades),
        **avg_win_loss(trades),
        **holding_period_stats(trades),
        **max_consecutive(trades),
    }

    # Risk metrics (Gap 6)
    m["var_95"] = historical_var(equity_series, 0.95)
    m["cvar_95"] = cvar(equity_series, 0.95)

    # Omega ratio (Gap 43)
    m["omega_ratio"] = omega_ratio(equity_series)

    # Benchmark-relative metrics (only when benchmark provided)
    if benchmark_series is not None and len(benchmark_series) >= 2:
        m["alpha"] = alpha(equity_series, benchmark_series, risk_free_rate)
        m["beta"] = beta(equity_series, benchmark_series)
        m["information_ratio"] = information_ratio(equity_series, benchmark_series)
        m["tracking_error"] = tracking_error(equity_series, benchmark_series)
        m["up_capture"] = capture_ratio(equity_series, benchmark_series, "up")
        m["down_capture"] = capture_ratio(equity_series, benchmark_series, "down")
        # Treynor ratio (Gap 43)
        m["treynor_ratio"] = treynor_ratio(equity_series, benchmark_series, risk_free_rate)

    return m
