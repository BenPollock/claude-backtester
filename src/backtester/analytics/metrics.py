"""Performance metrics: CAGR, Sharpe, Sortino, Max Drawdown, etc."""

import numpy as np
import pandas as pd


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


def compute_all_metrics(equity_series: pd.Series, trades, risk_free_rate: float = 0.0) -> dict:
    """Compute all available metrics and return as a dict."""
    return {
        "total_return": total_return(equity_series),
        "cagr": cagr(equity_series),
        "sharpe_ratio": sharpe_ratio(equity_series, risk_free_rate),
        "sortino_ratio": sortino_ratio(equity_series, risk_free_rate),
        "max_drawdown": max_drawdown(equity_series),
        "max_drawdown_duration_days": max_drawdown_duration(equity_series),
        "total_trades": len(trades),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
    }
