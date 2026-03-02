"""Transaction Cost Analysis, Turnover, and Capacity estimation (Gaps 22/23/24)."""

import pandas as pd


def compute_turnover(
    trades: list,
    equity_series: pd.Series,
) -> float:
    """Compute annual portfolio turnover rate.

    turnover = sum(trade_values) / avg_equity / years
    """
    if not trades or len(equity_series) < 2:
        return 0.0

    total_traded = sum(t.quantity * t.entry_price for t in trades)
    avg_equity = equity_series.mean()
    if avg_equity <= 0:
        return 0.0

    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25
    if years <= 0:
        return 0.0

    return total_traded / avg_equity / years


def compute_cost_attribution(
    trades: list,
    equity_series: pd.Series,
) -> dict:
    """Compute transaction cost attribution.

    Returns total fees, cost as % of average equity, and cost as %
    of total return.
    """
    if not trades:
        return {"total_fees": 0.0, "cost_pct_equity": 0.0, "cost_pct_return": 0.0}

    total_fees = sum(t.fees_total for t in trades)
    avg_equity = equity_series.mean() if len(equity_series) > 0 else 0.0

    cost_pct_equity = total_fees / avg_equity if avg_equity > 0 else 0.0

    total_return_val = equity_series.iloc[-1] - equity_series.iloc[0] if len(equity_series) >= 2 else 0.0
    cost_pct_return = total_fees / abs(total_return_val) if total_return_val != 0 else 0.0

    return {
        "total_fees": total_fees,
        "cost_pct_equity": cost_pct_equity,
        "cost_pct_return": cost_pct_return,
    }


def estimate_capacity(
    trades: list,
    price_data: dict[str, pd.DataFrame],
    max_volume_pct: float = 0.01,
) -> float:
    """Estimate strategy capacity (AUM at which fill constraints bind).

    For each trade, compute the max position value that would stay
    within max_volume_pct of daily volume. The minimum across all
    trades is the capacity bottleneck.
    """
    if not trades or not price_data:
        return 0.0

    capacities = []
    for trade in trades:
        df = price_data.get(trade.symbol)
        if df is None:
            continue

        entry_ts = pd.Timestamp(trade.entry_date)
        if entry_ts in df.index:
            row = df.loc[entry_ts]
            volume = row.get("Volume", 0)
            price = row.get("Close", 0)
            if volume > 0 and price > 0:
                max_shares = int(volume * max_volume_pct)
                max_value = max_shares * price
                if max_value > 0:
                    capacities.append(max_value)

    if not capacities:
        return 0.0

    return min(capacities)
