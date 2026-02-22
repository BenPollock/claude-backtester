"""Calendar-based analytics: monthly returns, drawdown periods, yearly stats."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def monthly_returns(equity_series: pd.Series) -> pd.DataFrame:
    """Compute monthly returns as a Year x Month pivot table.

    Returns DataFrame with years as rows and months (1-12) as columns.
    Values are fractional returns (e.g. 0.05 = 5%).
    """
    daily_ret = equity_series.pct_change().dropna()
    daily_ret.index = pd.DatetimeIndex(daily_ret.index)
    monthly = daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly.index.names = ["Year", "Month"]
    table = monthly.unstack(level="Month")
    table.columns = list(range(1, len(table.columns) + 1))
    # Add YTD column
    table["YTD"] = table.apply(lambda row: (1 + row.dropna()).prod() - 1, axis=1)
    return table


def drawdown_periods(equity_series: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Identify the top N drawdown periods by depth.

    Returns DataFrame with columns: start, trough, recovery, depth, duration_days.
    """
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax

    periods = []
    in_dd = False
    start = None
    trough_date = None
    trough_val = 0.0

    for i, (dt, dd) in enumerate(drawdown.items()):
        if dd < 0:
            if not in_dd:
                in_dd = True
                start = dt
                trough_date = dt
                trough_val = dd
            if dd < trough_val:
                trough_val = dd
                trough_date = dt
        else:
            if in_dd:
                periods.append({
                    "start": start,
                    "trough": trough_date,
                    "recovery": dt,
                    "depth": trough_val,
                    "duration_days": (dt - start).days,
                })
                in_dd = False

    # If still in drawdown at end
    if in_dd:
        periods.append({
            "start": start,
            "trough": trough_date,
            "recovery": None,
            "depth": trough_val,
            "duration_days": (equity_series.index[-1] - start).days,
        })

    df = pd.DataFrame(periods)
    if df.empty:
        return df
    return df.sort_values("depth").head(top_n).reset_index(drop=True)


def yearly_summary(equity_series: pd.Series) -> pd.DataFrame:
    """Yearly return, max drawdown, and Sharpe per calendar year."""
    daily_ret = equity_series.pct_change().dropna()
    daily_ret.index = pd.DatetimeIndex(daily_ret.index)
    years = daily_ret.index.year.unique()

    rows = []
    for yr in sorted(years):
        yr_ret = daily_ret[daily_ret.index.year == yr]
        yr_equity = (1 + yr_ret).cumprod()
        annual_return = yr_equity.iloc[-1] - 1 if len(yr_equity) > 0 else 0.0
        # Max drawdown
        cummax = yr_equity.cummax()
        dd = ((yr_equity - cummax) / cummax).min()
        # Sharpe
        std = yr_ret.std()
        sharpe = (yr_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        rows.append({
            "year": yr,
            "return": annual_return,
            "max_drawdown": dd,
            "sharpe": sharpe,
            "trading_days": len(yr_ret),
        })
    return pd.DataFrame(rows)


def print_calendar_report(equity_series: pd.Series) -> None:
    """Print calendar-based analytics to console."""
    # Monthly returns table
    mr = monthly_returns(equity_series)
    if not mr.empty:
        print("\n--- Monthly Returns ---")
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        cols = [month_names.get(c, str(c)) for c in mr.columns if c != "YTD"] + ["YTD"]
        header = f"{'Year':<6}" + "".join(f"{c:>7}" for c in cols)
        print(header)
        print("-" * len(header))
        for yr, row in mr.iterrows():
            vals = []
            for c in mr.columns:
                v = row.get(c)
                if pd.isna(v):
                    vals.append(f"{'':>7}")
                else:
                    vals.append(f"{v:>6.1%}")
            print(f"{yr:<6}" + " ".join(vals))

    # Yearly summary
    ys = yearly_summary(equity_series)
    if not ys.empty:
        print("\n--- Yearly Summary ---")
        print(f"{'Year':<6} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8}")
        print("-" * 32)
        for _, row in ys.iterrows():
            print(f"{int(row['year']):<6} {row['return']:>7.1%} {row['max_drawdown']:>7.1%} {row['sharpe']:>8.2f}")

    # Top drawdown periods
    dp = drawdown_periods(equity_series)
    if not dp.empty:
        print("\n--- Top Drawdown Periods ---")
        print(f"{'Start':<12} {'Trough':<12} {'Recovery':<12} {'Depth':>8} {'Days':>6}")
        print("-" * 52)
        for _, row in dp.iterrows():
            rec = str(row['recovery'])[:10] if row['recovery'] is not None else "ongoing"
            print(f"{str(row['start'])[:10]:<12} {str(row['trough'])[:10]:<12} "
                  f"{rec:<12} {row['depth']:>7.1%} {row['duration_days']:>6}")


def plot_monthly_heatmap(equity_series: pd.Series) -> None:
    """Plot a monthly returns heatmap."""
    mr = monthly_returns(equity_series)
    if mr.empty:
        return

    # Drop YTD column for the heatmap
    plot_data = mr.drop(columns=["YTD"], errors="ignore")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, max(3, len(plot_data) * 0.5)))
    data = plot_data.values * 100  # convert to percentage

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-10, vmax=10)

    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(plot_data.index)))
    ax.set_yticklabels(plot_data.index)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 6 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color=color, fontsize=8)

    ax.set_title("Monthly Returns (%)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    plt.show()
