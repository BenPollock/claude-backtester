"""HTML tearsheet / report export for backtest results.

Generates a self-contained HTML file with embedded charts, metrics tables,
monthly returns heatmap, and trade statistics.
"""

import base64
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for rendering to buffers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from backtester.analytics.metrics import (
    cagr, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    win_rate, profit_factor, avg_win_loss, holding_period_stats,
    max_consecutive, compute_all_metrics,
)
from backtester.analytics.calendar import monthly_returns
from backtester.result import BacktestResult


def _render_chart_as_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string.

    Args:
        fig: A matplotlib Figure object.

    Returns:
        Base64-encoded PNG string suitable for embedding in an HTML img tag.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _generate_monthly_returns_html(equity_curve: pd.Series) -> str:
    """Generate an HTML table showing monthly returns as a Year x Month heatmap.

    Args:
        equity_curve: Daily equity series indexed by date.

    Returns:
        HTML string containing the monthly returns table.
    """
    if len(equity_curve) < 2:
        return "<p>Insufficient data for monthly returns.</p>"

    try:
        mr = monthly_returns(equity_curve)
    except Exception:
        return "<p>Could not compute monthly returns.</p>"

    if mr.empty:
        return "<p>No monthly returns data available.</p>"

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        "YTD": "YTD",
    }

    html = '<table class="monthly-returns">\n<thead><tr><th>Year</th>'
    for col in mr.columns:
        label = month_names.get(col, str(col))
        html += f"<th>{label}</th>"
    html += "</tr></thead>\n<tbody>\n"

    for year, row in mr.iterrows():
        html += f"<tr><td class='year-cell'>{year}</td>"
        for col in mr.columns:
            val = row.get(col)
            if pd.isna(val):
                html += "<td></td>"
            else:
                pct = val * 100
                # Color gradient: green for positive, red for negative
                if pct > 0:
                    intensity = min(pct / 10.0, 1.0)
                    bg = f"rgba(39, 174, 96, {0.15 + 0.65 * intensity})"
                    color = "#1a1a2e" if intensity < 0.6 else "white"
                elif pct < 0:
                    intensity = min(abs(pct) / 10.0, 1.0)
                    bg = f"rgba(231, 76, 60, {0.15 + 0.65 * intensity})"
                    color = "#1a1a2e" if intensity < 0.6 else "white"
                else:
                    bg = "transparent"
                    color = "#1a1a2e"
                html += (
                    f'<td style="background:{bg};color:{color}">'
                    f"{pct:.1f}%</td>"
                )
        html += "</tr>\n"

    html += "</tbody></table>"
    return html


def _generate_trade_table_html(trades, max_rows: int = 20) -> str:
    """Generate an HTML table showing recent trades.

    Args:
        trades: List of Trade objects from the backtest.
        max_rows: Maximum number of trades to display.

    Returns:
        HTML string containing the trade list table.
    """
    if not trades:
        return "<p>No trades executed.</p>"

    # Show the last max_rows trades
    display_trades = trades[-max_rows:]

    html = '<table class="trade-table">\n<thead><tr>'
    html += "<th>Entry Date</th><th>Exit Date</th><th>Ticker</th>"
    html += "<th>Qty</th><th>Entry Price</th><th>Exit Price</th>"
    html += "<th>PnL</th><th>Return %</th><th>Hold Days</th>"
    html += "</tr></thead>\n<tbody>\n"

    for t in display_trades:
        pnl_class = "positive" if t.pnl >= 0 else "negative"
        html += "<tr>"
        html += f"<td>{t.entry_date}</td>"
        html += f"<td>{t.exit_date}</td>"
        html += f"<td>{t.symbol}</td>"
        html += f"<td>{t.quantity}</td>"
        html += f"<td>${t.entry_price:,.2f}</td>"
        html += f"<td>${t.exit_price:,.2f}</td>"
        html += f'<td class="{pnl_class}">${t.pnl:,.2f}</td>'
        html += f'<td class="{pnl_class}">{t.pnl_pct:.2%}</td>'
        html += f"<td>{t.holding_days}</td>"
        html += "</tr>\n"

    html += "</tbody></table>"

    if len(trades) > max_rows:
        html += f"<p class='note'>Showing last {max_rows} of {len(trades)} trades.</p>"

    return html


def _build_equity_chart(equity_curve: pd.Series, benchmark: pd.Series | None,
                        strategy_name: str) -> str:
    """Build the equity curve chart and return base64 PNG."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(equity_curve.index, equity_curve.values,
            label=strategy_name, linewidth=1.5, color="#2980b9")

    if benchmark is not None and len(benchmark) >= 2:
        ax.plot(benchmark.index, benchmark.values,
                label="Benchmark", linewidth=1.2, alpha=0.7, color="#95a5a6")

    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Equity Curve")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()

    return _render_chart_as_base64(fig)


def _build_drawdown_chart(equity_curve: pd.Series) -> str:
    """Build the underwater (drawdown) chart and return base64 PNG."""
    fig, ax = plt.subplots(figsize=(10, 3))

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax * 100  # as percentage

    ax.fill_between(drawdown.index, drawdown.values, 0,
                     color="#e74c3c", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values,
            color="#c0392b", linewidth=0.8)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Underwater Plot")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()

    return _render_chart_as_base64(fig)


def _build_rolling_metrics_chart(equity_curve: pd.Series) -> str | None:
    """Build rolling 12-month Sharpe and rolling returns chart.

    Returns None if there are fewer than 252 data points (roughly 1 year).
    """
    if len(equity_curve) < 252:
        return None

    daily_returns = equity_curve.pct_change().dropna()
    window = 252  # approximately 12 months of trading days

    # Rolling annualized return
    rolling_ret = daily_returns.rolling(window).apply(
        lambda x: (1 + x).prod() ** (252 / len(x)) - 1, raw=False
    )

    # Rolling Sharpe
    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    # Replace inf/nan
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Rolling return
    ax1.plot(rolling_ret.index, rolling_ret.values * 100,
             color="#2980b9", linewidth=1.2)
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Return (%)")
    ax1.set_title("Rolling 12-Month Metrics")
    ax1.grid(True, alpha=0.3)

    # Rolling Sharpe
    ax2.plot(rolling_sharpe.index, rolling_sharpe.values,
             color="#27ae60", linewidth=1.2)
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()

    return _render_chart_as_base64(fig)


def _format_metric(value, fmt: str = ".2f") -> str:
    """Format a metric value, handling infinity and None."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return "inf" if value > 0 else "-inf"
    return f"{value:{fmt}}"


def generate_tearsheet(result: BacktestResult,
                       output_path: str = "tearsheet.html") -> str:
    """Generate a self-contained HTML tearsheet from backtest results.

    The output HTML file includes embedded charts (as base64 PNGs), metrics
    tables, a monthly returns heatmap, trade statistics, and a trade list.
    No external CSS or JS dependencies are required.

    Args:
        result: A BacktestResult from BacktestEngine.run().
        output_path: File path to write the HTML file to.

    Returns:
        The output file path.
    """
    equity = result.equity_series
    trades = result.trades
    config = result.config
    benchmark = result.benchmark_series

    # Compute metrics
    metrics = compute_all_metrics(equity, trades, benchmark_series=benchmark)

    # Build charts
    equity_chart_b64 = _build_equity_chart(equity, benchmark, config.strategy_name)
    drawdown_chart_b64 = _build_drawdown_chart(equity)
    rolling_chart_b64 = _build_rolling_metrics_chart(equity)

    # Build HTML sections
    monthly_html = _generate_monthly_returns_html(equity)
    trade_table_html = _generate_trade_table_html(trades)

    # Trade statistics
    wl = avg_win_loss(trades)
    hp = holding_period_stats(trades)
    mc = max_consecutive(trades)

    # Format key metrics for display
    def pct(v):
        return _format_metric(v, ".2%") if isinstance(v, (int, float)) else "N/A"

    def dec(v):
        return _format_metric(v, ".2f")

    # Header info
    start_str = str(config.start_date)
    end_str = str(config.end_date)
    final_equity = equity.iloc[-1] if len(equity) > 0 else config.starting_cash

    # Build the full HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Tearsheet - {config.strategy_name}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        background: #f5f6fa;
        color: #1a1a2e;
        line-height: 1.6;
    }}
    .header {{
        background: #1a1a2e;
        color: white;
        padding: 24px 40px;
    }}
    .header h1 {{
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    .header .meta {{
        font-size: 14px;
        color: #a0a0c0;
    }}
    .header .meta span {{
        margin-right: 24px;
    }}
    .container {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px 40px;
    }}
    .section {{
        background: white;
        border-radius: 8px;
        border: 1px solid #e0e0e8;
        margin-bottom: 24px;
        overflow: hidden;
    }}
    .section-title {{
        font-size: 16px;
        font-weight: 600;
        padding: 14px 20px;
        border-bottom: 1px solid #e0e0e8;
        background: #fafbfc;
    }}
    .section-body {{
        padding: 20px;
    }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
    }}
    .metric-card {{
        text-align: center;
        padding: 12px 8px;
        border: 1px solid #e8e8f0;
        border-radius: 6px;
        background: #fafbfc;
    }}
    .metric-card .label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666;
        margin-bottom: 4px;
    }}
    .metric-card .value {{
        font-size: 20px;
        font-weight: 700;
    }}
    .chart-img {{
        width: 100%;
        display: block;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    th {{
        background: #f0f1f5;
        font-weight: 600;
        text-align: left;
        padding: 8px 12px;
        border-bottom: 2px solid #d0d0d8;
    }}
    td {{
        padding: 6px 12px;
        border-bottom: 1px solid #eee;
        text-align: left;
    }}
    .monthly-returns th,
    .monthly-returns td {{
        text-align: center;
        padding: 5px 8px;
        font-size: 12px;
    }}
    .monthly-returns .year-cell {{
        font-weight: 600;
        text-align: left;
    }}
    .trade-table th {{ text-align: right; }}
    .trade-table th:nth-child(1),
    .trade-table th:nth-child(2),
    .trade-table th:nth-child(3) {{ text-align: left; }}
    .trade-table td {{ text-align: right; }}
    .trade-table td:nth-child(1),
    .trade-table td:nth-child(2),
    .trade-table td:nth-child(3) {{ text-align: left; }}
    .positive {{ color: #27ae60; font-weight: 600; }}
    .negative {{ color: #e74c3c; font-weight: 600; }}
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }}
    .stat-row {{
        display: flex;
        justify-content: space-between;
        padding: 6px 12px;
        border-bottom: 1px solid #f0f0f0;
    }}
    .stat-row .stat-label {{
        color: #555;
        font-size: 13px;
    }}
    .stat-row .stat-value {{
        font-weight: 600;
        font-size: 13px;
    }}
    .note {{
        font-size: 12px;
        color: #888;
        margin-top: 8px;
        font-style: italic;
    }}
    .footer {{
        text-align: center;
        font-size: 12px;
        color: #aaa;
        padding: 20px;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>Backtest Tearsheet: {config.strategy_name}</h1>
    <div class="meta">
        <span>Period: {start_str} to {end_str}</span>
        <span>Initial Capital: ${config.starting_cash:,.2f}</span>
        <span>Final Equity: ${final_equity:,.2f}</span>
        <span>Tickers: {', '.join(config.tickers)}</span>
        {"" if benchmark is None else f'<span>Benchmark: {config.benchmark}</span>'}
    </div>
</div>

<div class="container">

    <!-- Key Metrics -->
    <div class="section">
        <div class="section-title">Key Performance Metrics</div>
        <div class="section-body">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">CAGR</div>
                    <div class="value">{metrics['cagr']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Sharpe Ratio</div>
                    <div class="value">{dec(metrics['sharpe_ratio'])}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Sortino Ratio</div>
                    <div class="value">{dec(metrics['sortino_ratio'])}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Max Drawdown</div>
                    <div class="value">{metrics['max_drawdown']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Calmar Ratio</div>
                    <div class="value">{dec(metrics['calmar_ratio'])}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Win Rate</div>
                    <div class="value">{metrics['win_rate']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Profit Factor</div>
                    <div class="value">{dec(metrics['profit_factor'])}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Total Trades</div>
                    <div class="value">{metrics['total_trades']}</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Equity Curve -->
    <div class="section">
        <div class="section-title">Equity Curve</div>
        <div class="section-body" style="padding:0;">
            <img class="chart-img" src="data:image/png;base64,{equity_chart_b64}"
                 alt="Equity Curve">
        </div>
    </div>

    <!-- Drawdown -->
    <div class="section">
        <div class="section-title">Drawdown</div>
        <div class="section-body" style="padding:0;">
            <img class="chart-img" src="data:image/png;base64,{drawdown_chart_b64}"
                 alt="Drawdown Chart">
        </div>
    </div>

    <!-- Rolling Metrics (only if enough data) -->
    {"" if rolling_chart_b64 is None else f'''
    <div class="section">
        <div class="section-title">Rolling 12-Month Metrics</div>
        <div class="section-body" style="padding:0;">
            <img class="chart-img" src="data:image/png;base64,{rolling_chart_b64}"
                 alt="Rolling Metrics">
        </div>
    </div>
    '''}

    <!-- Monthly Returns Heatmap -->
    <div class="section">
        <div class="section-title">Monthly Returns</div>
        <div class="section-body">
            {monthly_html}
        </div>
    </div>

    <!-- Trade Statistics -->
    <div class="section">
        <div class="section-title">Trade Statistics</div>
        <div class="section-body">
            <div class="stats-grid">
                <div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Winner</span>
                        <span class="stat-value">${wl['avg_win']:,.2f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Loser</span>
                        <span class="stat-value">${wl['avg_loss']:,.2f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Payoff Ratio</span>
                        <span class="stat-value">{dec(wl['payoff_ratio'])}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Expectancy</span>
                        <span class="stat-value">${metrics['trade_expectancy']:,.2f}</span>
                    </div>
                </div>
                <div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Holding Period</span>
                        <span class="stat-value">{hp['avg_days']:.0f} days</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Median Holding Period</span>
                        <span class="stat-value">{hp['median_days']} days</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Max Consecutive Wins</span>
                        <span class="stat-value">{mc['max_consecutive_wins']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Max Consecutive Losses</span>
                        <span class="stat-value">{mc['max_consecutive_losses']}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Trade List -->
    <div class="section">
        <div class="section-title">Recent Trades</div>
        <div class="section-body">
            {trade_table_html}
        </div>
    </div>

</div>

<div class="footer">
    Generated by claude-backtester
</div>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
