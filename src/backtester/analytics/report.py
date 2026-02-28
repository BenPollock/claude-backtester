"""Console and chart report output for backtest results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtester.analytics.metrics import (
    compute_all_metrics, cagr, sharpe_ratio, sortino_ratio, max_drawdown,
    max_drawdown_duration, total_return, calmar_ratio,
)
from backtester.result import BacktestResult


def _print_performance(label: str, equity, final_value: float, metrics: dict) -> None:
    """Print a performance section."""
    print(f"\n--- {label} ---")
    print(f"Final Equity:   ${final_value:,.2f}")
    print(f"Total Return:   {metrics['total_return']:.2%}")
    print(f"CAGR:           {metrics['cagr']:.2%}")
    print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:  {metrics['sortino_ratio']:.2f}")
    cr = metrics.get('calmar_ratio')
    if cr is not None:
        cr_str = f"{cr:.2f}" if cr != float("inf") else "inf"
        print(f"Calmar Ratio:   {cr_str}")
    print(f"Max Drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"Max DD Duration:{metrics['max_drawdown_duration_days']} days")


def print_report(result: BacktestResult) -> dict:
    """Print backtest results to console. Returns metrics dict."""
    equity = result.equity_series
    trades = result.trades
    config = result.config
    bm = result.benchmark_series

    metrics = compute_all_metrics(equity, trades, benchmark_series=bm)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nStrategy:       {config.strategy_name}")
    print(f"Tickers:        {', '.join(config.tickers)}")
    print(f"Benchmark:      {config.benchmark}")
    print(f"Period:         {config.start_date} to {config.end_date}")
    print(f"Starting Cash:  ${config.starting_cash:,.2f}")
    print(f"Max Positions:  {config.max_positions}")
    print(f"Max Allocation: {config.max_alloc_pct:.0%}")

    _print_performance("Strategy Performance", equity, equity.iloc[-1], metrics)

    # Benchmark buy & hold
    if bm is not None and len(bm) >= 2:
        bm_metrics = {
            "total_return": total_return(bm),
            "cagr": cagr(bm),
            "sharpe_ratio": sharpe_ratio(bm),
            "sortino_ratio": sortino_ratio(bm),
            "calmar_ratio": calmar_ratio(bm),
            "max_drawdown": max_drawdown(bm),
            "max_drawdown_duration_days": max_drawdown_duration(bm),
        }
        _print_performance(f"Benchmark Buy & Hold ({config.benchmark})", bm, bm.iloc[-1], bm_metrics)

    # Benchmark-relative metrics
    if "alpha" in metrics:
        print(f"\n--- Benchmark-Relative ---")
        print(f"Alpha (ann.):   {metrics['alpha']:.2%}")
        print(f"Beta:           {metrics['beta']:.2f}")
        ir = metrics['information_ratio']
        print(f"Info Ratio:     {ir:.2f}")
        print(f"Tracking Error: {metrics['tracking_error']:.2%}")
        print(f"Up Capture:     {metrics['up_capture']:.1f}%")
        print(f"Down Capture:   {metrics['down_capture']:.1f}%")

    print(f"\n--- Trades ---")
    print(f"Total Trades:   {metrics['total_trades']}")
    print(f"Win Rate:       {metrics['win_rate']:.2%}")
    pf = metrics['profit_factor']
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"Profit Factor:  {pf_str}")
    print(f"Expectancy:     ${metrics['trade_expectancy']:,.2f}")
    print(f"Exposure Time:  {metrics['exposure_time']:.1%}")

    if trades:
        pnls = [t.pnl for t in trades]
        print(f"Avg Trade PnL:  ${sum(pnls) / len(pnls):,.2f}")
        print(f"Best Trade:     ${max(pnls):,.2f}")
        print(f"Worst Trade:    ${min(pnls):,.2f}")

        wl = metrics
        aw = wl['avg_win']
        al = wl['avg_loss']
        pr = wl['payoff_ratio']
        pr_str = f"{pr:.2f}" if pr != float("inf") else "inf"
        print(f"Avg Winner:     ${aw:,.2f}")
        print(f"Avg Loser:      ${al:,.2f}")
        print(f"Payoff Ratio:   {pr_str}")

        print(f"Avg Hold Days:  {wl['avg_days']:.0f}  (W: {wl['avg_days_winners']:.0f} / L: {wl['avg_days_losers']:.0f})")
        print(f"Median Hold:    {wl['median_days']} days")
        print(f"Max Consec Win: {wl['max_consecutive_wins']}")
        print(f"Max Consec Loss:{wl['max_consecutive_losses']}")

    print(f"\nNote: Uses split-adjusted close prices. Dividends not included.")
    print("=" * 60 + "\n")

    # Calendar analytics
    from backtester.analytics.calendar import print_calendar_report
    print_calendar_report(equity)

    _print_activity_log(result)

    return metrics


def _print_activity_log(result: BacktestResult) -> None:
    """Print the per-fill activity log as a formatted table."""
    entries = result.activity_log
    if not entries:
        return

    print("--- Activity Log ---")
    header = f"{'Date':<12} {'Ticker':<8} {'Action':<6} {'Qty':>6} {'Price':>10} {'Value':>12} {'Cost Basis':>12} {'Fees':>8} {'Slippage':>10}"
    print(header)
    print("-" * len(header))
    for e in entries:
        cb = f"${e.avg_cost_basis:>10,.2f}" if e.avg_cost_basis is not None else f"{'N/A':>11}"
        print(
            f"{str(e.date):<12} {e.symbol:<8} {e.action.name:<6} {e.quantity:>6} "
            f"${e.price:>9,.2f} ${e.value:>11,.2f} {cb} ${e.fees:>7,.2f} ${e.slippage:>9,.4f}"
        )
    print()


def export_activity_log_csv(result: BacktestResult, filepath: str) -> None:
    """Write the activity log to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "symbol", "action", "quantity", "price", "value",
                         "avg_cost_basis", "fees", "slippage"])
        for e in result.activity_log:
            writer.writerow([
                str(e.date), e.symbol, e.action.name, e.quantity,
                f"{e.price:.4f}", f"{e.value:.2f}",
                f"{e.avg_cost_basis:.4f}" if e.avg_cost_basis is not None else "",
                f"{e.fees:.2f}", f"{e.slippage:.4f}",
            ])


def plot_results(result: BacktestResult) -> None:
    """Show equity curve vs benchmark, drawdown chart, and monthly heatmap."""
    equity = result.equity_series
    config = result.config

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # --- Top panel: equity curves ---
    ax1.plot(equity.index, equity.values, label=config.strategy_name, linewidth=1.5)

    bm = result.benchmark_series
    if bm is not None and len(bm) >= 2:
        ax1.plot(bm.index, bm.values, label=f"{config.benchmark} Buy & Hold",
                 linewidth=1.2, alpha=0.7)

    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"{config.strategy_name}  |  {config.start_date} to {config.end_date}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: strategy drawdown ---
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100  # as percentage
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.35)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

    # Monthly returns heatmap
    from backtester.analytics.calendar import plot_monthly_heatmap
    plot_monthly_heatmap(equity)
