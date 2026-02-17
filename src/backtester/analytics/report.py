"""Console and chart report output for backtest results."""

import matplotlib.pyplot as plt
import pandas as pd

from backtester.analytics.metrics import (
    compute_all_metrics, cagr, sharpe_ratio, sortino_ratio, max_drawdown,
    max_drawdown_duration, total_return,
)
from backtester.engine import BacktestResult


def _print_performance(label: str, equity, final_value: float, metrics: dict) -> None:
    """Print a performance section."""
    print(f"\n--- {label} ---")
    print(f"Final Equity:   ${final_value:,.2f}")
    print(f"Total Return:   {metrics['total_return']:.2%}")
    print(f"CAGR:           {metrics['cagr']:.2%}")
    print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:  {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"Max DD Duration:{metrics['max_drawdown_duration_days']} days")


def print_report(result: BacktestResult) -> dict:
    """Print backtest results to console. Returns metrics dict."""
    equity = result.equity_series
    trades = result.trades
    config = result.config

    metrics = compute_all_metrics(equity, trades)

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
    bm = result.benchmark_series
    if bm is not None and len(bm) >= 2:
        bm_metrics = {
            "total_return": total_return(bm),
            "cagr": cagr(bm),
            "sharpe_ratio": sharpe_ratio(bm),
            "sortino_ratio": sortino_ratio(bm),
            "max_drawdown": max_drawdown(bm),
            "max_drawdown_duration_days": max_drawdown_duration(bm),
        }
        _print_performance(f"Benchmark Buy & Hold ({config.benchmark})", bm, bm.iloc[-1], bm_metrics)

    print(f"\n--- Trades ---")
    print(f"Total Trades:   {metrics['total_trades']}")
    print(f"Win Rate:       {metrics['win_rate']:.2%}")
    pf = metrics['profit_factor']
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"Profit Factor:  {pf_str}")

    if trades:
        pnls = [t.pnl for t in trades]
        print(f"Avg Trade PnL:  ${sum(pnls) / len(pnls):,.2f}")
        print(f"Best Trade:     ${max(pnls):,.2f}")
        print(f"Worst Trade:    ${min(pnls):,.2f}")
        holding_days = [t.holding_days for t in trades]
        print(f"Avg Hold Days:  {sum(holding_days) / len(holding_days):.0f}")

    print(f"\nNote: Uses split-adjusted close prices. Dividends not included.")
    print("=" * 60 + "\n")

    return metrics


def plot_results(result: BacktestResult) -> None:
    """Show equity curve vs benchmark and drawdown chart."""
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
