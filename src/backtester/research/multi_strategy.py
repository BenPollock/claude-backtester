"""Multi-strategy portfolio: run multiple strategies with independent capital allocations."""

import logging
from dataclasses import dataclass, replace

import pandas as pd

from backtester.config import BacktestConfig
from backtester.engine import BacktestEngine
from backtester.result import BacktestResult
from backtester.analytics.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyAllocation:
    """One strategy's share of the combined portfolio.

    Attributes:
        strategy_name: Registered strategy name (e.g. 'sma_crossover').
        params: Strategy-specific parameters dict.
        weight: Fraction of total capital allocated, e.g. 0.5 = 50%.
    """
    strategy_name: str
    params: dict
    weight: float


@dataclass(frozen=True)
class MultiStrategyConfig:
    """Configuration for a multi-strategy portfolio run.

    Attributes:
        allocations: Tuple of StrategyAllocation entries.
        base_config: Shared config (tickers, dates, execution settings, etc.).
            The strategy_name, strategy_params, and starting_cash fields on
            base_config are overridden per allocation.
    """
    allocations: tuple[StrategyAllocation, ...]
    base_config: BacktestConfig

    def __post_init__(self) -> None:
        total_weight = sum(a.weight for a in self.allocations)
        if total_weight > 1.0 + 1e-9:
            raise ValueError(
                f"Allocation weights sum to {total_weight:.4f}, which exceeds 1.0"
            )
        for a in self.allocations:
            if a.weight < 0:
                raise ValueError(
                    f"Negative weight {a.weight} for strategy '{a.strategy_name}'"
                )


@dataclass(frozen=True)
class MultiStrategyResult:
    """Results from a multi-strategy portfolio run.

    Attributes:
        combined_equity_curve: Sum of individual equity curves, aligned by date.
        combined_metrics: CAGR, Sharpe, etc. computed on the combined curve.
        per_strategy_results: Individual BacktestResult keyed by strategy name.
        per_strategy_metrics: Metrics dict keyed by strategy name.
        per_strategy_weights: Allocation weights keyed by strategy name.
        combined_trades: All trades from all strategies merged into one list.
        attribution: Per-strategy contribution analysis DataFrame.
    """
    combined_equity_curve: pd.Series
    combined_metrics: dict
    per_strategy_results: dict[str, BacktestResult]
    per_strategy_metrics: dict[str, dict]
    per_strategy_weights: dict[str, float]
    combined_trades: list
    attribution: pd.DataFrame


def run_multi_strategy(
    multi_config: MultiStrategyConfig,
    data_manager=None,
) -> MultiStrategyResult:
    """Run each strategy allocation independently and combine results.

    Each allocation gets its proportional share of starting cash. Strategies
    run in complete isolation (no capital sharing or cross-strategy signals).

    Args:
        multi_config: Multi-strategy configuration.
        data_manager: Optional DataManager to inject (useful for testing with
            MockDataSource). If None, each engine creates its own.

    Returns:
        MultiStrategyResult with combined and per-strategy analytics.
    """
    base = multi_config.base_config
    per_strategy_results: dict[str, BacktestResult] = {}
    per_strategy_metrics: dict[str, dict] = {}
    per_strategy_weights: dict[str, float] = {}
    combined_trades: list = []

    # Build unique labels for each allocation. When the same strategy name
    # appears more than once, append a numeric suffix (e.g. sma_crossover_2).
    name_counts: dict[str, int] = {}
    labels: list[str] = []
    for alloc in multi_config.allocations:
        count = name_counts.get(alloc.strategy_name, 0) + 1
        name_counts[alloc.strategy_name] = count
        if count == 1:
            labels.append(alloc.strategy_name)
        else:
            labels.append(f"{alloc.strategy_name}_{count}")

    # If any strategy appeared more than once, retroactively fix the first
    # occurrence to also carry a _1 suffix for consistency.
    for name, count in name_counts.items():
        if count > 1:
            for i, lbl in enumerate(labels):
                if lbl == name:
                    labels[i] = f"{name}_1"
                    break  # only fix the first bare occurrence

    for alloc, label in zip(multi_config.allocations, labels):
        alloc_cash = base.starting_cash * alloc.weight
        config = replace(
            base,
            strategy_name=alloc.strategy_name,
            strategy_params=alloc.params,
            starting_cash=alloc_cash,
        )

        logger.info(
            f"Running strategy '{label}' with weight={alloc.weight:.2%}, "
            f"cash=${alloc_cash:,.2f}"
        )

        try:
            if data_manager is not None:
                engine = BacktestEngine(config, data_manager=data_manager)
            else:
                engine = BacktestEngine(config)
            result = engine.run()
        except Exception as e:
            logger.error(f"Strategy '{label}' failed: {e}")
            continue

        per_strategy_results[label] = result

        equity = result.equity_series
        trades = result.trades
        metrics = compute_all_metrics(equity, trades)
        per_strategy_metrics[label] = metrics
        per_strategy_weights[label] = alloc.weight
        combined_trades.extend(trades)

    if not per_strategy_results:
        raise RuntimeError("All strategy allocations failed; no results to combine")

    # Build combined equity curve by summing aligned individual curves.
    # Add uninvested cash (weight remainder) as a flat line.
    equity_frames: dict[str, pd.Series] = {}
    for name, result in per_strategy_results.items():
        equity_frames[name] = result.equity_series

    combined_df = pd.DataFrame(equity_frames)
    # Forward-fill any missing dates across strategies, then backfill the start
    combined_df = combined_df.ffill().bfill()
    combined_equity = combined_df.sum(axis=1)

    # Add uninvested cash remainder
    invested_weight = sum(per_strategy_weights.values())
    cash_remainder = base.starting_cash * (1.0 - invested_weight)
    if cash_remainder > 0:
        combined_equity = combined_equity + cash_remainder

    combined_equity.name = "Equity"

    # Compute combined metrics on the summed equity curve
    combined_metrics = compute_all_metrics(combined_equity, combined_trades)

    # Attribution analysis
    attribution = compute_attribution(per_strategy_results, per_strategy_weights)

    return MultiStrategyResult(
        combined_equity_curve=combined_equity,
        combined_metrics=combined_metrics,
        per_strategy_results=per_strategy_results,
        per_strategy_metrics=per_strategy_metrics,
        per_strategy_weights=per_strategy_weights,
        combined_trades=combined_trades,
        attribution=attribution,
    )


def compute_attribution(
    per_strategy_results: dict[str, BacktestResult],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Compute each strategy's contribution to the combined portfolio return.

    For each strategy the absolute PnL is computed (final equity - starting cash).
    The contribution percentage is each strategy's PnL as a fraction of the
    total combined PnL.

    Args:
        per_strategy_results: BacktestResult keyed by strategy name.
        weights: Allocation weights keyed by strategy name.

    Returns:
        DataFrame with columns: strategy_name, weight, cagr, sharpe,
        pnl, contribution_pct.
    """
    rows: list[dict] = []
    total_pnl = 0.0

    # First pass: gather per-strategy PnL and metrics
    pnl_map: dict[str, float] = {}
    metrics_map: dict[str, dict] = {}
    for name, result in per_strategy_results.items():
        equity = result.equity_series
        trades = result.trades
        metrics = compute_all_metrics(equity, trades)
        metrics_map[name] = metrics
        strategy_pnl = equity.iloc[-1] - equity.iloc[0] if len(equity) >= 2 else 0.0
        pnl_map[name] = strategy_pnl
        total_pnl += strategy_pnl

    # Second pass: build attribution rows
    for name in per_strategy_results:
        pnl = pnl_map[name]
        metrics = metrics_map[name]
        if total_pnl != 0:
            contribution = (pnl / total_pnl) * 100.0
        else:
            # If total PnL is zero, distribute equally
            contribution = 100.0 / len(per_strategy_results)

        rows.append({
            "strategy_name": name,
            "weight": weights.get(name, 0.0),
            "cagr": metrics.get("cagr", 0.0),
            "sharpe": metrics.get("sharpe_ratio", 0.0),
            "pnl": pnl,
            "contribution_pct": contribution,
        })

    return pd.DataFrame(rows)


def print_multi_strategy_report(result: MultiStrategyResult) -> None:
    """Print a formatted multi-strategy portfolio summary to the console."""
    print("\n" + "=" * 70)
    print("MULTI-STRATEGY PORTFOLIO REPORT")
    print("=" * 70)

    # Combined metrics
    m = result.combined_metrics
    eq = result.combined_equity_curve
    print(f"\n--- Combined Portfolio ---")
    print(f"Final Equity:   ${eq.iloc[-1]:,.2f}")
    print(f"Total Return:   {m.get('total_return', 0):.2%}")
    print(f"CAGR:           {m.get('cagr', 0):.2%}")
    print(f"Sharpe Ratio:   {m.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio:  {m.get('sortino_ratio', 0):.2f}")
    print(f"Max Drawdown:   {m.get('max_drawdown', 0):.2%}")
    print(f"Total Trades:   {m.get('total_trades', 0)}")

    # Per-strategy breakdown
    print(f"\n--- Per-Strategy Breakdown ---")
    header = (
        f"{'Strategy':<20} {'Weight':>7} {'CAGR':>8} {'Sharpe':>8} "
        f"{'MaxDD':>8} {'Trades':>7} {'Final $':>12}"
    )
    print(header)
    print("-" * len(header))

    for name, metrics in result.per_strategy_metrics.items():
        w = result.per_strategy_weights.get(name, 0.0)
        final_eq = result.per_strategy_results[name].equity_series.iloc[-1]
        print(
            f"{name:<20} {w:>6.0%} {metrics.get('cagr', 0):>7.2%} "
            f"{metrics.get('sharpe_ratio', 0):>8.2f} "
            f"{metrics.get('max_drawdown', 0):>7.2%} "
            f"{metrics.get('total_trades', 0):>7} "
            f"${final_eq:>11,.2f}"
        )

    # Attribution
    attr = result.attribution
    if not attr.empty:
        print(f"\n--- Attribution Analysis ---")
        attr_header = f"{'Strategy':<20} {'Weight':>7} {'PnL':>12} {'Contrib %':>10}"
        print(attr_header)
        print("-" * len(attr_header))
        for _, row in attr.iterrows():
            print(
                f"{row['strategy_name']:<20} {row['weight']:>6.0%} "
                f"${row['pnl']:>11,.2f} {row['contribution_pct']:>9.1f}%"
            )

    print("=" * 70 + "\n")
