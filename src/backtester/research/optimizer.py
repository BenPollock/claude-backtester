"""Parameter optimization: grid search, Bayesian optimization, and parallel sweeps."""

import itertools
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import pandas as pd
from tqdm import tqdm

from backtester.config import BacktestConfig
from backtester.engine import BacktestEngine
from backtester.analytics.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from a parameter sweep."""
    results_table: pd.DataFrame  # rows = param combos, cols = metrics
    best_params: dict
    best_metric_value: float
    optimize_metric: str


def _run_single_combo(args):
    """Run a single backtest combo (top-level for pickling in multiprocessing)."""
    base_config, params, param_names = args
    merged = {**base_config.strategy_params, **params}
    config = replace(base_config, strategy_params=merged)
    try:
        engine = BacktestEngine(config)
        result = engine.run()
        equity = result.equity_series
        trades = result.trades
        bm = result.benchmark_series
        metrics = compute_all_metrics(equity, trades, benchmark_series=bm)
        return {**params, **metrics}
    except Exception as e:
        return {**params, "error": str(e)}


def grid_search(
    base_config: BacktestConfig,
    param_grid: dict[str, list],
    optimize_metric: str = "sharpe_ratio",
    higher_is_better: bool = True,
    workers: int = 1,
) -> OptimizationResult:
    """Run backtest for every combination of parameters in param_grid.

    Args:
        base_config: Base config (strategy_params will be overridden per combo)
        param_grid: e.g. {"sma_fast": [50, 100, 150], "sma_slow": [200, 300]}
        optimize_metric: Which metric to rank by (default: sharpe_ratio)
        higher_is_better: Sort direction for the optimize metric
        workers: Number of parallel workers (Gap 29). 1 = sequential.

    Returns:
        OptimizationResult with full results table and best params
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))
    total = len(combos)

    logger.info(f"Grid search: {total} combinations of {param_names}")

    if workers > 1:
        # Parallel execution (Gap 29)
        args_list = [
            (base_config, dict(zip(param_names, combo)), param_names)
            for combo in combos
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(tqdm(
                executor.map(_run_single_combo, args_list),
                total=total, desc="Grid search",
                disable=not sys.stderr.isatty(),
            ))
    else:
        # Sequential execution
        rows = []
        for i, combo in enumerate(tqdm(
            combos, desc="Grid search",
            disable=not sys.stderr.isatty(),
        )):
            params = dict(zip(param_names, combo))
            merged = {**base_config.strategy_params, **params}
            config = replace(base_config, strategy_params=merged)

            logger.info(f"  [{i+1}/{total}] {params}")
            try:
                engine = BacktestEngine(config)
                result = engine.run()
                equity = result.equity_series
                trades = result.trades
                bm = result.benchmark_series
                metrics = compute_all_metrics(equity, trades, benchmark_series=bm)
                row = {**params, **metrics}
            except Exception as e:
                logger.warning(f"  Failed: {e}")
                row = {**params, "error": str(e)}
            rows.append(row)

    df = pd.DataFrame(rows)

    # Find best
    if optimize_metric in df.columns:
        valid = df[df[optimize_metric].notna()].copy()
        if not valid.empty:
            if higher_is_better:
                best_idx = valid[optimize_metric].idxmax()
            else:
                best_idx = valid[optimize_metric].idxmin()
            best_row = valid.loc[best_idx]
            best_params = {k: best_row[k] for k in param_names}
            best_val = best_row[optimize_metric]
        else:
            best_params, best_val = {}, 0.0
    else:
        best_params, best_val = {}, 0.0

    return OptimizationResult(
        results_table=df,
        best_params=best_params,
        best_metric_value=best_val,
        optimize_metric=optimize_metric,
    )


def bayesian_optimize(
    base_config: BacktestConfig,
    param_bounds: dict[str, tuple],
    optimize_metric: str = "sharpe_ratio",
    n_calls: int = 50,
    higher_is_better: bool = True,
) -> OptimizationResult:
    """Bayesian optimization using scikit-optimize (Gap 27).

    Args:
        base_config: Base configuration.
        param_bounds: Dict of param_name -> (low, high) tuples.
            Integers are inferred from tuple types.
        optimize_metric: Metric to optimize.
        n_calls: Number of optimization calls.
        higher_is_better: If True, maximize the metric.

    Returns:
        OptimizationResult with best params found.
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real
    except ImportError:
        raise ImportError(
            "scikit-optimize is required for Bayesian optimization. "
            "Install it with: pip install 'claude-backtester[optimize]'"
        )

    param_names = list(param_bounds.keys())
    dimensions = []
    for name, (low, high) in param_bounds.items():
        if isinstance(low, int) and isinstance(high, int):
            dimensions.append(Integer(low, high, name=name))
        else:
            dimensions.append(Real(float(low), float(high), name=name))

    all_rows = []

    def objective(values):
        params = dict(zip(param_names, values))
        merged = {**base_config.strategy_params, **params}
        config = replace(base_config, strategy_params=merged)
        try:
            engine = BacktestEngine(config)
            result = engine.run()
            equity = result.equity_series
            trades = result.trades
            bm = result.benchmark_series
            metrics = compute_all_metrics(equity, trades, benchmark_series=bm)
            all_rows.append({**params, **metrics})
            val = metrics.get(optimize_metric, 0.0)
            return -val if higher_is_better else val
        except Exception as e:
            logger.warning(f"Bayesian opt failed for {params}: {e}")
            all_rows.append({**params, "error": str(e)})
            return 0.0  # neutral value for failed runs

    result = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42)

    df = pd.DataFrame(all_rows)
    best_params = dict(zip(param_names, result.x))
    best_val = -result.fun if higher_is_better else result.fun

    return OptimizationResult(
        results_table=df,
        best_params=best_params,
        best_metric_value=best_val,
        optimize_metric=optimize_metric,
    )


def print_optimization_results(opt: OptimizationResult, top_n: int = 20) -> None:
    """Print optimization results table to console."""
    df = opt.results_table
    metric = opt.optimize_metric

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Optimize for: {metric}")
    print(f"Total combinations: {len(df)}")

    if "error" in df.columns:
        errors = df["error"].notna().sum()
        if errors:
            print(f"Failed runs: {errors}")

    # Sort by metric
    if metric in df.columns:
        show = df[df.get("error", pd.Series(dtype=str)).isna()].copy()
        show = show.sort_values(metric, ascending=False).head(top_n)
    else:
        show = df.head(top_n)

    # Pick display columns: param columns + key metrics
    key_metrics = ["sharpe_ratio", "cagr", "max_drawdown", "total_trades",
                   "win_rate", "profit_factor", "calmar_ratio"]
    param_cols = [c for c in show.columns if c not in key_metrics
                  and c != "error" and c not in [
                      "total_return", "sortino_ratio", "max_drawdown_duration_days",
                      "trade_expectancy", "exposure_time", "avg_win", "avg_loss",
                      "payoff_ratio", "avg_days", "median_days", "avg_days_winners",
                      "avg_days_losers", "max_consecutive_wins", "max_consecutive_losses",
                      "alpha", "beta", "information_ratio", "tracking_error",
                      "up_capture", "down_capture", "var_95", "cvar_95",
                      "omega_ratio", "treynor_ratio",
                  ]]
    display_cols = param_cols + [m for m in key_metrics if m in show.columns]
    show_df = show[display_cols].copy()

    # Format numeric columns
    for col in show_df.columns:
        if col in param_cols:
            continue
        if col in ("cagr", "max_drawdown", "win_rate"):
            show_df[col] = show_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        elif col == "total_trades":
            show_df[col] = show_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
        else:
            show_df[col] = show_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    print(f"\nTop {min(top_n, len(show_df))} results:")
    print(show_df.to_string(index=False))

    print(f"\nBest params: {opt.best_params}")
    print(f"Best {metric}: {opt.best_metric_value:.4f}")
    print("=" * 70)
