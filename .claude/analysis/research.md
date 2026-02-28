# Research Module Analysis

`src/backtester/research/`

## Module Purpose

Provides parameter optimization and walk-forward analysis tools for evaluating strategy robustness. The optimizer runs grid search over strategy parameter combinations, while walk-forward splits the date range into rolling in-sample/out-of-sample windows to detect overfitting. Both modules drive the backtest engine repeatedly and aggregate performance metrics.

## Key Classes/Functions

### `optimizer.py`

| Name | Description |
|---|---|
| `OptimizationResult` | Dataclass holding results table (DataFrame), best params, best metric value, and metric name. |
| `grid_search()` | Runs a full cartesian-product sweep of `param_grid`, executing a backtest per combo and ranking by `optimize_metric`. |
| `print_optimization_results()` | Formats and prints the top-N optimization results table to console. |

### `walk_forward.py`

| Name | Description |
|---|---|
| `walk_forward()` | Splits the config date range into rolling IS/OOS windows, optimizes params in-sample via `grid_search`, then tests on out-of-sample. Returns per-window results and a degradation ratio (OOS Sharpe / IS Sharpe). |
| `print_walk_forward_results()` | Formats and prints the walk-forward window table and degradation quality rating to console. |
| `_add_months()` | Helper to add N months to a `date`, clamping the day to the last valid day of the target month. |

## Critical Data Flows

1. **Grid Search:** `base_config` + `param_grid` -> `itertools.product` generates all combos -> each combo produces a `BacktestConfig` via `dataclasses.replace(base_config, strategy_params=merged)` -> `BacktestEngine(config).run()` returns equity series, trades, benchmark -> `compute_all_metrics()` produces metric dict -> all rows collected into a DataFrame -> best row selected by `optimize_metric`.
2. **Walk-Forward:** Date range split into IS/OOS windows (rolling or anchored) -> each IS window calls `grid_search()` to find best params -> best params applied to OOS window via a fresh `BacktestEngine` run -> OOS metrics collected per window -> aggregate averages and degradation ratio computed across all windows.
3. **Config mutation:** Both modules use `dataclasses.replace()` to create modified configs. The original `base_config` is never mutated. The `strategy_params` dict is merged (`{**base_config.strategy_params, **params}`) so grid params override base params while preserving any non-grid params.

## External Dependencies

### Internal (backtester modules)

| Import | Used By |
|---|---|
| `backtester.config.BacktestConfig` | Both modules — config dataclass for each engine run |
| `backtester.engine.BacktestEngine` | Both modules — executes backtests |
| `backtester.analytics.metrics.compute_all_metrics` | Both modules — computes performance metrics from equity/trades |
| `backtester.research.optimizer.grid_search` | `walk_forward.py` — delegates IS optimization |

### Third-Party

| Library | Usage |
|---|---|
| `pandas` | DataFrame for results table, metric aggregation |
| `itertools` | `product()` for cartesian param combos |
| `calendar` (stdlib) | `monthrange()` in `_add_months` for day clamping |

## "Do Not Touch" Warnings

1. **Param merge order:** `{**base_config.strategy_params, **params}` — grid params must override base params, not the other way around. Reversing this silently ignores the grid search.
2. **`replace()` immutability:** `base_config` must never be mutated directly. All per-run configs are created via `dataclasses.replace()`. Breaking this causes cross-contamination between optimization runs.
3. **IS/OOS boundary:** OOS starts at `is_end + timedelta(days=1)`. This one-day gap prevents any data leakage from the in-sample period into out-of-sample. Do not remove.
4. **`_add_months` clamping:** The day-clamping logic (e.g., Jan 31 + 1 month = Feb 28) prevents `ValueError` on invalid dates. The `calendar` import is intentionally inside the function body.
5. **Error tolerance in grid search:** Failed backtest runs produce rows with an `"error"` column instead of raising. Downstream code (best-param selection, display) filters on `notna()`. Changing error handling may break ranking.
6. **Walk-forward window advancement:** `is_start` advances by `oos_months` (not `is_months`) each iteration, ensuring OOS windows tile without overlap. Changing this breaks the non-overlapping OOS guarantee.
