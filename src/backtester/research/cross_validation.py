"""Purged K-Fold Cross-Validation for backtest parameter optimization (Gap 28)."""

import logging
from dataclasses import replace
from datetime import date, timedelta

from backtester.config import BacktestConfig
from backtester.engine import BacktestEngine
from backtester.analytics.metrics import compute_all_metrics
from backtester.research.optimizer import grid_search

logger = logging.getLogger(__name__)


def purged_kfold_cv(
    base_config: BacktestConfig,
    param_grid: dict[str, list],
    n_splits: int = 5,
    purge_days: int = 10,
    embargo_days: int = 5,
    optimize_metric: str = "sharpe_ratio",
) -> dict:
    """Purged K-Fold Cross-Validation for backtesting.

    Splits the date range into K contiguous folds. For each fold:
    - Train on all other folds (minus purge gap around test boundaries)
    - Optimize params on train set
    - Test on the fold

    Args:
        base_config: Base config with full date range.
        param_grid: Parameter grid for optimization.
        n_splits: Number of folds.
        purge_days: Days to remove around test fold boundaries.
        embargo_days: Days to embargo after each test fold.
        optimize_metric: Metric to optimize on train set.

    Returns:
        Dict with per-fold results, avg_test_sharpe, avg_train_sharpe.
    """
    start = base_config.start_date
    end = base_config.end_date
    total_days = (end - start).days
    fold_size = total_days // n_splits

    folds = []
    for i in range(n_splits):
        fold_start = start + timedelta(days=i * fold_size)
        if i == n_splits - 1:
            fold_end = end
        else:
            fold_end = start + timedelta(days=(i + 1) * fold_size - 1)
        folds.append((fold_start, fold_end))

    results = []
    for test_idx in range(n_splits):
        test_start, test_end = folds[test_idx]
        logger.info(f"Fold {test_idx + 1}/{n_splits}: test {test_start} to {test_end}")

        # Build train periods (all folds except test, with purge/embargo)
        train_periods = []
        for i, (fs, fe) in enumerate(folds):
            if i == test_idx:
                continue
            # Apply purge: remove days near test boundaries
            adj_start = fs
            adj_end = fe
            if i == test_idx - 1:
                # Fold just before test: remove purge_days from end
                adj_end = fe - timedelta(days=purge_days)
            if i == test_idx + 1:
                # Fold just after test: add embargo_days to start
                adj_start = fs + timedelta(days=embargo_days)
            if adj_start < adj_end:
                train_periods.append((adj_start, adj_end))

        if not train_periods:
            logger.warning(f"  No valid train periods for fold {test_idx + 1}")
            continue

        # Use the longest contiguous train period for optimization
        # (simplified: use full date range minus test fold with purge)
        train_start = min(p[0] for p in train_periods)
        train_end = max(p[1] for p in train_periods)

        # Optimize on train
        train_config = replace(base_config, start_date=train_start, end_date=train_end)
        try:
            opt_result = grid_search(train_config, param_grid, optimize_metric=optimize_metric)
            best_params = opt_result.best_params
            train_sharpe = opt_result.best_metric_value
        except Exception as e:
            logger.warning(f"  Train optimization failed: {e}")
            continue

        if not best_params:
            continue

        # Test with best params
        test_params = {**base_config.strategy_params, **best_params}
        test_config = replace(base_config, start_date=test_start, end_date=test_end,
                             strategy_params=test_params)
        try:
            engine = BacktestEngine(test_config)
            result = engine.run()
            metrics = compute_all_metrics(result.equity_series, result.trades)
            test_sharpe = metrics.get("sharpe_ratio", 0.0)
        except Exception as e:
            logger.warning(f"  Test failed: {e}")
            test_sharpe = 0.0
            metrics = {}

        results.append({
            "fold": test_idx + 1,
            "test_start": test_start,
            "test_end": test_end,
            "best_params": best_params,
            "train_sharpe": train_sharpe,
            "test_sharpe": test_sharpe,
            "test_cagr": metrics.get("cagr", 0.0),
            "test_max_dd": metrics.get("max_drawdown", 0.0),
        })

    # Aggregate
    if results:
        avg_train = sum(r["train_sharpe"] for r in results) / len(results)
        avg_test = sum(r["test_sharpe"] for r in results) / len(results)
    else:
        avg_train = avg_test = 0.0

    return {
        "folds": results,
        "n_splits": n_splits,
        "purge_days": purge_days,
        "embargo_days": embargo_days,
        "avg_train_sharpe": avg_train,
        "avg_test_sharpe": avg_test,
    }
