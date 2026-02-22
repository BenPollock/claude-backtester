"""Walk-forward analysis: rolling or anchored optimization + out-of-sample testing."""

import logging
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from backtester.config import BacktestConfig
from backtester.engine import BacktestEngine
from backtester.analytics.metrics import compute_all_metrics
from backtester.research.optimizer import grid_search

logger = logging.getLogger(__name__)


def walk_forward(
    base_config: BacktestConfig,
    param_grid: dict[str, list],
    is_months: int = 12,
    oos_months: int = 3,
    anchored: bool = False,
    optimize_metric: str = "sharpe_ratio",
) -> dict:
    """Run walk-forward analysis.

    Splits the date range into rolling windows:
      - In-sample (IS): optimize params via grid search
      - Out-of-sample (OOS): test best params from IS

    Args:
        base_config: Base config with full date range
        param_grid: Parameter grid for optimization
        is_months: In-sample window in months
        oos_months: Out-of-sample window in months
        anchored: If True, IS start stays fixed (expanding window)
        optimize_metric: Metric to optimize in IS period

    Returns:
        Dict with 'windows' (list of per-window results), 'oos_metrics' (aggregate),
        and 'degradation_ratio' (OOS Sharpe / IS Sharpe).
    """
    start = base_config.start_date
    end = base_config.end_date

    windows = []
    window_num = 0

    # Walk through the date range
    is_start = start
    while True:
        if anchored:
            current_is_start = start
        else:
            current_is_start = is_start

        is_end = _add_months(is_start, is_months)
        oos_start = is_end + timedelta(days=1)
        oos_end = _add_months(oos_start, oos_months)

        if oos_end > end:
            oos_end = end
        if oos_start >= end:
            break

        window_num += 1
        logger.info(f"Window {window_num}: IS {current_is_start} to {is_end}, "
                     f"OOS {oos_start} to {oos_end}")

        # In-sample optimization
        is_config = replace(base_config,
                           start_date=current_is_start,
                           end_date=is_end)
        try:
            is_result = grid_search(is_config, param_grid,
                                    optimize_metric=optimize_metric)
            best_params = is_result.best_params
            is_sharpe = is_result.best_metric_value
        except Exception as e:
            logger.warning(f"  IS optimization failed: {e}")
            is_start = _add_months(is_start, oos_months)
            continue

        if not best_params:
            logger.warning(f"  No valid IS results, skipping window")
            is_start = _add_months(is_start, oos_months)
            continue

        # Out-of-sample test with best IS params
        oos_params = {**base_config.strategy_params, **best_params}
        oos_config = replace(base_config,
                            start_date=oos_start,
                            end_date=oos_end,
                            strategy_params=oos_params)
        try:
            engine = BacktestEngine(oos_config)
            oos_result = engine.run()
            oos_equity = oos_result.equity_series
            oos_trades = oos_result.trades
            oos_metrics = compute_all_metrics(oos_equity, oos_trades)
            oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
        except Exception as e:
            logger.warning(f"  OOS test failed: {e}")
            oos_metrics = {}
            oos_sharpe = 0.0

        windows.append({
            "window": window_num,
            "is_start": current_is_start,
            "is_end": is_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "best_params": best_params,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "oos_cagr": oos_metrics.get("cagr", 0.0),
            "oos_max_dd": oos_metrics.get("max_drawdown", 0.0),
            "oos_trades": oos_metrics.get("total_trades", 0),
            "oos_win_rate": oos_metrics.get("win_rate", 0.0),
        })

        is_start = _add_months(is_start, oos_months)

    # Aggregate OOS results
    if windows:
        avg_is = sum(w["is_sharpe"] for w in windows) / len(windows)
        avg_oos = sum(w["oos_sharpe"] for w in windows) / len(windows)
        degradation = avg_oos / avg_is if avg_is != 0 else 0.0
    else:
        avg_is = avg_oos = degradation = 0.0

    return {
        "windows": windows,
        "avg_is_sharpe": avg_is,
        "avg_oos_sharpe": avg_oos,
        "degradation_ratio": degradation,
        "num_windows": len(windows),
    }


def print_walk_forward_results(wf: dict) -> None:
    """Print walk-forward results to console."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 70)
    print(f"Windows: {wf['num_windows']}")
    print(f"Avg IS Sharpe:  {wf['avg_is_sharpe']:.3f}")
    print(f"Avg OOS Sharpe: {wf['avg_oos_sharpe']:.3f}")
    ratio = wf['degradation_ratio']
    quality = "GOOD" if ratio > 0.5 else "WEAK" if ratio > 0 else "POOR"
    print(f"Degradation:    {ratio:.2f} ({quality})")

    if wf["windows"]:
        print(f"\n{'Win':>4} {'IS Period':<25} {'OOS Period':<25} "
              f"{'IS Sharpe':>10} {'OOS Sharpe':>10} {'OOS CAGR':>9} {'Params'}")
        print("-" * 110)
        for w in wf["windows"]:
            params_str = str(w["best_params"])
            if len(params_str) > 30:
                params_str = params_str[:27] + "..."
            print(f"{w['window']:>4} "
                  f"{str(w['is_start'])} - {str(w['is_end']):<12} "
                  f"{str(w['oos_start'])} - {str(w['oos_end']):<12} "
                  f"{w['is_sharpe']:>10.3f} {w['oos_sharpe']:>10.3f} "
                  f"{w['oos_cagr']:>8.1%} {params_str}")

    print("=" * 70)


def _add_months(d: date, months: int) -> date:
    """Add months to a date, clamping to valid day."""
    month = d.month + months
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    # Clamp day
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)
