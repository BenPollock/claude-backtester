"""Stress testing scenarios (Gap 38)."""

import logging
from dataclasses import replace
from datetime import date

from backtester.config import BacktestConfig
from backtester.analytics.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# Pre-defined stress scenarios: name -> (start_date, end_date)
STRESS_SCENARIOS = {
    "dot_com_crash": (date(2000, 3, 1), date(2002, 10, 31)),
    "gfc_2008": (date(2007, 10, 1), date(2009, 3, 31)),
    "flash_crash_2010": (date(2010, 4, 1), date(2010, 7, 31)),
    "taper_tantrum_2013": (date(2013, 5, 1), date(2013, 9, 30)),
    "china_devaluation_2015": (date(2015, 7, 1), date(2016, 2, 29)),
    "covid_crash": (date(2020, 2, 1), date(2020, 4, 30)),
    "rate_hike_2022": (date(2022, 1, 1), date(2022, 10, 31)),
}


def run_stress_test(
    base_config: BacktestConfig,
    scenarios: list[str] | None = None,
) -> list[dict]:
    """Run backtest across stress scenarios.

    Args:
        base_config: Base config to modify with scenario dates.
        scenarios: List of scenario names (default: all).

    Returns:
        List of dicts with scenario name and metrics.
    """
    from backtester.engine import BacktestEngine

    if scenarios is None:
        scenarios = list(STRESS_SCENARIOS.keys())

    results = []
    for name in scenarios:
        if name not in STRESS_SCENARIOS:
            logger.warning(f"Unknown stress scenario: {name}")
            continue

        start, end = STRESS_SCENARIOS[name]
        scenario_config = replace(base_config, start_date=start, end_date=end)

        try:
            engine = BacktestEngine(scenario_config)
            result = engine.run()
            equity = result.equity_series
            trades = result.trades
            metrics = compute_all_metrics(equity, trades)
            results.append({
                "scenario": name,
                "start": start,
                "end": end,
                **metrics,
            })
        except Exception as e:
            logger.warning(f"Stress test '{name}' failed: {e}")
            results.append({
                "scenario": name,
                "start": start,
                "end": end,
                "error": str(e),
            })

    return results


def print_stress_results(results: list[dict]) -> None:
    """Print stress test results to console."""
    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS")
    print("=" * 70)

    header = f"{'Scenario':<25} {'Period':<25} {'Return':>10} {'Max DD':>10} {'Sharpe':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['scenario']:<25} {str(r['start'])} - {str(r['end']):<10} {'ERROR':>10}")
            continue
        print(
            f"{r['scenario']:<25} "
            f"{str(r['start'])} - {str(r['end']):<10} "
            f"{r.get('total_return', 0):.2%} "
            f"{r.get('max_drawdown', 0):.2%} "
            f"{r.get('sharpe_ratio', 0):>8.2f}"
        )
    print("=" * 70)
