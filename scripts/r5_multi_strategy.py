#!/usr/bin/env python3
"""Round 5 Agent 2: Multi-Strategy Portfolio Construction & Analysis.

Runs each strategy component independently (to handle ticker mismatch),
then combines equity curves with configurable weights to evaluate
multi-strategy portfolios.
"""

import sys
import os
import json
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester.config import BacktestConfig
from backtester.engine import BacktestEngine
from backtester.analytics.metrics import compute_all_metrics
from backtester.strategies.registry import discover_strategies

# Discover all strategies
discover_strategies()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Strategy Definitions ─────────────────────────────────────────────

MOMENTUM_CONFIG = dict(
    strategy_name="momentum_rotation",
    tickers=["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"],
    benchmark="SPY",
    max_positions=5,
    max_alloc_pct=0.20,
    rebalance_schedule="monthly",
    strategy_params={
        "roc_period": 126,
        "top_n": 5,
        "abs_momentum": True,
        "sma_period": 200,
    },
)

RSI2_MR_CONFIG = dict(
    strategy_name="rule_based",
    tickers=["QQQ"],
    benchmark="SPY",
    max_positions=1,
    max_alloc_pct=1.0,
    rebalance_schedule="daily",
    strategy_params={
        "buy_when": [["rsi_2", "<", 10]],
        "sell_when": [["Close", ">", "sma_5"]],
        "indicators": {
            "rsi_2": {"fn": "rsi", "period": 2},
            "sma_5": {"fn": "sma", "period": 5},
        },
    },
)

HYBRID_LS_CONFIG = dict(
    strategy_name="rule_based",
    tickers=["QQQ"],
    benchmark="SPY",
    max_positions=1,
    max_alloc_pct=1.0,
    rebalance_schedule="daily",
    allow_short=True,
    strategy_params={
        "buy_when": [["rsi_2", "<", 10]],
        "sell_when": [["Close", ">", "sma_5"]],
        "short_when": [["rsi_2", ">", 95], ["Close", "<", "sma_200"]],
        "cover_when": [["rsi_2", "<", 50]],
        "indicators": {
            "rsi_2": {"fn": "rsi", "period": 2},
            "sma_5": {"fn": "sma", "period": 5},
            "sma_200": {"fn": "sma", "period": 200},
        },
    },
)

# Shared execution cost settings
COST_SETTINGS = dict(
    fee_model="composite_us",
    fee_per_trade=3.0,
    slippage_model="sqrt",
    slippage_impact_factor=0.1,
)


def build_config(strategy_cfg, start, end, cash):
    """Build a BacktestConfig from strategy-specific + shared settings."""
    params = {**COST_SETTINGS, **strategy_cfg}
    return BacktestConfig(
        start_date=start,
        end_date=end,
        starting_cash=cash,
        **params,
    )


def run_strategy(name, strategy_cfg, start, end, cash):
    """Run a single strategy and return (equity_series, metrics, result)."""
    config = build_config(strategy_cfg, start, end, cash)
    print(f"  Running {name} (cash=${cash:,.0f}, {start}..{end})...", end="", flush=True)
    try:
        engine = BacktestEngine(config)
        result = engine.run()
        equity = result.equity_series
        metrics = compute_all_metrics(equity, result.trades)
        print(f" done. Final=${equity.iloc[-1]:,.2f}, CAGR={metrics.get('cagr', 0):.2%}")
        return equity, metrics, result
    except Exception as e:
        print(f" FAILED: {e}")
        return None, None, None


def combine_equity_curves(components, total_cash):
    """Combine equity curves with weights into a portfolio.

    components: list of (name, weight, equity_series)
    total_cash: total starting cash

    Returns: combined pd.Series
    """
    frames = {}
    for name, weight, equity in components:
        if equity is not None:
            frames[name] = equity

    combined_df = pd.DataFrame(frames)
    combined_df = combined_df.ffill().bfill()
    combined_equity = combined_df.sum(axis=1)

    # Add uninvested cash
    invested_weight = sum(w for _, w, e in components if e is not None)
    cash_remainder = total_cash * (1.0 - invested_weight)
    if cash_remainder > 0:
        combined_equity += cash_remainder

    combined_equity.name = "Equity"
    return combined_equity


def compute_dsr(sharpe, n_tests, num_years):
    """Compute Deflated Sharpe Ratio using Bailey-de Prado formula.

    Approximate DSR using the normal CDF of a test statistic that
    adjusts for multiple testing.
    """
    from scipy import stats

    if sharpe <= 0 or n_tests <= 1:
        return 0.0

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    expected_max_sharpe = (
        stats.norm.ppf(1 - 1 / n_tests) * (1 - euler_mascheroni)
        + euler_mascheroni * stats.norm.ppf(1 - 1 / (n_tests * np.e))
    )

    # Standard error of Sharpe (assuming ~252 obs/year)
    n_obs = num_years * 252
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n_obs)

    # Test statistic
    test_stat = (sharpe - expected_max_sharpe) / se_sharpe

    # DSR = Prob(Sharpe > expected max | observed Sharpe)
    dsr = stats.norm.cdf(test_stat)
    return dsr


def run_portfolio(portfolio_name, allocations, start, end, total_cash):
    """Run a multi-strategy portfolio.

    allocations: list of (name, weight, strategy_cfg)
    Returns: (combined_equity, combined_metrics, per_strategy_data)
    """
    print(f"\n{'='*70}")
    print(f"PORTFOLIO: {portfolio_name}")
    print(f"Period: {start} to {end}, Cash: ${total_cash:,.0f}")
    print(f"{'='*70}")

    components = []
    per_strategy = {}

    for name, weight, cfg in allocations:
        alloc_cash = total_cash * weight
        equity, metrics, result = run_strategy(name, cfg, start, end, alloc_cash)
        components.append((name, weight, equity))
        per_strategy[name] = {
            "weight": weight,
            "equity": equity,
            "metrics": metrics,
            "result": result,
        }

    combined_equity = combine_equity_curves(components, total_cash)

    # Gather all trades
    all_trades = []
    for name, data in per_strategy.items():
        if data["result"] is not None:
            all_trades.extend(data["result"].trades)

    combined_metrics = compute_all_metrics(combined_equity, all_trades)

    # Print results
    print(f"\n--- Combined Portfolio: {portfolio_name} ---")
    print(f"Final Equity:   ${combined_equity.iloc[-1]:,.2f}")
    print(f"Total Return:   {combined_metrics.get('total_return', 0):.2%}")
    print(f"CAGR:           {combined_metrics.get('cagr', 0):.2%}")
    print(f"Sharpe Ratio:   {combined_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio:  {combined_metrics.get('sortino_ratio', 0):.2f}")
    print(f"Max Drawdown:   {combined_metrics.get('max_drawdown', 0):.2%}")
    print(f"Max DD Duration:{combined_metrics.get('max_dd_duration', 'N/A')}")

    print(f"\n--- Per-Strategy Breakdown ---")
    for name, data in per_strategy.items():
        if data["metrics"] is not None:
            m = data["metrics"]
            eq = data["equity"]
            print(
                f"  {name:<30} wt={data['weight']:.0%}  "
                f"CAGR={m.get('cagr', 0):.2%}  "
                f"Sharpe={m.get('sharpe_ratio', 0):.2f}  "
                f"MaxDD={m.get('max_drawdown', 0):.2%}  "
                f"Final=${eq.iloc[-1]:,.2f}"
            )

    return combined_equity, combined_metrics, per_strategy


def run_benchmark(start, end, cash):
    """Run SPY buy-and-hold as benchmark."""
    config = BacktestConfig(
        strategy_name="rule_based",
        tickers=["SPY"],
        benchmark="SPY",
        start_date=start,
        end_date=end,
        starting_cash=cash,
        max_positions=1,
        max_alloc_pct=1.0,
        fee_model="composite_us",
        fee_per_trade=3.0,
        slippage_model="sqrt",
        slippage_impact_factor=0.1,
        strategy_params={
            "buy_when": [["Close", ">", 0]],  # Always buy
            "sell_when": [],  # Never sell
            "indicators": {},
        },
    )
    engine = BacktestEngine(config)
    result = engine.run()
    equity = result.equity_series
    metrics = compute_all_metrics(equity, result.trades)
    return equity, metrics


# ── Portfolio Definitions ────────────────────────────────────────────

PORTFOLIO_A = {
    "name": "A: Momentum 60 / MR 40",
    "allocations": [
        ("Dual Momentum (60%)", 0.60, MOMENTUM_CONFIG),
        ("QQQ RSI-2 MR (40%)", 0.40, RSI2_MR_CONFIG),
    ],
}

PORTFOLIO_B = {
    "name": "B: Momentum 50 / MR 50",
    "allocations": [
        ("Dual Momentum (50%)", 0.50, MOMENTUM_CONFIG),
        ("QQQ RSI-2 MR (50%)", 0.50, RSI2_MR_CONFIG),
    ],
}

PORTFOLIO_C = {
    "name": "C: Momentum 50 / MR 30 / LS 20",
    "allocations": [
        ("Dual Momentum (50%)", 0.50, MOMENTUM_CONFIG),
        ("QQQ RSI-2 MR (30%)", 0.30, RSI2_MR_CONFIG),
        ("Hybrid L/S (20%)", 0.20, HYBRID_LS_CONFIG),
    ],
}

PORTFOLIO_D = {
    "name": "D: Momentum 40 / MR 40 / LS 20",
    "allocations": [
        ("Dual Momentum (40%)", 0.40, MOMENTUM_CONFIG),
        ("QQQ RSI-2 MR (40%)", 0.40, RSI2_MR_CONFIG),
        ("Hybrid L/S (20%)", 0.20, HYBRID_LS_CONFIG),
    ],
}


# ── Time Periods ─────────────────────────────────────────────────────

PERIODS = {
    "Full (2005-2024)": (date(2005, 1, 1), date(2024, 12, 31)),
    "Bear (2007-2012)": (date(2007, 1, 1), date(2012, 12, 31)),
    "Bull (2012-2024)": (date(2012, 1, 1), date(2024, 12, 31)),
}


def main():
    total_cash = 10000

    all_results = {}

    # Run all portfolios across all time periods
    for period_name, (start, end) in PERIODS.items():
        print(f"\n\n{'#'*70}")
        print(f"# TIME PERIOD: {period_name}")
        print(f"{'#'*70}")

        # Benchmark
        print(f"\n  Running SPY Buy-and-Hold benchmark...", end="", flush=True)
        bm_equity, bm_metrics = run_benchmark(start, end, total_cash)
        print(f" done. CAGR={bm_metrics.get('cagr', 0):.2%}, Sharpe={bm_metrics.get('sharpe_ratio', 0):.2f}")

        period_results = {"benchmark": bm_metrics}

        for portfolio in [PORTFOLIO_A, PORTFOLIO_B, PORTFOLIO_C, PORTFOLIO_D]:
            pname = portfolio["name"]
            eq, metrics, per_strat = run_portfolio(
                pname, portfolio["allocations"], start, end, total_cash
            )
            period_results[pname] = {
                "combined_metrics": metrics,
                "per_strategy": {
                    k: v["metrics"] for k, v in per_strat.items()
                },
            }

        all_results[period_name] = period_results

    # ── Summary Table ────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("GRAND SUMMARY: ALL PORTFOLIOS x ALL PERIODS")
    print("=" * 100)

    for period_name, period_data in all_results.items():
        bm = period_data["benchmark"]
        print(f"\n--- {period_name} ---")
        print(f"{'Portfolio':<40} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'vs SPY CAGR':>12} {'vs SPY Sharpe':>14}")
        print("-" * 100)
        print(
            f"{'SPY Buy & Hold':<40} "
            f"{bm.get('cagr', 0):>6.2%} "
            f"{bm.get('sharpe_ratio', 0):>7.2f} "
            f"{bm.get('sortino_ratio', 0):>8.2f} "
            f"{bm.get('max_drawdown', 0):>7.2%} "
            f"{'--':>12} {'--':>14}"
        )

        for pname, pdata in period_data.items():
            if pname == "benchmark":
                continue
            m = pdata["combined_metrics"]
            cagr_diff = m.get("cagr", 0) - bm.get("cagr", 0)
            sharpe_diff = m.get("sharpe_ratio", 0) - bm.get("sharpe_ratio", 0)
            print(
                f"{pname:<40} "
                f"{m.get('cagr', 0):>6.2%} "
                f"{m.get('sharpe_ratio', 0):>7.2f} "
                f"{m.get('sortino_ratio', 0):>8.2f} "
                f"{m.get('max_drawdown', 0):>7.2%} "
                f"{cagr_diff:>+11.2%} "
                f"{sharpe_diff:>+13.2f}"
            )

    # ── DSR Analysis ─────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("DEFLATED SHARPE RATIO ANALYSIS (n_tests=174)")
    print("=" * 70)

    n_tests = 174
    for period_name, period_data in all_results.items():
        start, end = PERIODS[period_name]
        num_years = (end - start).days / 365.25
        print(f"\n--- {period_name} (years={num_years:.1f}) ---")
        for pname, pdata in period_data.items():
            if pname == "benchmark":
                m = pdata
            else:
                m = pdata["combined_metrics"]
            sharpe = m.get("sharpe_ratio", 0)
            dsr = compute_dsr(sharpe, n_tests, num_years)
            label = pname if pname == "benchmark" else pname
            print(f"  {label:<40} Sharpe={sharpe:.2f}  DSR={dsr:.4f}  {'PASS' if dsr > 0.5 else 'FAIL'}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
