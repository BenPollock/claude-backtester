"""CLI entry point using Click."""

import json
import logging
from datetime import date

import click

from backtester.config import BacktestConfig, RegimeFilter, StopConfig
from backtester.engine import BacktestEngine
from backtester.analytics.report import print_report, plot_results, export_activity_log_csv
from backtester.strategies.registry import discover_strategies, list_strategies

# Auto-discover all strategy modules so they register with the registry
discover_strategies()


# ---------------------------------------------------------------------------
# Shared option decorators — used by run, optimize, and walk-forward
# ---------------------------------------------------------------------------

_common_options = [
    click.option("--strategy", required=True, help="Strategy name (e.g. sma_crossover)"),
    click.option("--tickers", required=False, default=None, help="Comma-separated ticker symbols"),
    click.option("--market", type=click.Choice(["us", "ca", "us_ca"]), default="us_ca", help="Market scope when tickers omitted"),
    click.option("--universe", type=click.Choice(["index", "all"]), default="index", help="Universe breadth when tickers omitted"),
    click.option("--benchmark", required=True, help="Benchmark ticker (e.g. SPY)"),
    click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]), help="Start date (YYYY-MM-DD)"),
    click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]), help="End date (YYYY-MM-DD)"),
    click.option("--cash", default=10000.0, type=float, help="Starting cash (default: 10000)"),
    click.option("--max-positions", default=100, type=int, help="Max concurrent positions"),
    click.option("--max-alloc", default=0.10, type=float, help="Max allocation per position (0.10 = 10%)"),
    click.option("--fee", default=0.05, type=float, help="Fee amount: dollars for per_trade, basis points for percentage/composite_us"),
    click.option("--slippage-bps", default=10.0, type=float, help="Slippage in basis points (for fixed model)"),
    click.option("--params", default="{}", help="Strategy params as JSON string"),
    click.option("--cache-dir", default="~/.backtester/cache", help="Data cache directory"),
    # Regime filter
    click.option("--regime-benchmark", default=None, help="Regime filter benchmark (e.g. SPY)"),
    click.option("--regime-fast", default=100, type=int, help="Regime filter fast SMA period"),
    click.option("--regime-slow", default=200, type=int, help="Regime filter slow SMA period"),
    click.option("--regime-condition", type=click.Choice(["fast_above_slow", "fast_below_slow"]), default="fast_above_slow", help="Regime filter condition"),
    # Stop config
    click.option("--stop-loss", default=None, type=float, help="Stop-loss as fraction (e.g. 0.05 = 5%)"),
    click.option("--take-profit", default=None, type=float, help="Take-profit as fraction (e.g. 0.20 = 20%)"),
    click.option("--trailing-stop", default=None, type=float, help="Trailing stop as fraction (e.g. 0.08 = 8%)"),
    click.option("--stop-loss-atr", default=None, type=float, help="Stop-loss in ATR multiples (e.g. 2.0)"),
    click.option("--take-profit-atr", default=None, type=float, help="Take-profit in ATR multiples (e.g. 3.0)"),
    # Position sizing
    click.option("--position-sizing", type=click.Choice(["fixed_fractional", "atr", "vol_parity"]), default="fixed_fractional", help="Position sizing model"),
    click.option("--risk-pct", default=0.01, type=float, help="Risk per trade for ATR sizer (e.g. 0.01 = 1%)"),
    click.option("--atr-multiple", default=2.0, type=float, help="ATR multiple for stop distance in ATR sizer"),
    # Short selling
    click.option("--allow-short", is_flag=True, default=False, help="Enable short selling"),
    click.option("--short-borrow-rate", default=0.02, type=float, help="Annualized short borrow rate (default: 0.02)"),
    click.option("--margin-requirement", default=1.5, type=float, help="Initial margin requirement (default: 1.5 = 150%)"),
    # Slippage model
    click.option("--slippage-model", type=click.Choice(["fixed", "volume"]), default="fixed", help="Slippage model"),
    click.option("--slippage-impact", default=0.1, type=float, help="Impact factor for volume slippage model"),
    # Fee model
    click.option("--fee-model", type=click.Choice(["per_trade", "percentage", "composite_us"]), default="per_trade", help="Fee model"),
    # Volatility parity sizing
    click.option("--vol-target", default=0.10, type=float, help="Target volatility for vol_parity sizer (default: 0.10)"),
    click.option("--vol-lookback", default=20, type=int, help="Lookback window for vol_parity sizer (default: 20)"),
]


def _add_common_options(func):
    """Apply all common options to a Click command."""
    for option in reversed(_common_options):
        func = option(func)
    return func


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """claude-backtester: A modular stock backtesting engine."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@_add_common_options
@click.option("--export-log", default=None, type=click.Path(), help="Export activity log to CSV file")
@click.option("--monte-carlo-runs", default=1000, type=int, help="Number of Monte Carlo simulations (default: 1000)")
@click.option("--tearsheet", default=None, type=click.Path(), help="Generate HTML tearsheet at PATH")
@click.option("--report-regime", is_flag=True, default=False, help="Print regime performance breakdown")
@click.option("--report-signal-decay", is_flag=True, default=False, help="Print signal decay analysis")
@click.option("--report-correlation", is_flag=True, default=False, help="Print correlation matrix")
@click.option("--report-concentration", is_flag=True, default=False, help="Print portfolio concentration (HHI)")
def run(strategy, tickers, market, universe, benchmark, start, end, cash, max_positions,
        max_alloc, fee, slippage_bps, params, cache_dir, regime_benchmark, regime_fast,
        regime_slow, regime_condition, export_log, position_sizing, risk_pct, atr_multiple,
        stop_loss, take_profit, trailing_stop, stop_loss_atr, take_profit_atr,
        allow_short, short_borrow_rate, margin_requirement,
        slippage_model, slippage_impact, fee_model, vol_target, vol_lookback,
        monte_carlo_runs, tearsheet, report_regime, report_signal_decay, report_correlation,
        report_concentration):
    """Run a backtest."""
    config = _build_config(
        strategy=strategy, tickers=tickers, market=market, universe=universe,
        benchmark=benchmark, start=start, end=end, cash=cash,
        max_positions=max_positions, max_alloc=max_alloc, fee=fee,
        slippage_bps=slippage_bps, params=params, cache_dir=cache_dir,
        regime_benchmark=regime_benchmark, regime_fast=regime_fast,
        regime_slow=regime_slow, regime_condition=regime_condition,
        position_sizing=position_sizing,
        risk_pct=risk_pct, atr_multiple=atr_multiple,
        stop_loss=stop_loss, take_profit=take_profit,
        trailing_stop=trailing_stop, stop_loss_atr=stop_loss_atr,
        take_profit_atr=take_profit_atr,
        allow_short=allow_short, short_borrow_rate=short_borrow_rate,
        margin_requirement=margin_requirement,
        slippage_model=slippage_model, slippage_impact=slippage_impact,
        fee_model=fee_model, vol_target=vol_target, vol_lookback=vol_lookback,
        monte_carlo_runs=monte_carlo_runs,
    )

    engine = BacktestEngine(config)
    result = engine.run()
    print_report(result)

    if export_log:
        export_activity_log_csv(result, export_log)
        click.echo(f"Activity log exported to {export_log}")

    # Analytics output flags
    if tearsheet:
        from backtester.analytics.tearsheet import generate_tearsheet
        path = generate_tearsheet(result, output_path=tearsheet)
        click.echo(f"Tearsheet written to {path}")

    if report_regime:
        from backtester.analytics.regime import regime_summary
        if result.benchmark_prices is not None:
            summary = regime_summary(
                equity_curve=result.equity_series,
                benchmark_prices=result.benchmark_prices,
            )
            click.echo("\n=== Regime Performance ===")
            click.echo("\nMarket Regime:")
            click.echo(summary["market_regime_perf"].to_string())
            click.echo("\nVolatility Regime:")
            click.echo(summary["vol_regime_perf"].to_string())
        else:
            click.echo("Regime analysis requires benchmark data.")

    if report_signal_decay:
        from backtester.analytics.signal_decay import signal_decay_summary
        if result.universe_data and result.trades:
            summary = signal_decay_summary(
                trades=result.trades,
                price_data=result.universe_data,
            )
            click.echo("\n=== Signal Decay Analysis ===")
            click.echo(f"Total signals: {summary['total_signals']}")
            click.echo(f"\nAverage decay:\n{summary['avg_decay'].to_string()}")
            optimal = summary["optimal_holding"]
            click.echo(f"\nOptimal holding period: {optimal['optimal_days']} days "
                       f"(peak return: {optimal['peak_return']:.4f})")
        else:
            click.echo("Signal decay analysis requires trades and universe data.")

    if report_correlation:
        from backtester.analytics.correlation import compute_correlation_matrix
        if result.universe_data and len(result.universe_data) > 1:
            corr = compute_correlation_matrix(result.universe_data)
            click.echo("\n=== Correlation Matrix ===")
            click.echo(corr.to_string())
        else:
            click.echo("Correlation analysis requires at least 2 tickers.")

    if report_concentration:
        from backtester.analytics.correlation import compute_portfolio_concentration
        positions = {
            sym: pos.market_value
            for sym, pos in result.portfolio.positions.items()
        }
        if positions:
            conc = compute_portfolio_concentration(positions)
            click.echo("\n=== Portfolio Concentration ===")
            click.echo(f"HHI:              {conc['hhi']:.3f}")
            click.echo(f"Effective N:      {conc['effective_n']:.1f}")
            click.echo(f"Max Weight:       {conc['max_weight']:.2%} ({conc['max_weight_ticker']})")
            click.echo(f"\nPosition Weights:")
            for ticker, weight in sorted(conc['weights'].items(), key=lambda x: -x[1]):
                click.echo(f"  {ticker:<8} {weight:.2%}")
        else:
            click.echo("Concentration analysis requires open positions at end of backtest.")

    plot_results(result)


@cli.command("list-strategies")
def list_strats():
    """List available strategies."""
    strategies = list_strategies()
    if not strategies:
        click.echo("No strategies registered.")
        return
    click.echo("Available strategies:")
    for name in strategies:
        click.echo(f"  - {name}")


def _build_config(strategy, tickers, market, universe, benchmark, start, end,
                   cash, max_positions, max_alloc, fee, slippage_bps, params,
                   cache_dir, regime_benchmark=None, regime_fast=100,
                   regime_slow=200, regime_condition="fast_above_slow",
                   position_sizing="fixed_fractional",
                   risk_pct=0.01, atr_multiple=2.0, stop_loss=None,
                   take_profit=None, trailing_stop=None, stop_loss_atr=None,
                   take_profit_atr=None, allow_short=False,
                   short_borrow_rate=0.02, margin_requirement=1.5,
                   slippage_model="fixed", slippage_impact=0.1,
                   fee_model="per_trade", vol_target=0.10, vol_lookback=20,
                   monte_carlo_runs=1000, **kwargs) -> BacktestConfig:
    """Shared config builder for run/optimize/walk-forward commands."""
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        from backtester.data.universe import UniverseProvider
        provider = UniverseProvider()
        ticker_list = provider.get_tickers(market=market, universe=universe)
    strategy_params = json.loads(params)

    regime_filter = None
    if regime_benchmark:
        regime_filter = RegimeFilter(
            benchmark=regime_benchmark,
            indicator="sma",
            fast_period=regime_fast,
            slow_period=regime_slow,
            condition=regime_condition,
        )

    stop_config = None
    if any(v is not None for v in [stop_loss, take_profit, trailing_stop,
                                    stop_loss_atr, take_profit_atr]):
        stop_config = StopConfig(
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            trailing_stop_pct=trailing_stop,
            stop_loss_atr=stop_loss_atr,
            take_profit_atr=take_profit_atr,
        )

    return BacktestConfig(
        strategy_name=strategy,
        tickers=ticker_list,
        benchmark=benchmark.upper(),
        start_date=start.date(),
        end_date=end.date(),
        starting_cash=cash,
        max_positions=max_positions,
        max_alloc_pct=max_alloc,
        fee_per_trade=fee,
        slippage_bps=slippage_bps,
        slippage_model=slippage_model,
        data_cache_dir=cache_dir,
        strategy_params=strategy_params,
        regime_filter=regime_filter,
        stop_config=stop_config,
        position_sizing=position_sizing,
        sizing_risk_pct=risk_pct,
        sizing_atr_multiple=atr_multiple,
        allow_short=allow_short,
        short_borrow_rate=short_borrow_rate,
        margin_requirement=margin_requirement,
        fee_model=fee_model,
        sizing_target_vol=vol_target,
        sizing_vol_lookback=vol_lookback,
        slippage_impact_factor=slippage_impact,
        monte_carlo_runs=monte_carlo_runs,
    )


@cli.command()
@_add_common_options
@click.option("--grid", required=True, help='Param grid as JSON: {"sma_fast":[50,100],"sma_slow":[200,300]}')
@click.option("--metric", default="sharpe_ratio", help="Metric to optimize (default: sharpe_ratio)")
def optimize(strategy, tickers, benchmark, start, end, cash, max_positions,
             max_alloc, fee, slippage_bps, params, cache_dir, grid, metric,
             market, universe, regime_benchmark, regime_fast, regime_slow,
             regime_condition, stop_loss, take_profit, trailing_stop,
             stop_loss_atr, take_profit_atr, position_sizing, risk_pct,
             atr_multiple, allow_short, short_borrow_rate, margin_requirement,
             slippage_model, slippage_impact, fee_model, vol_target, vol_lookback):
    """Run parameter grid search optimization."""
    from backtester.research.optimizer import grid_search, print_optimization_results

    base_config = _build_config(
        strategy=strategy, tickers=tickers, market=market, universe=universe,
        benchmark=benchmark, start=start, end=end, cash=cash,
        max_positions=max_positions, max_alloc=max_alloc, fee=fee,
        slippage_bps=slippage_bps, params=params, cache_dir=cache_dir,
        regime_benchmark=regime_benchmark, regime_fast=regime_fast,
        regime_slow=regime_slow, regime_condition=regime_condition,
        position_sizing=position_sizing,
        risk_pct=risk_pct, atr_multiple=atr_multiple,
        stop_loss=stop_loss, take_profit=take_profit,
        trailing_stop=trailing_stop, stop_loss_atr=stop_loss_atr,
        take_profit_atr=take_profit_atr,
        allow_short=allow_short, short_borrow_rate=short_borrow_rate,
        margin_requirement=margin_requirement,
        slippage_model=slippage_model, slippage_impact=slippage_impact,
        fee_model=fee_model, vol_target=vol_target, vol_lookback=vol_lookback,
    )
    param_grid = json.loads(grid)
    result = grid_search(base_config, param_grid, optimize_metric=metric)
    print_optimization_results(result)


@cli.command("walk-forward")
@_add_common_options
@click.option("--grid", required=True, help='Param grid as JSON')
@click.option("--is-months", default=12, type=int, help="In-sample window months")
@click.option("--oos-months", default=3, type=int, help="Out-of-sample window months")
@click.option("--anchored", is_flag=True, help="Use anchored (expanding) IS window")
@click.option("--metric", default="sharpe_ratio", help="Metric to optimize")
def walk_forward_cmd(strategy, tickers, benchmark, start, end, cash, max_positions,
                     max_alloc, fee, slippage_bps, params, cache_dir, grid,
                     is_months, oos_months, anchored, metric, market, universe,
                     regime_benchmark, regime_fast, regime_slow, regime_condition,
                     stop_loss, take_profit, trailing_stop, stop_loss_atr,
                     take_profit_atr, position_sizing, risk_pct, atr_multiple,
                     allow_short, short_borrow_rate, margin_requirement,
                     slippage_model, slippage_impact, fee_model,
                     vol_target, vol_lookback):
    """Run walk-forward analysis with rolling optimization windows."""
    from backtester.research.walk_forward import walk_forward, print_walk_forward_results

    base_config = _build_config(
        strategy=strategy, tickers=tickers, market=market, universe=universe,
        benchmark=benchmark, start=start, end=end, cash=cash,
        max_positions=max_positions, max_alloc=max_alloc, fee=fee,
        slippage_bps=slippage_bps, params=params, cache_dir=cache_dir,
        regime_benchmark=regime_benchmark, regime_fast=regime_fast,
        regime_slow=regime_slow, regime_condition=regime_condition,
        position_sizing=position_sizing,
        risk_pct=risk_pct, atr_multiple=atr_multiple,
        stop_loss=stop_loss, take_profit=take_profit,
        trailing_stop=trailing_stop, stop_loss_atr=stop_loss_atr,
        take_profit_atr=take_profit_atr,
        allow_short=allow_short, short_borrow_rate=short_borrow_rate,
        margin_requirement=margin_requirement,
        slippage_model=slippage_model, slippage_impact=slippage_impact,
        fee_model=fee_model, vol_target=vol_target, vol_lookback=vol_lookback,
    )
    param_grid = json.loads(grid)
    result = walk_forward(
        base_config, param_grid,
        is_months=is_months, oos_months=oos_months,
        anchored=anchored, optimize_metric=metric,
    )
    print_walk_forward_results(result)


if __name__ == "__main__":
    cli()
