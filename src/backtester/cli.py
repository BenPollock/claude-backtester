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
@click.option("--strategy", required=True, help="Strategy name (e.g. sma_crossover)")
@click.option("--tickers", required=False, default=None, help="Comma-separated ticker symbols")
@click.option("--market", type=click.Choice(["us", "ca", "us_ca"]), default="us_ca", help="Market scope when tickers omitted")
@click.option("--universe", type=click.Choice(["index", "all"]), default="index", help="Universe breadth when tickers omitted")
@click.option("--benchmark", required=True, help="Benchmark ticker (e.g. SPY)")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]), help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]), help="End date (YYYY-MM-DD)")
@click.option("--cash", default=10000.0, type=float, help="Starting cash (default: 10000)")
@click.option("--max-positions", default=100, type=int, help="Max concurrent positions")
@click.option("--max-alloc", default=0.10, type=float, help="Max allocation per position (0.10 = 10%)")
@click.option("--fee", default=0.05, type=float, help="Fee per trade in dollars")
@click.option("--slippage-bps", default=10.0, type=float, help="Slippage in basis points")
@click.option("--params", default="{}", help="Strategy params as JSON string")
@click.option("--cache-dir", default="~/.backtester/cache", help="Data cache directory")
@click.option("--regime-benchmark", default=None, help="Regime filter benchmark (e.g. SPY)")
@click.option("--regime-fast", default=100, type=int, help="Regime filter fast SMA period")
@click.option("--regime-slow", default=200, type=int, help="Regime filter slow SMA period")
@click.option("--export-log", default=None, type=click.Path(), help="Export activity log to CSV file")
@click.option("--position-sizing", type=click.Choice(["fixed_fractional", "atr", "vol_parity"]), default="fixed_fractional", help="Position sizing model")
@click.option("--risk-pct", default=0.01, type=float, help="Risk per trade for ATR sizer (e.g. 0.01 = 1%)")
@click.option("--atr-multiple", default=2.0, type=float, help="ATR multiple for stop distance in ATR sizer")
@click.option("--stop-loss", default=None, type=float, help="Stop-loss as fraction (e.g. 0.05 = 5%)")
@click.option("--take-profit", default=None, type=float, help="Take-profit as fraction (e.g. 0.20 = 20%)")
@click.option("--trailing-stop", default=None, type=float, help="Trailing stop as fraction (e.g. 0.08 = 8%)")
@click.option("--stop-loss-atr", default=None, type=float, help="Stop-loss in ATR multiples (e.g. 2.0)")
@click.option("--take-profit-atr", default=None, type=float, help="Take-profit in ATR multiples (e.g. 3.0)")
def run(strategy, tickers, market, universe, benchmark, start, end, cash, max_positions,
        max_alloc, fee, slippage_bps, params, cache_dir, regime_benchmark, regime_fast,
        regime_slow, export_log, position_sizing, risk_pct, atr_multiple,
        stop_loss, take_profit, trailing_stop, stop_loss_atr, take_profit_atr):
    """Run a backtest."""
    config = _build_config(
        strategy=strategy, tickers=tickers, market=market, universe=universe,
        benchmark=benchmark, start=start, end=end, cash=cash,
        max_positions=max_positions, max_alloc=max_alloc, fee=fee,
        slippage_bps=slippage_bps, params=params, cache_dir=cache_dir,
        regime_benchmark=regime_benchmark, regime_fast=regime_fast,
        regime_slow=regime_slow, position_sizing=position_sizing,
        risk_pct=risk_pct, atr_multiple=atr_multiple,
        stop_loss=stop_loss, take_profit=take_profit,
        trailing_stop=trailing_stop, stop_loss_atr=stop_loss_atr,
        take_profit_atr=take_profit_atr,
    )

    engine = BacktestEngine(config)
    result = engine.run()
    print_report(result)
    if export_log:
        export_activity_log_csv(result, export_log)
        click.echo(f"Activity log exported to {export_log}")
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
                   regime_slow=200, position_sizing="fixed_fractional",
                   risk_pct=0.01, atr_multiple=2.0, stop_loss=None,
                   take_profit=None, trailing_stop=None, stop_loss_atr=None,
                   take_profit_atr=None, **kwargs) -> BacktestConfig:
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
        data_cache_dir=cache_dir,
        strategy_params=strategy_params,
        regime_filter=regime_filter,
        stop_config=stop_config,
        position_sizing=position_sizing,
        sizing_risk_pct=risk_pct,
        sizing_atr_multiple=atr_multiple,
    )


@cli.command()
@click.option("--strategy", required=True, help="Strategy name")
@click.option("--tickers", required=True, help="Comma-separated ticker symbols")
@click.option("--benchmark", required=True, help="Benchmark ticker")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--cash", default=10000.0, type=float)
@click.option("--max-positions", default=100, type=int)
@click.option("--max-alloc", default=0.10, type=float)
@click.option("--fee", default=0.05, type=float)
@click.option("--slippage-bps", default=10.0, type=float)
@click.option("--params", default="{}", help="Base strategy params JSON")
@click.option("--cache-dir", default="~/.backtester/cache")
@click.option("--grid", required=True, help='Param grid as JSON: {"sma_fast":[50,100],"sma_slow":[200,300]}')
@click.option("--metric", default="sharpe_ratio", help="Metric to optimize (default: sharpe_ratio)")
@click.option("--market", default="us_ca")
@click.option("--universe", default="index")
def optimize(strategy, tickers, benchmark, start, end, cash, max_positions,
             max_alloc, fee, slippage_bps, params, cache_dir, grid, metric,
             market, universe):
    """Run parameter grid search optimization."""
    from backtester.research.optimizer import grid_search, print_optimization_results

    base_config = _build_config(
        strategy, tickers, market, universe, benchmark, start, end,
        cash, max_positions, max_alloc, fee, slippage_bps, params, cache_dir,
    )
    param_grid = json.loads(grid)
    result = grid_search(base_config, param_grid, optimize_metric=metric)
    print_optimization_results(result)


@cli.command("walk-forward")
@click.option("--strategy", required=True, help="Strategy name")
@click.option("--tickers", required=True, help="Comma-separated ticker symbols")
@click.option("--benchmark", required=True, help="Benchmark ticker")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--cash", default=10000.0, type=float)
@click.option("--max-positions", default=100, type=int)
@click.option("--max-alloc", default=0.10, type=float)
@click.option("--fee", default=0.05, type=float)
@click.option("--slippage-bps", default=10.0, type=float)
@click.option("--params", default="{}", help="Base strategy params JSON")
@click.option("--cache-dir", default="~/.backtester/cache")
@click.option("--grid", required=True, help='Param grid as JSON')
@click.option("--is-months", default=12, type=int, help="In-sample window months")
@click.option("--oos-months", default=3, type=int, help="Out-of-sample window months")
@click.option("--anchored", is_flag=True, help="Use anchored (expanding) IS window")
@click.option("--metric", default="sharpe_ratio", help="Metric to optimize")
@click.option("--market", default="us_ca")
@click.option("--universe", default="index")
def walk_forward_cmd(strategy, tickers, benchmark, start, end, cash, max_positions,
                     max_alloc, fee, slippage_bps, params, cache_dir, grid,
                     is_months, oos_months, anchored, metric, market, universe):
    """Run walk-forward analysis with rolling optimization windows."""
    from backtester.research.walk_forward import walk_forward, print_walk_forward_results

    base_config = _build_config(
        strategy, tickers, market, universe, benchmark, start, end,
        cash, max_positions, max_alloc, fee, slippage_bps, params, cache_dir,
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
