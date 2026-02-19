"""CLI entry point using Click."""

import json
import logging
from datetime import date

import click

# Import to trigger strategy registration
import backtester.strategies.sma_crossover  # noqa: F401
from backtester.config import BacktestConfig, RegimeFilter
from backtester.engine import BacktestEngine
from backtester.analytics.report import print_report, plot_results, export_activity_log_csv
from backtester.strategies.registry import list_strategies


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
def run(strategy, tickers, market, universe, benchmark, start, end, cash, max_positions,
        max_alloc, fee, slippage_bps, params, cache_dir, regime_benchmark, regime_fast,
        regime_slow, export_log):
    """Run a backtest."""
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        from backtester.data.universe import UniverseProvider
        provider = UniverseProvider()
        ticker_list = provider.get_tickers(market=market, universe=universe)
        click.echo(f"Universe: {len(ticker_list)} tickers ({market}/{universe})")
    strategy_params = json.loads(params)

    regime_filter = None
    if regime_benchmark:
        regime_filter = RegimeFilter(
            benchmark=regime_benchmark,
            indicator="sma",
            fast_period=regime_fast,
            slow_period=regime_slow,
        )

    config = BacktestConfig(
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


if __name__ == "__main__":
    cli()
