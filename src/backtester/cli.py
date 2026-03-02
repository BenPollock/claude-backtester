"""CLI entry point using Click."""

import json
import logging
import sys
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
    click.option("--position-sizing", type=click.Choice(["fixed_fractional", "atr", "vol_parity", "kelly", "risk_parity"]), default="fixed_fractional", help="Position sizing model"),
    click.option("--risk-pct", default=0.01, type=float, help="Risk per trade for ATR sizer (e.g. 0.01 = 1%)"),
    click.option("--atr-multiple", default=2.0, type=float, help="ATR multiple for stop distance in ATR sizer"),
    # Short selling
    click.option("--allow-short", is_flag=True, default=False, help="Enable short selling"),
    click.option("--short-borrow-rate", default=0.02, type=float, help="Annualized short borrow rate (default: 0.02)"),
    click.option("--margin-requirement", default=1.5, type=float, help="Initial margin requirement (default: 1.5 = 150%)"),
    # Slippage model
    click.option("--slippage-model", type=click.Choice(["fixed", "volume", "sqrt"]), default="fixed", help="Slippage model"),
    click.option("--slippage-impact", default=0.1, type=float, help="Impact factor for volume/sqrt slippage model"),
    # Fee model
    click.option("--fee-model", type=click.Choice(["per_trade", "percentage", "composite_us"]), default="per_trade", help="Fee model"),
    # Volatility parity sizing
    click.option("--vol-target", default=0.10, type=float, help="Target volatility for vol_parity sizer (default: 0.10)"),
    click.option("--vol-lookback", default=20, type=int, help="Lookback window for vol_parity sizer (default: 20)"),
    # Gap 1: Historical universe
    click.option("--universe-file", default=None, type=click.Path(exists=True), help="CSV with date,symbol columns for survivorship-bias-free universe"),
    # Gap 2: Corporate actions
    click.option("--adjust-prices", type=click.Choice(["none", "splits", "splits_and_dividends"]), default="splits", help="Price adjustment mode"),
    # Gap 3: Partial fills
    click.option("--max-volume-pct", default=0.10, type=float, help="Max fraction of daily volume fillable (default: 0.10)"),
    click.option("--partial-fill-policy", type=click.Choice(["cancel", "requeue"]), default="cancel", help="Policy for unfilled remainder"),
    # Gap 5: Drawdown kill switch
    click.option("--max-drawdown", default=None, type=float, help="Max drawdown before halting (e.g. 0.10 = 10%)"),
    # Gap 8: Multi-source data
    click.option("--data-source", type=click.Choice(["yahoo", "csv", "parquet"]), default="yahoo", help="Data source type"),
    click.option("--data-path", default=None, type=click.Path(), help="Path for CSV/Parquet data files"),
    # Gap 9: Fundamental data
    click.option("--fundamental-data", default=None, type=click.Path(exists=True), help="Path to fundamental data CSV"),
    # Gap 14: Fill price model
    click.option("--fill-price", type=click.Choice(["open", "close", "vwap", "random"]), default="open", help="Fill price model for market orders"),
    # Gap 16: DRIP
    click.option("--drip", is_flag=True, default=False, help="Enable dividend reinvestment"),
    # Gap 17: Kelly fraction
    click.option("--kelly-fraction", default=0.5, type=float, help="Kelly fraction (0.5 = half-Kelly)"),
    # Gap 18: Sector exposure limits
    click.option("--max-sector-exposure", default=None, type=float, help="Max sector weight (e.g. 0.30 = 30%)"),
    click.option("--sector-map", default=None, type=click.Path(exists=True), help="CSV with symbol,sector columns"),
    # Gap 19: Gross/net exposure limits
    click.option("--max-gross-exposure", default=None, type=float, help="Max gross exposure as fraction of equity"),
    click.option("--max-net-exposure", default=None, type=float, help="Max net exposure as fraction of equity"),
    # Gap 20: Portfolio vol targeting
    click.option("--target-portfolio-vol", default=None, type=float, help="Target annualized portfolio volatility"),
    click.option("--portfolio-vol-lookback", default=60, type=int, help="Lookback for portfolio vol computation"),
    # Gap 30: Save results
    click.option("--save-results", default=None, type=click.Path(), help="Save results to directory"),
    # Gap 36: Lot method
    click.option("--lot-method", type=click.Choice(["fifo", "lifo", "highest_cost", "lowest_cost"]), default="fifo", help="Lot accounting method"),
    # Gap 45: Rebalance schedule
    click.option("--rebalance-schedule", type=click.Choice(["daily", "weekly", "monthly", "quarterly"]), default="daily", help="Signal generation frequency"),
    # Gap 49: TOML config file
    click.option("--config-file", default=None, type=click.Path(exists=True), help="Load config from TOML file"),
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
@click.option("--report-risk", is_flag=True, default=False, help="Print VaR/CVaR risk metrics")
@click.option("--report-mae-mfe", is_flag=True, default=False, help="Print per-trade MAE/MFE analysis")
@click.option("--report-tca", is_flag=True, default=False, help="Print transaction cost analysis")
@click.option("--trials", default=None, type=int, help="Number of trials for Deflated Sharpe Ratio")
@click.option("--permutation-test", default=None, type=int, help="Number of permutations for significance test")
def run(strategy, tickers, market, universe, benchmark, start, end, cash, max_positions,
        max_alloc, fee, slippage_bps, params, cache_dir, regime_benchmark, regime_fast,
        regime_slow, regime_condition, export_log, position_sizing, risk_pct, atr_multiple,
        stop_loss, take_profit, trailing_stop, stop_loss_atr, take_profit_atr,
        allow_short, short_borrow_rate, margin_requirement,
        slippage_model, slippage_impact, fee_model, vol_target, vol_lookback,
        monte_carlo_runs, tearsheet, report_regime, report_signal_decay, report_correlation,
        report_concentration, universe_file, adjust_prices, max_volume_pct,
        partial_fill_policy, max_drawdown, data_source, data_path, fundamental_data,
        fill_price, drip, kelly_fraction, max_sector_exposure, sector_map,
        max_gross_exposure, max_net_exposure, target_portfolio_vol,
        portfolio_vol_lookback, save_results, lot_method, rebalance_schedule,
        config_file, report_risk, report_mae_mfe, report_tca, trials,
        permutation_test):
    """Run a backtest."""
    # Gap 49: Load TOML config file if provided
    file_overrides = {}
    if config_file:
        file_overrides = _load_config_file(config_file)

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
        universe_file=universe_file, adjust_prices=adjust_prices,
        max_volume_pct=max_volume_pct, partial_fill_policy=partial_fill_policy,
        max_drawdown=max_drawdown, data_source=data_source, data_path=data_path,
        fundamental_data=fundamental_data, fill_price=fill_price,
        drip=drip, kelly_fraction=kelly_fraction,
        max_sector_exposure=max_sector_exposure, sector_map=sector_map,
        max_gross_exposure=max_gross_exposure, max_net_exposure=max_net_exposure,
        target_portfolio_vol=target_portfolio_vol,
        portfolio_vol_lookback=portfolio_vol_lookback,
        save_results=save_results, lot_method=lot_method,
        rebalance_schedule=rebalance_schedule,
        **file_overrides,
    )

    engine = BacktestEngine(config)
    result = engine.run()
    metrics = print_report(result)

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

    # Gap 6: VaR/CVaR report
    if report_risk:
        click.echo("\n=== Risk Metrics ===")
        click.echo(f"VaR (95%):     {metrics.get('var_95', 0):.4f}")
        click.echo(f"CVaR (95%):    {metrics.get('cvar_95', 0):.4f}")

    # Gap 21: MAE/MFE report
    if report_mae_mfe:
        from backtester.analytics.trade_analysis import compute_mae_mfe, mae_mfe_summary
        if result.trades and result.universe_data:
            mae_mfe = compute_mae_mfe(result.trades, result.universe_data)
            summary = mae_mfe_summary(mae_mfe)
            click.echo("\n=== MAE/MFE Analysis ===")
            click.echo(f"Avg MAE:       {summary['avg_mae']:.4f}")
            click.echo(f"Avg MFE:       {summary['avg_mfe']:.4f}")
            click.echo(f"Efficiency:    {summary['efficiency']:.2f}")
        else:
            click.echo("MAE/MFE analysis requires trades and universe data.")

    # Gap 22/23/24: TCA report
    if report_tca:
        from backtester.analytics.tca import compute_turnover, compute_cost_attribution, estimate_capacity
        equity = result.equity_series
        click.echo("\n=== Transaction Cost Analysis ===")
        turnover = compute_turnover(result.trades, equity)
        click.echo(f"Annual Turnover:  {turnover:.2f}x")
        costs = compute_cost_attribution(result.trades, equity)
        click.echo(f"Total Fees:       ${costs['total_fees']:,.2f}")
        click.echo(f"Cost % Equity:    {costs['cost_pct_equity']:.4f}")
        click.echo(f"Cost % Return:    {costs['cost_pct_return']:.4f}")
        if result.universe_data:
            cap = estimate_capacity(result.trades, result.universe_data)
            click.echo(f"Est. Capacity:    ${cap:,.0f}")

    # Gap 7: Overfitting metrics
    if trials:
        from backtester.analytics.overfitting import deflated_sharpe_ratio
        observed = metrics.get("sharpe_ratio", 0)
        n_ret = len(result.equity_series)
        click.echo(f"\n=== Overfitting Analysis ===")
        # Use variance 1.0 as placeholder (proper usage is in grid_search)
        dsr = deflated_sharpe_ratio(observed, trials, 1.0, n_ret)
        click.echo(f"Deflated Sharpe Ratio: {dsr:.4f} (trials={trials})")

    if permutation_test:
        from backtester.analytics.overfitting import permutation_test as perm_test
        result_perm = perm_test(result.equity_series, n_permutations=permutation_test)
        click.echo(f"\n=== Permutation Test ===")
        click.echo(f"Observed Sharpe: {result_perm['observed_sharpe']:.4f}")
        click.echo(f"P-value:         {result_perm['p_value']:.4f}")

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


def _load_config_file(path: str) -> dict:
    """Load configuration from TOML or YAML file (Gap 49)."""
    if path.endswith(".toml"):
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise click.ClickException("PyYAML required for YAML config files. pip install pyyaml")
    else:
        raise click.ClickException(f"Unsupported config file format: {path}")


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
                   monte_carlo_runs=1000,
                   universe_file=None, adjust_prices="splits",
                   max_volume_pct=0.10, partial_fill_policy="cancel",
                   max_drawdown=None, data_source="yahoo", data_path=None,
                   fundamental_data=None, fill_price="open",
                   drip=False, kelly_fraction=0.5,
                   max_sector_exposure=None, sector_map=None,
                   max_gross_exposure=None, max_net_exposure=None,
                   target_portfolio_vol=None, portfolio_vol_lookback=60,
                   save_results=None, lot_method="fifo",
                   rebalance_schedule="daily",
                   **kwargs) -> BacktestConfig:
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
        data_source=data_source,
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
        universe_file=universe_file,
        adjust_prices=adjust_prices,
        max_volume_pct=max_volume_pct,
        partial_fill_policy=partial_fill_policy,
        max_drawdown_pct=max_drawdown,
        data_path=data_path,
        fundamental_data_path=fundamental_data,
        fill_price_model=fill_price,
        drip=drip,
        kelly_fraction=kelly_fraction,
        max_sector_exposure=max_sector_exposure,
        sector_map_path=sector_map,
        max_gross_exposure=max_gross_exposure,
        max_net_exposure=max_net_exposure,
        target_portfolio_vol=target_portfolio_vol,
        portfolio_vol_lookback=portfolio_vol_lookback,
        save_results_path=save_results,
        lot_method=lot_method,
        rebalance_schedule=rebalance_schedule,
    )


@cli.command()
@_add_common_options
@click.option("--grid", required=True, help='Param grid as JSON: {"sma_fast":[50,100],"sma_slow":[200,300]}')
@click.option("--metric", default="sharpe_ratio", help="Metric to optimize (default: sharpe_ratio)")
@click.option("--workers", default=1, type=int, help="Parallel workers for grid search")
@click.option("--optimize-method", type=click.Choice(["grid", "bayesian"]), default="grid", help="Optimization method")
@click.option("--n-trials", default=50, type=int, help="Number of trials for Bayesian optimization")
def optimize(strategy, tickers, benchmark, start, end, cash, max_positions,
             max_alloc, fee, slippage_bps, params, cache_dir, grid, metric,
             market, universe, regime_benchmark, regime_fast, regime_slow,
             regime_condition, stop_loss, take_profit, trailing_stop,
             stop_loss_atr, take_profit_atr, position_sizing, risk_pct,
             atr_multiple, allow_short, short_borrow_rate, margin_requirement,
             slippage_model, slippage_impact, fee_model, vol_target, vol_lookback,
             universe_file, adjust_prices, max_volume_pct, partial_fill_policy,
             max_drawdown, data_source, data_path, fundamental_data,
             fill_price, drip, kelly_fraction, max_sector_exposure, sector_map,
             max_gross_exposure, max_net_exposure, target_portfolio_vol,
             portfolio_vol_lookback, save_results, lot_method, rebalance_schedule,
             config_file, workers, optimize_method, n_trials):
    """Run parameter optimization."""
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
        universe_file=universe_file, adjust_prices=adjust_prices,
        max_volume_pct=max_volume_pct, partial_fill_policy=partial_fill_policy,
        max_drawdown=max_drawdown, data_source=data_source, data_path=data_path,
        fundamental_data=fundamental_data, fill_price=fill_price,
        drip=drip, kelly_fraction=kelly_fraction,
        max_sector_exposure=max_sector_exposure, sector_map=sector_map,
        max_gross_exposure=max_gross_exposure, max_net_exposure=max_net_exposure,
        target_portfolio_vol=target_portfolio_vol,
        portfolio_vol_lookback=portfolio_vol_lookback,
        save_results=save_results, lot_method=lot_method,
        rebalance_schedule=rebalance_schedule,
    )
    param_grid = json.loads(grid)

    if optimize_method == "bayesian":
        from backtester.research.optimizer import bayesian_optimize
        # Convert grid to bounds
        param_bounds = {k: (min(v), max(v)) for k, v in param_grid.items()}
        result = bayesian_optimize(base_config, param_bounds,
                                   optimize_metric=metric, n_calls=n_trials)
    else:
        result = grid_search(base_config, param_grid, optimize_metric=metric,
                            workers=workers)
    print_optimization_results(result)


@cli.command("walk-forward")
@_add_common_options
@click.option("--grid", required=True, help='Param grid as JSON')
@click.option("--is-months", default=12, type=int, help="In-sample window months")
@click.option("--oos-months", default=3, type=int, help="Out-of-sample window months")
@click.option("--anchored", is_flag=True, help="Use anchored (expanding) IS window")
@click.option("--metric", default="sharpe_ratio", help="Metric to optimize")
@click.option("--cv-method", type=click.Choice(["walkforward", "purged_kfold"]), default="walkforward", help="Cross-validation method")
@click.option("--purge-days", default=10, type=int, help="Purge gap in days for purged K-fold")
@click.option("--embargo-days", default=5, type=int, help="Embargo gap in days for purged K-fold")
def walk_forward_cmd(strategy, tickers, benchmark, start, end, cash, max_positions,
                     max_alloc, fee, slippage_bps, params, cache_dir, grid,
                     is_months, oos_months, anchored, metric, market, universe,
                     regime_benchmark, regime_fast, regime_slow, regime_condition,
                     stop_loss, take_profit, trailing_stop, stop_loss_atr,
                     take_profit_atr, position_sizing, risk_pct, atr_multiple,
                     allow_short, short_borrow_rate, margin_requirement,
                     slippage_model, slippage_impact, fee_model,
                     vol_target, vol_lookback,
                     universe_file, adjust_prices, max_volume_pct, partial_fill_policy,
                     max_drawdown, data_source, data_path, fundamental_data,
                     fill_price, drip, kelly_fraction, max_sector_exposure, sector_map,
                     max_gross_exposure, max_net_exposure, target_portfolio_vol,
                     portfolio_vol_lookback, save_results, lot_method, rebalance_schedule,
                     config_file, cv_method, purge_days, embargo_days):
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
        universe_file=universe_file, adjust_prices=adjust_prices,
        max_volume_pct=max_volume_pct, partial_fill_policy=partial_fill_policy,
        max_drawdown=max_drawdown, data_source=data_source, data_path=data_path,
        fundamental_data=fundamental_data, fill_price=fill_price,
        drip=drip, kelly_fraction=kelly_fraction,
        max_sector_exposure=max_sector_exposure, sector_map=sector_map,
        max_gross_exposure=max_gross_exposure, max_net_exposure=max_net_exposure,
        target_portfolio_vol=target_portfolio_vol,
        portfolio_vol_lookback=portfolio_vol_lookback,
        save_results=save_results, lot_method=lot_method,
        rebalance_schedule=rebalance_schedule,
    )
    param_grid = json.loads(grid)

    if cv_method == "purged_kfold":
        from backtester.research.cross_validation import purged_kfold_cv
        result = purged_kfold_cv(
            base_config, param_grid,
            n_splits=is_months,  # reuse is_months as n_splits
            purge_days=purge_days,
            embargo_days=embargo_days,
            optimize_metric=metric,
        )
        click.echo(f"\nPurged K-Fold CV ({result['n_splits']} folds)")
        click.echo(f"Avg Train Sharpe: {result['avg_train_sharpe']:.3f}")
        click.echo(f"Avg Test Sharpe:  {result['avg_test_sharpe']:.3f}")
        for fold in result.get("folds", []):
            click.echo(f"  Fold {fold['fold']}: train={fold['train_sharpe']:.3f} test={fold['test_sharpe']:.3f}")
    else:
        result = walk_forward(
            base_config, param_grid,
            is_months=is_months, oos_months=oos_months,
            anchored=anchored, optimize_metric=metric,
        )
        print_walk_forward_results(result)


@cli.command("stress-test")
@_add_common_options
@click.option("--scenario", multiple=True, help="Stress scenario name (can repeat)")
def stress_test_cmd(strategy, tickers, benchmark, start, end, cash, max_positions,
                    max_alloc, fee, slippage_bps, params, cache_dir,
                    market, universe, regime_benchmark, regime_fast, regime_slow,
                    regime_condition, stop_loss, take_profit, trailing_stop,
                    stop_loss_atr, take_profit_atr, position_sizing, risk_pct,
                    atr_multiple, allow_short, short_borrow_rate, margin_requirement,
                    slippage_model, slippage_impact, fee_model, vol_target, vol_lookback,
                    universe_file, adjust_prices, max_volume_pct, partial_fill_policy,
                    max_drawdown, data_source, data_path, fundamental_data,
                    fill_price, drip, kelly_fraction, max_sector_exposure, sector_map,
                    max_gross_exposure, max_net_exposure, target_portfolio_vol,
                    portfolio_vol_lookback, save_results, lot_method, rebalance_schedule,
                    config_file, scenario):
    """Run stress test across historical scenarios (Gap 38)."""
    from backtester.analytics.stress import run_stress_test, print_stress_results

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
        universe_file=universe_file, adjust_prices=adjust_prices,
        max_volume_pct=max_volume_pct, partial_fill_policy=partial_fill_policy,
        max_drawdown=max_drawdown, data_source=data_source, data_path=data_path,
        fundamental_data=fundamental_data, fill_price=fill_price,
        drip=drip, kelly_fraction=kelly_fraction,
        max_sector_exposure=max_sector_exposure, sector_map=sector_map,
        max_gross_exposure=max_gross_exposure, max_net_exposure=max_net_exposure,
        target_portfolio_vol=target_portfolio_vol,
        portfolio_vol_lookback=portfolio_vol_lookback,
        save_results=save_results, lot_method=lot_method,
        rebalance_schedule=rebalance_schedule,
    )

    scenarios = list(scenario) if scenario else None
    results = run_stress_test(base_config, scenarios)
    print_stress_results(results)


@cli.command("compare")
@click.argument("paths", nargs=-1, required=True)
def compare_cmd(paths):
    """Compare saved backtest results side-by-side (Gap 30)."""
    from backtester.result import BacktestResult
    df = BacktestResult.compare(list(paths))
    key_cols = ["path", "strategy", "total_return", "cagr", "sharpe_ratio",
                "max_drawdown", "total_trades", "win_rate"]
    display = [c for c in key_cols if c in df.columns]
    click.echo(df[display].to_string(index=False))


if __name__ == "__main__":
    cli()
