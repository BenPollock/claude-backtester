# Root Module Analysis: cli.py, config.py, types.py

## Module Purpose

These three files form the entry layer of the backtester. `cli.py` parses user input via Click and assembles configuration objects. `config.py` defines frozen dataclasses that carry all backtest parameters through the system. `types.py` provides the core enumerations (`Side`, `OrderType`, `OrderStatus`, `SignalAction`) used by every downstream module.

## Key Classes/Functions

### cli.py
| Name | Description |
|------|-------------|
| `cli()` | Click group; sets up logging via `--verbose` flag |
| `run()` | Main command — parses all options, builds `BacktestConfig`, runs engine, prints report |
| `list_strats()` | Lists registered strategies to stdout |
| `optimize()` | Runs parameter grid search via `research.optimizer` |
| `walk_forward_cmd()` | Runs walk-forward analysis via `research.walk_forward` |
| `_build_config()` | Shared helper that constructs a `BacktestConfig` from CLI args (used by optimize/walk-forward) |

### config.py
| Name | Description |
|------|-------------|
| `RegimeFilter` | Frozen dataclass for benchmark-based regime filter (SMA fast/slow comparison) |
| `StopConfig` | Frozen dataclass for stop-loss, take-profit, and trailing-stop settings (pct or ATR) |
| `BacktestConfig` | Frozen dataclass holding all backtest parameters — the single source of truth passed to engine |

### types.py
| Name | Description |
|------|-------------|
| `Side` | Enum: BUY, SELL |
| `OrderType` | Enum: MARKET, LIMIT, STOP, STOP_LIMIT |
| `OrderStatus` | Enum: PENDING, FILLED, CANCELLED |
| `SignalAction` | Enum: BUY, SELL, HOLD — returned by strategies |

## Critical Data Flows

1. **CLI args → Config**: `run()` parses Click options, builds `RegimeFilter` (optional), `StopConfig` (optional), then assembles `BacktestConfig`.
2. **Config → Engine**: `BacktestConfig` is passed to `BacktestEngine(config)`, which calls `engine.run()`.
3. **Engine result → Output**: `engine.run()` returns a result object consumed by `print_report()`, `plot_results()`, and optionally `export_activity_log_csv()`.
4. **Tickers resolution**: If `--tickers` is omitted, `UniverseProvider` is lazily imported to resolve tickers from `--market` and `--universe`.
5. **Strategy params**: JSON string from `--params` is deserialized and stored in `BacktestConfig.strategy_params` dict.

## External Dependencies

### Third-party
- `click` — CLI framework (cli.py)
- `json`, `logging`, `datetime` — stdlib (cli.py, config.py)
- `enum` — stdlib (types.py)

### Internal backtester imports (from cli.py)
- `backtester.config` — `BacktestConfig`, `RegimeFilter`, `StopConfig`
- `backtester.engine` — `BacktestEngine`
- `backtester.analytics.report` — `print_report`, `plot_results`, `export_activity_log_csv`
- `backtester.strategies.registry` — `list_strategies`
- `backtester.strategies.sma_crossover` — imported at module level to trigger `@register` side effect
- `backtester.data.universe` — `UniverseProvider` (lazy import)
- `backtester.research.optimizer` — `grid_search`, `print_optimization_results` (lazy import)
- `backtester.research.walk_forward` — `walk_forward`, `print_walk_forward_results` (lazy import)

## "Do Not Touch" Warnings

1. **`fill_delay_days = 1` in `BacktestConfig`**: Encodes the lookahead-prevention invariant (signal on close T, fill at open T+1). Changing this breaks backtest integrity.
2. **`frozen=True` on all config dataclasses**: Configs are immutable by design; the engine and downstream consumers assume configs never mutate mid-run.
3. **`import backtester.strategies.sma_crossover` (line 10, cli.py)**: This side-effect import triggers strategy registration. Removing it silently breaks `--strategy sma_crossover`. Any new strategy module needs an analogous import here.
4. **`_build_config()` vs `run()`**: `run()` has its own inline config construction with regime/stop support. `_build_config()` is a simpler shared builder used by optimize/walk-forward and does NOT include regime or stop config. These are intentionally different.
5. **`SignalAction` vs `Side`**: `SignalAction` (BUY/SELL/HOLD) is for strategy signals; `Side` (BUY/SELL) is for order execution. Do not conflate them — they serve different layers.
