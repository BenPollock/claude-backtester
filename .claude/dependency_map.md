# Dependency Map

## Internal Dependency Graph

| Module | Imports From |
|--------|-------------|
| `cli.py` | config, engine, analytics.report, strategies.registry, strategies.sma_crossover, data.universe, research.optimizer, research.walk_forward |
| `config.py` | (none -- leaf) |
| `types.py` | (none -- leaf) |
| `engine.py` | config, types, data.manager, data.calendar, strategies.base, strategies.registry, strategies.indicators, execution.broker, execution.slippage, execution.fees, execution.position_sizing, portfolio.portfolio, portfolio.order, portfolio.position |
| `data/*` | (none -- leaf; no internal backtester imports) |
| `strategies/*` | types.SignalAction, portfolio.portfolio.PortfolioState, portfolio.position.Position |
| `execution/*` | portfolio.order, portfolio.portfolio, types.Side, types.OrderStatus |
| `portfolio/*` | types.Side, types.OrderType, types.OrderStatus |
| `analytics/*` | engine.BacktestResult (in report.py), analytics.calendar (lazy self-import) |
| `research/*` | config.BacktestConfig, engine.BacktestEngine, analytics.metrics.compute_all_metrics, research.optimizer.grid_search (walk_forward uses optimizer) |

## Leaf Modules (no internal deps -- safest to modify)

- **`data/*`** -- No backtester imports at all. Pure external deps (pandas, yfinance, exchange_calendars).
- **`config.py`** -- Only stdlib imports. Defines frozen dataclasses.
- **`types.py`** -- Only stdlib enum. Defines Side, SignalAction, OrderType, OrderStatus.

## Hub Modules (imported by many -- most dangerous to modify)

| Module | Imported By |
|--------|------------|
| `types.py` | engine, strategies, execution, portfolio |
| `config.py` | engine, cli, research |
| `portfolio.portfolio` | engine, execution, strategies (via PortfolioState) |
| `portfolio.order` | engine, execution |
| `portfolio.position` | engine, strategies |
| `engine.py` | cli, analytics.report, research |

## Cross-Cutting Concerns

These concepts span multiple modules. Changes require coordinated updates:

- **`SignalAction` enum** (types.py) -- Used by strategies to emit signals; engine interprets them for regime filter and order routing. Adding a new action requires changes in strategies, engine, and potentially broker.
- **`Side` enum** (types.py) -- Used by portfolio (Order), execution (broker fills), and engine. Distinct from SignalAction by design.
- **`-1` sell sentinel** -- Strategy `size_order()` returns -1; broker `process_fills()` resolves to full position qty. Convention spans strategies, engine, and execution.
- **DataFrame column names** -- `compute_indicators()` adds columns to DataFrames; `generate_signals()` reads them from row; engine passes rows. Column name changes require matching updates across strategy indicator/signal methods.
- **Frozen config pattern** -- `BacktestConfig` is frozen; engine reads it, research clones via `replace()`, CLI constructs it. Any field addition touches config, CLI arg parsing, and engine consumption.
- **T/T+1 fill timing** -- Encoded in `fill_delay_days=1` (config), enforced by broker (fill at Open), assumed by engine (day loop order). Cannot be changed in isolation.
- **FIFO lot tracking** -- Position manages lots; broker delegates sells to `sell_lots_fifo()`; engine reads `_market_price`. Lot structure changes affect all three.
