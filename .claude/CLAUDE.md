# Claude-Backtester Technical Guide

## Module Map

| Module | Path | Purpose |
|--------|------|---------|
| CLI / Config / Types | `src/backtester/cli.py`, `config.py`, `types.py` | Click CLI, frozen config dataclasses, Side/SignalAction/OrderType enums |
| Engine | `src/backtester/engine.py` | Central orchestrator: data load, day loop, regime filter, stop triggers |
| Data | `src/backtester/data/` | Cache-first OHLCV loading (Parquet), NYSE calendar, Yahoo fetcher, universe scraper |
| Strategies | `src/backtester/strategies/` | Strategy ABC + registry, SMA crossover, rule-based DSL, 18 indicator functions |
| Execution | `src/backtester/execution/` | SimulatedBroker (next-day fills), slippage/fee models, position sizers |
| Portfolio | `src/backtester/portfolio/` | Mutable Portfolio state, FIFO Position/Lot tracking, Order/Fill/Trade models |
| Analytics | `src/backtester/analytics/` | CAGR/Sharpe/drawdown metrics, report output, Monte Carlo, calendar analytics |
| Research | `src/backtester/research/` | Grid search parameter optimization, walk-forward IS/OOS analysis |
| Tests | `tests/` | ~195 tests, 14 files, all synthetic data, no network calls |

## Key Data Flows

**Main Backtest Pipeline:**
CLI parses args --> `BacktestConfig` (frozen) --> `BacktestEngine.run()` --> `DataManager.load_many()` (cache-first) --> `strategy.compute_indicators()` (vectorized, once per ticker) --> day-by-day loop --> `BacktestResult` --> `print_report()`

**Day Loop (strict order -- do not reorder):**
1. `broker.process_fills()` -- fill yesterday's orders at today's Open
2. `_set_stops_for_fills()` -- attach stops to new positions
3. `_check_stop_triggers()` -- intraday H/L stop exits (bypasses broker)
4. Update position market prices to Close
5. `portfolio.record_equity(day)` -- snapshot equity curve
6. `strategy.generate_signals()` per symbol --> regime filter --> position limit check
7. Size orders --> `broker.submit_order()` (fills next day)

**Order Lifecycle:**
`strategy.generate_signals()` returns `SignalAction` --> engine sizes order (BUY: PositionSizer; SELL: `strategy.size_order()` returning -1 sentinel) --> `broker.submit_order()` queues --> next day `process_fills()` reads Open, applies slippage/fees, mutates Portfolio

**Research Pipeline:**
`grid_search()` builds Cartesian param sweep --> per-combo `BacktestEngine.run()` --> `compute_all_metrics()` --> rank by metric. `walk_forward()` splits into rolling IS/OOS windows, uses `grid_search` on IS, validates best params on OOS.

## Shared Interfaces

| Interface | Location | Implementations |
|-----------|----------|-----------------|
| `Strategy` ABC | `strategies/base.py` | `SmaCrossover`, `RuleBasedStrategy` |
| `@register_strategy` | `strategies/registry.py` | Decorator populates `_REGISTRY` dict; lookup via `get_strategy(name)` |
| `DataSource` ABC | `data/manager.py` | `YahooDataSource` |
| `SlippageModel` ABC | `execution/slippage.py` | `FixedSlippage`, `VolumeSlippage` |
| `FeeModel` ABC | `execution/fees.py` | `PerTradeFee` |
| `PositionSizer` ABC | `execution/position_sizing.py` | `FixedFractional`, `ATRSizer`, `VolatilityParity` |

## Critical Invariants

- **T/T+1 Lookahead Prevention:** Signals use day T close; orders fill at day T+1 open. `fill_delay_days=1` in config. Fill at Open price, never Close.
- **FIFO Lot Accounting:** `Position.sell_lots_fifo()` always consumes `lots[0]` first. Partial lot commission adjusted in-place. Lot management lives in Position only (not duplicated in broker).
- **-1 Sell Sentinel:** `strategy.size_order()` returns -1 meaning "sell all"; broker resolves to `pos.total_quantity`. All strategies use this convention.
- **Frozen Configs:** All config dataclasses are `frozen=True`. Research uses `dataclasses.replace()` to create modified copies; base config is never mutated.
- **Strategy Auto-Discovery:** `discover_strategies()` in `registry.py` uses `pkgutil` to scan and import all strategy modules. Called at startup in `cli.py`, `conftest.py`, and `strategies/__init__.py`. New strategies are auto-discovered -- no manual imports needed.
- **Regime Filter is Engine-Level:** Strategies never handle benchmark logic. Engine suppresses BUY signals when regime is "off".
- **DataFrame Column Contracts:** `compute_indicators()` must call `df.copy()` to avoid mutating caller data. Indicators must be backward-looking only.
- **PortfolioState Immutability:** Strategies receive frozen `PortfolioState` snapshots, never mutable `Portfolio`.
- **SignalAction vs Side:** `SignalAction` (BUY/SELL/HOLD) is strategy-layer; `Side` (BUY/SELL) is execution-layer. Do not conflate.
- **Stop Triggers Bypass Broker:** Engine `_check_stop_triggers()` directly mutates portfolio (same-day execution).
- **252 Trading Days:** All annualization uses fixed 252-day assumption.

## Patterns

- **Frozen Dataclasses:** Config, Fill, Trade, BacktestResult, PortfolioState -- immutability prevents accidental mutation
- **ABC + Implementation:** Strategy, DataSource, SlippageModel, FeeModel, PositionSizer -- all use abstract base with concrete subclasses
- **Cache-First Loading:** ParquetCache checked before Yahoo fetch; stale-cache fallback for universe scraping
- **Vectorized-Then-Iterative:** `compute_indicators()` runs once vectorized per ticker; `generate_signals()` runs per-day iteratively
- **Silent Fallbacks:** ATR sizers fall back to FixedFractional when ATR unavailable; TradingCalendar falls back to bdate_range pre-1960s; ffill(limit=5) for gaps
- **Error-Tolerant Sweeps:** Grid search logs failed runs as error rows rather than raising exceptions

## Test Approach

```bash
pytest tests/ -v                                          # full suite
pytest tests/test_portfolio.py::TestPosition::test_sell_fifo -v  # single test
```

**Key Fixtures** (`conftest.py`): `make_price_df(seed=42)` generates deterministic OHLCV; `MockDataSource` serves pre-loaded DataFrames; `basic_config` provides standard BacktestConfig; `sample_df` is 252-day synthetic data; `portfolio` is fresh $100k Portfolio.

**Do not change:** conftest `discover_strategies()` call (registration), `make_price_df` seed=42, `basic_config` defaults, `MockDataSource` inclusive date filtering.

**Coverage gaps:** CLI layer (cli.py), report formatting (report.py), calendar analytics (monthly_returns, drawdown_periods) are untested.

**Test counts by area:** strategies (50), metrics (42), position sizing (17), portfolio (15), optimizer (12), stops (11), engine (9), broker (8), activity log (7), universe (7), data (6), slippage (4), Monte Carlo (4), calendar (3).
