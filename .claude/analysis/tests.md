# Tests Module Analysis: tests/

## Module Purpose

Comprehensive pytest test suite covering all backtester modules. Uses synthetic price data generation (`make_price_df`), a `MockDataSource` for deterministic data, and shared fixtures. ~195 test functions across 14 files.

## Key Test Files

| File | Tests | What It Covers |
|------|------:|----------------|
| `test_strategies.py` | 50 | Strategy ABC, SMA crossover, rule-based DSL, indicator registry, signal generation |
| `test_metrics.py` | 42 | All analytics metrics: CAGR, Sharpe, Sortino, drawdown, trade stats, benchmark-relative |
| `test_position_sizing.py` | 17 | FixedFractional, ATRSizer, VolatilityParity position sizers |
| `test_portfolio.py` | 15 | Portfolio state, Position FIFO lots, Order/Fill/Trade models |
| `test_optimizer.py` | 12 | Grid search, walk-forward analysis, parameter optimization |
| `test_stops.py` | 11 | Stop-loss, take-profit, trailing stop logic |
| `test_engine.py` | 9 | Full backtest engine integration tests |
| `test_broker.py` | 8 | SimulatedBroker order submission, filling, fee/slippage application |
| `test_activity_log.py` | 7 | TradeLogEntry, activity log CSV export |
| `test_universe.py` | 7 | UniverseProvider with mocked HTTP calls |
| `test_data.py` | 6 | ParquetCache, DataManager cache-first loading |
| `test_slippage.py` | 4 | FixedSlippage, VolumeSlippage models |
| `test_montecarlo.py` | 4 | Monte Carlo bootstrap simulation |
| `test_calendar.py` | 3 | TradingCalendar NYSE session queries |

## Test Patterns

### Shared Fixtures (conftest.py)
- `make_price_df()` — Generates synthetic OHLCV with seeded RNG (`seed=42`), configurable days/price/return
- `MockDataSource` — `DataSource` subclass that serves pre-loaded DataFrames by symbol
- `sample_df` fixture — 252-day synthetic data
- `portfolio` fixture — Fresh Portfolio with $100k cash
- `mock_source` fixture — Empty MockDataSource
- `basic_config` fixture — BacktestConfig with sma_crossover, TEST ticker, 2020 date range

### Common Approaches
- **No network calls**: All data sources mocked; `test_universe.py` patches `urllib.request`
- **Deterministic RNG**: `np.random.default_rng(42)` for reproducible synthetic data
- **Temp directories**: `tempfile.mkdtemp()` for cache/file tests (test_data.py, test_activity_log.py)
- **Parametrized tests**: Used in test_strategies.py and test_metrics.py for indicator/metric coverage
- **Direct unit tests**: Most tests instantiate classes directly rather than going through CLI

### Strategy registration side-effect
- `conftest.py` imports `sma_crossover` and `rule_based` modules to trigger `@register_strategy` decorators

## Coverage Gaps

- **CLI layer** (`cli.py`): No test file — Click commands untested
- **Report formatting** (`report.py`): No dedicated tests for console output or plot generation
- **Calendar analytics** (`analytics/calendar.py`): `test_calendar.py` only covers `TradingCalendar`, not `monthly_returns`/`drawdown_periods`
- **Integration with real data**: All tests use synthetic data; no smoke tests with real Yahoo data

## "Do Not Touch" Warnings

1. **`conftest.py` strategy imports**: The `import backtester.strategies.sma_crossover` and `import backtester.strategies.rule_based` lines trigger registration. Removing them breaks any test that uses `get_strategy()`.
2. **`make_price_df()` seed=42**: Many tests depend on the exact synthetic data shape. Changing the seed or generation logic will cascade test failures.
3. **`basic_config` fixture**: Many test files depend on this exact configuration (sma_crossover, TEST ticker, 2020 dates, $100k). Modifying defaults will break multiple test files.
4. **`MockDataSource` date filtering**: Uses `>=` and `<=` for inclusive date range. This matches the real DataManager behavior — must stay in sync.
