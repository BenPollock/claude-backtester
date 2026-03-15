# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project Overview

**claude-backtester** — Modular Python stock backtesting engine. Pluggable strategies, NYSE calendar, Parquet data caching, analytics output.

## Language & Tooling

- Python 3.11+, setuptools/pyproject.toml, venv
- Key deps: numpy, pandas, click, yfinance, matplotlib, pyarrow, exchange_calendars
- Optional deps: edgartools (SEC EDGAR integration — `pip install -e ".[edgar]"`)

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# Run
backtester run --strategy sma_crossover --tickers SPY --benchmark SPY \
  --start 2001-01-01 --end 2010-12-31 --cash 10000 \
  --max-positions 100 --max-alloc 0.10 --params '{"sma_fast":100,"sma_slow":200}'

# Test
pytest tests/ -v
pytest tests/test_portfolio.py::TestPosition::test_sell_fifo -v  # single test

# List strategies
backtester list-strategies
```

## Architecture

```
src/backtester/
  cli.py              — Click CLI entry point
  engine.py           — Main backtest loop (orchestrates everything)
  config.py           — BacktestConfig, RegimeFilter dataclasses
  types.py            — Side, OrderType, OrderStatus, SignalAction enums
  data/               — DataManager, ParquetCache, TradingCalendar, YahooDataSource
    edgar_source.py   — EDGAR 10-K/10-Q financial statements (edgartools)
    edgar_insider.py  — EDGAR Form 4 insider trading data
    edgar_institutional.py — EDGAR 13F institutional holdings
    edgar_events.py   — EDGAR 8-K material event signals
    fundamental.py    — EdgarDataManager (unified EDGAR + CSV data manager)
    fundamental_cache.py — Parquet cache for all EDGAR data types
  strategies/
    base.py           — Strategy ABC
    registry.py       — Strategy registration/lookup
    indicators.py     — Vectorized indicator helpers
    sma_crossover.py  — SMA crossover strategy
    rule_based.py     — Rule-based strategy
    value_quality.py  — Value + quality + trend filter (EDGAR)
    earnings_growth.py — Growth momentum combo (EDGAR)
    fundamental_screener.py — Flexible JSON rule-based screening (EDGAR)
    insider_following.py — Follow insider buying/selling signals (EDGAR)
    smart_money.py    — Institutional + insider + fundamental combo (EDGAR)
  execution/
    broker.py         — SimulatedBroker
    slippage.py       — SlippageModel
    fees.py           — FeeModel
    position_sizing.py — Position sizing logic
  portfolio/
    portfolio.py      — Portfolio state
    position.py       — Position with FIFO lots
    order.py          — Order/Fill/Trade models
  analytics/
    metrics.py        — CAGR, Sharpe, MaxDrawdown
    montecarlo.py     — Monte Carlo simulation
    report.py         — Output report generation
    calendar.py       — Trading calendar utilities
  research/
    optimizer.py      — Parameter optimization
    walk_forward.py   — Walk-forward analysis
```

## Key Design Rules

- **Data flow:** CLI → Engine → DataManager (cache-first) → EdgarDataManager.merge_all_onto_daily (adds `fund_`/`insider_`/`inst_`/`event_` columns) → Strategy.compute_indicators (vectorized) → day-by-day loop: Broker.process_fills → Strategy.generate_signals → Broker.submit_order → Analytics
- **Lookahead prevention:** Signals use close of day T; orders fill at open of T+1. Strategies only see current row.
- **Regime filter:** Engine-level — suppresses BUY signals when benchmark SMA condition is "off". Strategies don't handle benchmark logic.
- **Position sizing:** FIFO lot tracking in `position.py`
- **Strategy pattern:** Subclass Strategy ABC; register via `registry.py`

## Testing Strategy

This project has two tiers of tests:

1. **Unit tests** (`tests/test_*.py`) — test individual modules in isolation
2. **E2E integration tests** (`tests/test_e2e.py`) — run full backtests through the entire pipeline with only the data source mocked

**When writing tests, always consider whether an E2E test is appropriate in addition to (or instead of) a unit test.** E2E tests are preferred when:
- The change affects the data flow across multiple modules (e.g., engine → broker → portfolio)
- The change touches order execution, signal generation, or portfolio accounting
- The change involves configuration options that alter engine behavior (stops, regime filter, fees, sizing)
- You need to verify that an invariant holds end-to-end (e.g., no lookahead, cash accounting)

E2E tests use `MockDataSource` + `make_controlled_df()` to create deterministic price data, and run full backtests via `BacktestEngine.run()`. See `tests/test_e2e.py` for patterns.

```bash
# Run all tests
pytest tests/ -v

# Run only E2E tests
pytest tests/test_e2e.py -v

# Run only unit tests (exclude E2E)
pytest tests/ -v --ignore=tests/test_e2e.py
```

## Test Files

```
tests/
  test_e2e.py             — E2E integration tests (full pipeline, 28 tests)
  test_edgar_e2e.py       — EDGAR E2E integration tests (18 tests)
  test_fundamental.py     — Financial statement unit tests (53 tests)
  test_insider.py         — Insider trading unit tests (16 tests)
  test_institutional.py   — 13F institutional unit tests (12 tests)
  test_portfolio.py       test_broker.py       test_engine.py
  test_strategies.py      test_metrics.py      test_montecarlo.py
  test_data.py            test_calendar.py     test_slippage.py
  test_stops.py           test_optimizer.py    test_universe.py
  test_position_sizing.py test_activity_log.py conftest.py
```
