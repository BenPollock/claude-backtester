# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project Overview

**claude-backtester** — Modular Python stock backtesting engine. Pluggable strategies, NYSE calendar, Parquet data caching, analytics output.

## Language & Tooling

- Python 3.11+, setuptools/pyproject.toml, venv
- Key deps: numpy, pandas, click, yfinance, matplotlib, pyarrow, exchange_calendars

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
  strategies/
    base.py           — Strategy ABC
    registry.py       — Strategy registration/lookup
    indicators.py     — Vectorized indicator helpers
    sma_crossover.py  — SMA crossover strategy
    rule_based.py     — Rule-based strategy
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

- **Data flow:** CLI → Engine → DataManager (cache-first) → Strategy.compute_indicators (vectorized) → day-by-day loop: Broker.process_fills → Strategy.generate_signals → Broker.submit_order → Analytics
- **Lookahead prevention:** Signals use close of day T; orders fill at open of T+1. Strategies only see current row.
- **Regime filter:** Engine-level — suppresses BUY signals when benchmark SMA condition is "off". Strategies don't handle benchmark logic.
- **Position sizing:** FIFO lot tracking in `position.py`
- **Strategy pattern:** Subclass Strategy ABC; register via `registry.py`

## Test Files

```
tests/
  test_portfolio.py       test_broker.py       test_engine.py
  test_strategies.py      test_metrics.py      test_montecarlo.py
  test_data.py            test_calendar.py     test_slippage.py
  test_stops.py           test_optimizer.py    test_universe.py
  test_position_sizing.py test_activity_log.py conftest.py
```
