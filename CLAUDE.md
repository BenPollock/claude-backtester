# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**claude-backtester** — A modular, research-grade stock backtesting engine in Python. Supports pluggable strategies, NYSE trading calendar, Parquet data caching, and professional analytics output.

## Language & Tooling

- **Language:** Python 3.11+
- **Build/dependency tooling:** setuptools via pyproject.toml, venv for isolation
- **Key deps:** numpy, pandas, click, yfinance, matplotlib, pyarrow, exchange_calendars

## Commands

- **Setup:** `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
- **Run:** `backtester run --strategy sma_crossover --tickers SPY --benchmark SPY --start 2001-01-01 --end 2010-12-31 --cash 10000 --max-positions 100 --max-alloc 0.10 --params '{"sma_fast":100,"sma_slow":200}'`
- **Test:** `pytest tests/ -v`
- **Single test:** `pytest tests/test_portfolio.py::TestPosition::test_sell_fifo -v`
- **List strategies:** `backtester list-strategies`

## Architecture

```
src/backtester/
  cli.py          — Click CLI entry point
  engine.py       — Main backtest loop (orchestrates everything)
  config.py       — BacktestConfig, RegimeFilter dataclasses
  types.py        — Side, OrderType, OrderStatus, SignalAction enums
  data/           — DataManager, ParquetCache, TradingCalendar, YahooDataSource
  strategies/     — Strategy ABC, registry, indicators, sma_crossover
  execution/      — SimulatedBroker, SlippageModel, FeeModel
  portfolio/      — Portfolio, Position (FIFO lots), Order/Fill/Trade
  analytics/      — Metrics (CAGR, Sharpe, MaxDD), Monte Carlo, report
```

**Data flow:** CLI → Engine → DataManager (cache-first) → Strategy.compute_indicators (vectorized) → day-by-day loop: Broker.process_fills → Strategy.generate_signals → Broker.submit_order → Analytics.

**Lookahead prevention:** Signals use close of day T; orders fill at open of T+1. Strategies only see the current row.

**Regime filter:** Engine-level — suppresses BUY signals when benchmark SMA condition is "off". Strategies don't handle benchmark checks.
