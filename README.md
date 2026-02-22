# claude-backtester

A modular, research-grade stock backtesting engine in Python. Supports pluggable strategies, NYSE trading calendar, Parquet data caching, and professional analytics output.

## Features

- **Pluggable strategies** — SMA Crossover and Rule-Based strategies included, with an ABC for custom strategies
- **Lookahead-free execution** — Signals use close of day T; orders fill at open of T+1
- **Regime filtering** — Engine-level benchmark SMA filter suppresses BUY signals in unfavorable regimes
- **Position sizing models** — Fixed fractional, ATR-based, and volatility parity
- **Risk management** — Stop-loss, take-profit, and trailing stop (fixed or ATR-based)
- **Parameter optimization** — Grid search across strategy parameters with configurable objective metric
- **Walk-forward analysis** — Rolling or anchored in-sample/out-of-sample optimization windows
- **Monte Carlo simulation** — Statistical confidence intervals on backtest results
- **Multi-market universes** — US, Canadian, or combined ticker universes (index constituents or full market)
- **Data caching** — Parquet-based local cache with Yahoo Finance as the data source
- **Professional analytics** — CAGR, Sharpe ratio, max drawdown, monthly/yearly return calendars, and more

## Requirements

- Python 3.11+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI Commands

### `backtester run`

Run a single backtest.

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2001-01-01 \
  --end 2010-12-31 \
  --cash 10000 \
  --max-positions 100 \
  --max-alloc 0.10 \
  --params '{"sma_fast":100,"sma_slow":200}'
```

**Key options:**

| Option | Description |
|---|---|
| `--strategy` | Strategy name (`sma_crossover`, `rule_based`) |
| `--tickers` | Comma-separated ticker symbols |
| `--market` | Market scope when tickers omitted (`us`, `ca`, `us_ca`) |
| `--universe` | Universe breadth (`index`, `all`) |
| `--benchmark` | Benchmark ticker for comparison |
| `--start` / `--end` | Date range (YYYY-MM-DD) |
| `--cash` | Starting capital (default: 10,000) |
| `--max-positions` | Max concurrent positions |
| `--max-alloc` | Max allocation per position (e.g. 0.10 = 10%) |
| `--fee` | Fee per trade in dollars |
| `--slippage-bps` | Slippage in basis points |
| `--params` | Strategy parameters as JSON string |
| `--position-sizing` | Sizing model: `fixed_fractional`, `atr`, `vol_parity` |
| `--risk-pct` | Risk per trade for ATR sizer (e.g. 0.01 = 1%) |
| `--stop-loss` | Stop-loss as fraction (e.g. 0.05 = 5%) |
| `--take-profit` | Take-profit as fraction (e.g. 0.20 = 20%) |
| `--trailing-stop` | Trailing stop as fraction (e.g. 0.08 = 8%) |
| `--stop-loss-atr` | Stop-loss in ATR multiples |
| `--take-profit-atr` | Take-profit in ATR multiples |
| `--regime-benchmark` | Regime filter benchmark ticker |
| `--regime-fast` / `--regime-slow` | Regime filter SMA periods |
| `--export-log` | Export activity log to CSV |

### `backtester optimize`

Run a parameter grid search to find optimal strategy settings.

```bash
backtester optimize \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2001-01-01 \
  --end 2010-12-31 \
  --grid '{"sma_fast":[50,100,150],"sma_slow":[200,250,300]}' \
  --metric sharpe_ratio
```

### `backtester walk-forward`

Run walk-forward analysis with rolling optimization windows.

```bash
backtester walk-forward \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2001-01-01 \
  --end 2020-12-31 \
  --grid '{"sma_fast":[50,100,150],"sma_slow":[200,250,300]}' \
  --is-months 36 \
  --oos-months 12 \
  --anchored
```

| Option | Description |
|---|---|
| `--is-months` | In-sample window length in months |
| `--oos-months` | Out-of-sample window length in months |
| `--anchored` | Use expanding (anchored) in-sample window |
| `--metric` | Metric to optimize (default: `sharpe_ratio`) |

### `backtester list-strategies`

List all registered strategies.

```bash
backtester list-strategies
```

## Architecture

```
src/backtester/
  cli.py            — Click CLI entry point
  engine.py         — Main backtest loop
  config.py         — BacktestConfig, RegimeFilter dataclasses
  types.py          — Side, OrderType, OrderStatus, SignalAction enums
  data/             — DataManager, ParquetCache, TradingCalendar, YahooDataSource
  strategies/       — Strategy ABC, registry, indicators, sma_crossover, rule_based
  execution/        — SimulatedBroker, SlippageModel, FeeModel, PositionSizing
  portfolio/        — Portfolio, Position (FIFO lots), Order/Fill/Trade
  analytics/        — Metrics, Monte Carlo, report, calendar returns
  research/         — Optimizer, WalkForward
```

## Testing

```bash
pytest tests/ -v
```

Run a single test:

```bash
pytest tests/test_portfolio.py::TestPosition::test_sell_fifo -v
```
