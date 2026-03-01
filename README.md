# claude-backtester

A modular, research-grade stock backtesting engine in Python. Supports pluggable strategies, NYSE trading calendar, Parquet data caching, and professional analytics output.

## Features

- **Pluggable strategies** — SMA Crossover and Rule-Based strategies included, with an ABC for custom strategies
- **Lookahead-free execution** — Signals use close of day T; orders fill at open of T+1
- **Regime filtering** — Engine-level benchmark SMA filter suppresses BUY signals in unfavorable regimes
- **Position sizing models** — Fixed fractional, ATR-based, and volatility parity
- **Risk management** — Stop-loss, take-profit, and trailing stop (fixed or ATR-based)
- **Short selling** — SHORT/COVER signals, negative positions, margin tracking, borrow cost accrual
- **Limit orders** — Limit price fills within H/L range, DAY and GTC time-in-force, expiry dates
- **Realistic fee models** — Flat per-trade, percentage (bps), tiered, SEC fee, TAF fee, and composite fee stacking
- **Multi-timeframe data** — Strategies can access weekly/monthly resampled data alongside daily bars
- **Parameter optimization** — Grid search across strategy parameters with configurable objective metric
- **Walk-forward analysis** — Rolling or anchored in-sample/out-of-sample optimization windows
- **Multi-strategy portfolios** — Run multiple strategies simultaneously with capital allocation and attribution
- **Monte Carlo simulation** — Statistical confidence intervals on backtest results
- **Multi-market universes** — US, Canadian, or combined ticker universes (index constituents or full market)
- **Data caching** — Parquet-based local cache with Yahoo Finance as the data source
- **Professional analytics** — CAGR, Sharpe, max drawdown, monthly returns, HTML tearsheets, regime analysis, signal decay, correlation/concentration metrics

## Requirements

- Python 3.11+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Examples

### 1. Simplest run — SPY SMA crossover, 10 years

```bash
backtester run --strategy sma_crossover --tickers SPY --benchmark SPY --start 2010-01-01 --end 2020-12-31
```

Default cash ($10k), default SMA params. Good for a quick sanity check.

---

### 2. Multi-ticker portfolio with allocation limits

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,NVDA \
  --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --cash 50000 \
  --max-positions 5 \
  --max-alloc 0.20
```

Caps each position at 20% of portfolio, no more than 5 open at once.

---

### 3. Add stop-loss and take-profit

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY,QQQ,IWM \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --stop-loss 0.05 \
  --take-profit 0.20 \
  --trailing-stop 0.08
```

Exit if down 5%, lock in profits if up 20%, or trail with an 8% stop from peak.

---

### 4. ATR-based position sizing with ATR stops

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --position-sizing atr \
  --risk-pct 0.01 \
  --stop-loss-atr 2.0 \
  --take-profit-atr 4.0
```

Risk 1% of capital per trade. Stop placed 2× ATR from entry, target at 4× ATR.

---

### 5. Regime filter — only buy when market is trending up

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL \
  --benchmark SPY \
  --start 2001-01-01 --end 2023-12-31 \
  --regime-benchmark SPY \
  --regime-fast 50 \
  --regime-slow 200
```

Suppresses new BUY signals when SPY 50-day SMA is below its 200-day SMA.

---

### 6. Tune strategy parameters with grid search

```bash
backtester optimize \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2001-01-01 --end 2015-12-31 \
  --grid '{"sma_fast":[20,50,100],"sma_slow":[100,200,300]}' \
  --metric sharpe_ratio
```

Finds the SMA combination with the best Sharpe ratio over the in-sample period.

---

### 7. Walk-forward analysis — validate out-of-sample

```bash
backtester walk-forward \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2001-01-01 --end 2023-12-31 \
  --grid '{"sma_fast":[20,50,100],"sma_slow":[100,200,300]}' \
  --is-months 36 \
  --oos-months 12
```

Re-optimizes every 12 months using the prior 3 years of data. Helps detect overfitting.

---

### 8. Export trade log for further analysis

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2020-12-31 \
  --export-log trades.csv
```

Writes a full activity log (entries, exits, fills) to `trades.csv`.

---

### 9. Rule-based strategy — RSI mean reversion with trend filter

```bash
backtester run \
  --strategy rule_based \
  --tickers SPY,QQQ,IWM \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --cash 30000 \
  --max-positions 3 \
  --max-alloc 0.33 \
  --params '{
    "indicators": {
      "rsi":    {"fn": "rsi",  "period": 14},
      "sma200": {"fn": "sma",  "period": 200}
    },
    "buy_when":  [["rsi", "<", 35], ["Close", ">", "sma200"]],
    "sell_when": [["rsi", ">", 65]]
  }'
```

Buy when RSI drops below 35 (oversold) **and** price is above its 200-day SMA (uptrend). Sell when RSI recovers above 65. Rules use AND logic — all conditions must be true to trigger.

---

### 10. Short selling via CLI

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --allow-short \
  --short-borrow-rate 0.02 \
  --margin-requirement 1.5
```

Enables SHORT/COVER signals from the strategy. Borrow cost is 2% annualized, margin requirement is 150%. Short positions profit when price falls, and stop-losses trigger on price rises.

---

### 11. Generate an HTML tearsheet with analytics

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL \
  --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --tearsheet report.html \
  --report-regime \
  --report-signal-decay \
  --report-correlation
```

Produces a self-contained HTML tearsheet plus console output for regime breakdown, signal decay analysis, and correlation matrix.

---

### 12. Volume slippage + percentage fees

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --slippage-model volume \
  --slippage-impact 0.1 \
  --fee-model percentage \
  --fee 5
```

Uses a volume-impact slippage model and charges 5 basis points per trade. Alternatively, `--fee-model composite_us` applies a realistic SEC + FINRA TAF fee stack.

---

### 13. Volatility parity position sizing

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN \
  --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --position-sizing vol_parity \
  --vol-target 0.10 \
  --vol-lookback 20
```

Sizes positions so each contributes roughly equal volatility (10% target annualized), using a 20-day lookback for realized volatility.

---

## Advanced Examples

The examples below cover advanced CLI flag combinations and Python-API-only features (limit orders, multi-timeframe strategies, multi-strategy portfolios).

### 14. Short selling with stops and regime filter

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --allow-short \
  --short-borrow-rate 0.02 \
  --margin-requirement 1.5 \
  --stop-loss 0.05 \
  --take-profit 0.15 \
  --regime-benchmark SPY \
  --regime-fast 50 \
  --regime-slow 200
```

Combines short selling (2% borrow cost, 150% margin) with risk management (5% stop-loss, 15% take-profit) and a regime filter. Strategies can return `SignalAction.SHORT` and `SignalAction.COVER`. Short positions have inverted PnL (profit when price falls), FIFO lot tracking, and stop-losses that trigger on price rises.

---

### 15. Limit orders — use Signal dataclass in custom strategies

```python
from backtester.strategies.base import Signal
from backtester.types import SignalAction

class MyStrategy(Strategy):
    def generate_signals(self, row, portfolio_state):
        # Buy with a limit order 1% below current close
        return Signal(
            action=SignalAction.BUY,
            limit_price=row["Close"] * 0.99,
            time_in_force="GTC",  # good-til-cancelled (persists across days)
        )
```

BUY limit orders fill when the day's Low reaches the limit price. SELL limits fill when High reaches it. DAY orders expire at end of day; GTC orders persist until filled or an optional `expiry_date`.

---

### 16. Realistic fee models — percentage, tiered, and regulatory fees

**Percentage fees** — charge 5 basis points on notional value:

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --fee-model percentage \
  --fee 5
```

**Composite US equity fees** — SEC fee + FINRA TAF + broker commission (3 bps):

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --fee-model composite_us \
  --fee 3
```

**Tiered fees** — lower rates for larger orders (Python API only, no CLI equivalent):

```python
from backtester.execution.fees import TieredFee

fee_model = TieredFee(tiers=[
    (0, 10),        # 0-10k notional: 10 bps
    (10_000, 5),    # 10k-100k: 5 bps
    (100_000, 2),   # 100k+: 2 bps
])
```

Fee models follow the `FeeModel` ABC. Use `CompositeFee` to stack multiple models via the Python API — each component's fee is summed.

---

### 17. Multi-timeframe strategy — weekly trend + daily entry

```python
class WeeklyTrendDaily(Strategy):
    name = "weekly_trend_daily"

    @property
    def timeframes(self):
        return ["daily", "weekly"]  # request weekly resampled data

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["sma_fast"] = df["Close"].rolling(10).mean()
        # Weekly data is forward-filled onto daily index
        if timeframe_data and "weekly" in timeframe_data:
            weekly = timeframe_data["weekly"]
            df["weekly_sma"] = weekly["Close"].rolling(20).mean()
        return df

    def generate_signals(self, row, portfolio_state):
        # Use weekly_Close and weekly_sma columns (merged into daily row)
        if row.get("weekly_Close") and row.get("weekly_sma"):
            if row["Close"] > row["sma_fast"] and row["weekly_Close"] > row["weekly_sma"]:
                return SignalAction.BUY
        return SignalAction.HOLD
```

The engine automatically resamples daily OHLCV to weekly/monthly bars (Open=first, High=max, Low=min, Close=last, Volume=sum) and forward-fills onto the daily index. Weekly columns appear in the row as `weekly_Open`, `weekly_Close`, etc.

---

### 18. Generate an HTML tearsheet

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --tearsheet my_tearsheet.html
```

Produces a self-contained HTML file (no external dependencies) with: equity curve, drawdown chart, rolling metrics, monthly returns heatmap, trade statistics, and key metrics table. Open in any browser.

---

### 19. Multi-strategy portfolio — run strategies side by side

```python
from backtester.research.multi_strategy import (
    StrategyAllocation, MultiStrategyConfig,
    run_multi_strategy, print_multi_strategy_report,
)

multi_config = MultiStrategyConfig(
    allocations=(
        StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 200}, weight=0.6),
        StrategyAllocation("rule_based", {
            "indicators": {"rsi": {"fn": "rsi", "period": 14}},
            "buy_when": [["rsi", "<", 30]],
            "sell_when": [["rsi", ">", 70]],
        }, weight=0.4),
    ),
    base_config=base_config,  # shared tickers, dates, benchmark
)

result = run_multi_strategy(multi_config)
print_multi_strategy_report(result)

# Access individual strategy results
for name, metrics in result.per_strategy_metrics.items():
    print(f"{name}: CAGR={metrics['cagr']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}")
```

Each strategy runs independently with its allocated share of capital. The combined equity curve is the sum of individual curves. Attribution analysis shows each strategy's contribution to total return.

---

### 20. Signal decay analysis — find optimal holding period

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --report-signal-decay
```

Measures how entry signals perform over T+1 through T+N days. Identifies the horizon where average cumulative return peaks — useful for tuning holding periods.

---

### 21. Regime performance breakdown

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --report-regime
```

Classifies each trading day as bull/bear/sideways (SMA-based) and low/medium/high volatility (rolling realized vol percentiles), then computes return, Sharpe, and drawdown metrics for each regime.

---

### 22. Correlation and concentration analysis

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL \
  --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --report-correlation \
  --report-concentration
```

Prints the pairwise return correlation matrix across tickers and portfolio concentration metrics (HHI, effective N, position weights). Sector exposure analysis (`compute_sector_exposure`) requires a user-supplied sector map and is available via the Python API only.

---

## CLI Commands

All commands support `--verbose` / `-v` for debug logging.

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

**Options:**

| Option | Description | Default |
|---|---|---|
| **Core** | | |
| `--strategy` | Strategy name (`sma_crossover`, `rule_based`) | **required** |
| `--tickers` | Comma-separated ticker symbols | uses `--market`/`--universe` |
| `--market` | Market scope when tickers omitted (`us`, `ca`, `us_ca`) | `us_ca` |
| `--universe` | Universe breadth (`index`, `all`) | `index` |
| `--benchmark` | Benchmark ticker for comparison | **required** |
| `--start` / `--end` | Date range (YYYY-MM-DD) | **required** |
| `--cash` | Starting capital | `10000` |
| `--max-positions` | Max concurrent positions | `100` |
| `--max-alloc` | Max allocation per position (e.g. 0.10 = 10%) | `0.10` |
| `--params` | Strategy parameters as JSON string | `{}` |
| `--cache-dir` | Parquet data cache directory | `~/.backtester/cache` |
| **Fees & Slippage** | | |
| `--fee-model` | Fee model: `per_trade`, `percentage`, `composite_us` | `per_trade` |
| `--fee` | Fee amount: dollars for `per_trade`, basis points for `percentage`/`composite_us` | `0.05` |
| `--slippage-model` | Slippage model: `fixed`, `volume` | `fixed` |
| `--slippage-bps` | Slippage in basis points (for `fixed` model) | `10.0` |
| `--slippage-impact` | Impact factor for `volume` slippage model | `0.1` |
| **Position Sizing** | | |
| `--position-sizing` | Sizing model: `fixed_fractional`, `atr`, `vol_parity` | `fixed_fractional` |
| `--risk-pct` | Risk per trade for `atr` sizer (e.g. 0.01 = 1%) | `0.01` |
| `--atr-multiple` | ATR multiple for stop distance in `atr` sizer | `2.0` |
| `--vol-target` | Target annualized volatility for `vol_parity` sizer | `0.10` |
| `--vol-lookback` | Lookback window (days) for `vol_parity` sizer | `20` |
| **Stops** | | |
| `--stop-loss` | Stop-loss as fraction (e.g. 0.05 = 5%) | disabled |
| `--take-profit` | Take-profit as fraction (e.g. 0.20 = 20%) | disabled |
| `--trailing-stop` | Trailing stop as fraction (e.g. 0.08 = 8%) | disabled |
| `--stop-loss-atr` | Stop-loss in ATR multiples | disabled |
| `--take-profit-atr` | Take-profit in ATR multiples | disabled |
| **Regime Filter** | | |
| `--regime-benchmark` | Regime filter benchmark ticker | disabled |
| `--regime-fast` | Regime filter fast SMA period | `100` |
| `--regime-slow` | Regime filter slow SMA period | `200` |
| `--regime-condition` | Regime condition: `fast_above_slow`, `fast_below_slow` | `fast_above_slow` |
| **Short Selling** | | |
| `--allow-short` | Enable short selling (flag) | `false` |
| `--short-borrow-rate` | Annualized short borrow rate | `0.02` |
| `--margin-requirement` | Initial margin requirement (1.5 = 150%) | `1.5` |
| **Output & Analytics** | | |
| `--export-log` | Export activity log to CSV file path | disabled |
| `--tearsheet` | Generate HTML tearsheet at file path | disabled |
| `--report-regime` | Print regime performance breakdown (flag) | `false` |
| `--report-signal-decay` | Print signal decay analysis (flag) | `false` |
| `--report-correlation` | Print correlation matrix (flag) | `false` |
| `--monte-carlo-runs` | Number of Monte Carlo simulations | `1000` |

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

Supports all common options from `backtester run` (core, fees, slippage, position sizing, stops, regime, short selling), plus:

| Option | Description | Default |
|---|---|---|
| `--grid` | Parameter grid as JSON (e.g. `{"sma_fast":[50,100]}`) | **required** |
| `--metric` | Metric to optimize | `sharpe_ratio` |

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

Supports all common options from `backtester run` (core, fees, slippage, position sizing, stops, regime, short selling), plus:

| Option | Description | Default |
|---|---|---|
| `--grid` | Parameter grid as JSON | **required** |
| `--is-months` | In-sample window length in months | `12` |
| `--oos-months` | Out-of-sample window length in months | `3` |
| `--anchored` | Use expanding (anchored) in-sample window (flag) | `false` |
| `--metric` | Metric to optimize | `sharpe_ratio` |

### `backtester list-strategies`

List all registered strategies.

```bash
backtester list-strategies
```

## Architecture

```
src/backtester/
  cli.py              — Click CLI entry point
  engine.py           — Main backtest loop (orchestrates everything)
  config.py           — BacktestConfig, RegimeFilter, StopConfig dataclasses
  types.py            — Side, OrderType, OrderStatus, SignalAction enums
  data/
    manager.py        — DataManager, resample_ohlcv (daily → weekly/monthly)
    cache.py          — ParquetCache for local OHLCV storage
    calendar.py       — TradingCalendar (NYSE-aware)
    source.py         — YahooDataSource
    universe.py       — Universe provider (S&P 500, TSX, etc.)
  strategies/
    base.py           — Strategy ABC, Signal dataclass (limit order support)
    registry.py       — Strategy auto-discovery and lookup
    indicators.py     — 18 vectorized indicator functions
    sma_crossover.py  — SMA crossover strategy
    rule_based.py     — Rule-based DSL strategy
  execution/
    broker.py         — SimulatedBroker (market + limit fills, short selling)
    slippage.py       — FixedSlippage, VolumeSlippage
    fees.py           — PerTradeFee, PercentageFee, TieredFee, SECFee, TAFFee, CompositeFee
    position_sizing.py — FixedFractional, ATRSizer, VolatilityParity
    stops.py          — StopManager (long + short stop/take-profit/trailing)
  portfolio/
    portfolio.py      — Portfolio state, margin tracking
    position.py       — Position with FIFO lots (long + short)
    order.py          — Order (market/limit, DAY/GTC), Fill, Trade
  analytics/
    metrics.py        — CAGR, Sharpe, Sortino, drawdown, benchmark-relative metrics
    montecarlo.py     — Monte Carlo simulation
    report.py         — Console report output
    calendar.py       — Monthly returns, drawdown periods
    tearsheet.py      — Self-contained HTML tearsheet with embedded charts
    correlation.py    — Correlation matrix, HHI, sector exposure
    signal_decay.py   — Signal return attribution, optimal holding period
    regime.py         — Bull/bear/sideways classification, per-regime metrics
  research/
    optimizer.py      — Grid search parameter optimization
    walk_forward.py   — Walk-forward IS/OOS analysis
    multi_strategy.py — Multi-strategy allocation, combined equity, attribution
```

## Testing

```bash
pytest tests/ -v          # full suite (~492 tests)
```

Run a single test:

```bash
pytest tests/test_portfolio.py::TestPosition::test_sell_fifo -v
```
