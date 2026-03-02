# claude-backtester

A modular, research-grade stock backtesting engine in Python. Supports pluggable strategies, NYSE trading calendar, Parquet data caching, and professional analytics output.

## Features

- **Pluggable strategies** — SMA Crossover and Rule-Based strategies included, plus a cross-sectional strategy ABC for universe ranking
- **Lookahead-free execution** — Signals use close of day T; orders fill at open of T+1
- **Regime filtering** — Engine-level benchmark SMA filter suppresses BUY signals in unfavorable regimes
- **Position sizing models** — Fixed fractional, ATR-based, volatility parity, Kelly criterion, and risk parity
- **Risk management** — Stop-loss, take-profit, trailing stop (fixed or ATR-based), drawdown kill switch, VaR/CVaR
- **Exposure controls** — Sector exposure limits, gross/net exposure limits, portfolio-level volatility targeting
- **Short selling** — SHORT/COVER signals, negative positions, margin tracking, borrow cost deduction from cash
- **Order types** — Market, limit (DAY/GTC), stop, stop-limit, and bracket/OCO orders
- **Execution realism** — Partial fills with volume constraints, fill price models (open/close/VWAP/random), square-root market impact (Almgren-Chriss)
- **Realistic fee models** — Flat per-trade, percentage (bps), tiered, SEC fee, TAF fee, and composite fee stacking
- **Lot accounting** — FIFO (default), LIFO, highest-cost, and lowest-cost lot selection
- **Multi-timeframe data** — Strategies can access weekly/monthly resampled data alongside daily bars
- **Multi-source data** — Yahoo Finance, CSV files, or Parquet files as data sources
- **Fundamental data** — Point-in-time fundamental data sidecar with binary search lookups
- **Survivorship-bias-free universes** — Historical universe membership via CSV snapshots
- **Corporate actions** — Configurable price adjustment (none, splits, splits + dividends), delisting detection with auto-close
- **Dividend reinvestment** — DRIP support for automatic reinvestment
- **Parameter optimization** — Grid search and Bayesian optimization (scikit-optimize) with parallel execution
- **Walk-forward analysis** — Rolling or anchored in-sample/out-of-sample windows, purged K-fold cross-validation
- **Overfitting detection** — Deflated Sharpe Ratio (Bailey & Lopez de Prado), permutation significance tests
- **Multi-strategy portfolios** — Run multiple strategies simultaneously with capital allocation and attribution
- **Monte Carlo simulation** — Statistical confidence intervals on backtest results
- **Result persistence** — Save/load backtest results to disk, side-by-side comparison
- **Stress testing** — Run strategy across historical crisis periods (dot-com, GFC, COVID, etc.)
- **Target-weight rebalancing** — Rebalance to target portfolio weights on configurable schedules (daily/weekly/monthly/quarterly)
- **Multi-market universes** — US, Canadian, or combined ticker universes (index constituents or full market)
- **Data caching** — Parquet-based local cache with Yahoo Finance as default data source
- **Professional analytics** — CAGR, Sharpe, Sortino, max drawdown, VaR, CVaR, Omega, Treynor, MAE/MFE, turnover, TCA, capacity estimation, monthly returns, HTML tearsheets, regime analysis, signal decay, correlation/concentration metrics
- **TOML config files** — Load all settings from a TOML file, with CLI overrides
- **Progress bars** — tqdm progress indicators for backtest loops and parameter sweeps

## Requirements

- Python 3.11+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# For Bayesian optimization (optional)
pip install -e ".[optimize]"
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

Risk 1% of capital per trade. Stop placed 2x ATR from entry, target at 4x ATR.

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

### 14. Drawdown kill switch with risk reporting

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --max-drawdown 0.15 \
  --report-risk
```

Halts all trading if portfolio drawdown exceeds 15%. The `--report-risk` flag adds VaR and CVaR to the output.

---

### 15. Square-root market impact slippage

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --slippage-model sqrt \
  --slippage-impact 0.1
```

Applies Almgren-Chriss square-root market impact: larger orders incur superlinear slippage based on order size relative to volume.

---

### 16. Partial fills with volume constraints

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT \
  --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --max-volume-pct 0.05 \
  --partial-fill-policy requeue
```

Limits fills to 5% of daily volume. Unfilled remainder is requeued for the next day (`cancel` discards it instead).

---

### 17. Fill price model — VWAP fills

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --fill-price vwap
```

Fill market orders at estimated VWAP instead of the open. Options: `open` (default), `close`, `vwap`, `random` (uniform within day's range).

---

### 18. Kelly criterion position sizing

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --position-sizing kelly \
  --kelly-fraction 0.5
```

Sizes positions using half-Kelly criterion. Falls back to fixed fractional when win rate / payoff data is unavailable.

---

### 19. Exposure limits and sector constraints

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,JPM,GS,XOM \
  --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --max-gross-exposure 1.5 \
  --max-net-exposure 1.0 \
  --max-sector-exposure 0.30 \
  --sector-map sectors.csv
```

Limits gross exposure to 150% of equity, net exposure to 100%, and any single sector to 30%. The sector map CSV has columns `symbol,sector`.

---

### 20. Portfolio volatility targeting

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --target-portfolio-vol 0.10 \
  --portfolio-vol-lookback 60
```

Scales all position sizes to target 10% annualized portfolio volatility using a 60-day lookback. Scale factor is capped at 2x.

---

### 21. Transaction cost analysis and trade analytics

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --report-tca \
  --report-mae-mfe
```

Prints transaction cost analysis (turnover, cost attribution, capacity estimate) and per-trade MAE/MFE (Maximum Adverse/Favorable Excursion).

---

### 22. Overfitting detection

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --trials 50 \
  --permutation-test 1000
```

Computes the Deflated Sharpe Ratio (adjusted for multiple testing with 50 trials) and runs a 1000-permutation significance test on the strategy's returns.

---

### 23. Bayesian optimization with parallel workers

```bash
backtester optimize \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2001-01-01 --end 2015-12-31 \
  --grid '{"sma_fast":[20,200],"sma_slow":[100,400]}' \
  --optimize-method bayesian \
  --n-trials 50 \
  --workers 4
```

Uses Gaussian process optimization (scikit-optimize) to explore the parameter space in 50 trials. Grid bounds define the search range. `--workers 4` parallelizes grid search across 4 processes.

---

### 24. Purged K-fold cross-validation

```bash
backtester walk-forward \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2001-01-01 --end 2023-12-31 \
  --grid '{"sma_fast":[20,50,100],"sma_slow":[100,200,300]}' \
  --cv-method purged_kfold \
  --purge-days 10 \
  --embargo-days 5
```

Splits the date range into K contiguous folds with purge gaps (10 days) and embargo periods (5 days) around test boundaries to prevent data leakage.

---

### 25. Survivorship-bias-free universe

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,ENRON \
  --benchmark SPY \
  --start 2000-01-01 --end 2005-12-31 \
  --universe-file historical_members.csv
```

The universe file (CSV with `date,symbol` columns) defines which tickers are tradeable on each date. Prevents trading delisted stocks outside their membership window.

---

### 26. CSV/Parquet data sources

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2020-12-31 \
  --data-source csv \
  --data-path ./my_data/
```

Loads OHLCV data from `./my_data/SPY.csv` instead of Yahoo Finance. Parquet files work the same way with `--data-source parquet`. CSV files must have columns: Date, Open, High, Low, Close, Volume.

---

### 27. Save results and compare runs

```bash
# Save two parameter runs
backtester run \
  --strategy sma_crossover --tickers SPY --benchmark SPY \
  --start 2010-01-01 --end 2020-12-31 \
  --params '{"sma_fast":50,"sma_slow":200}' \
  --save-results results/run_50_200

backtester run \
  --strategy sma_crossover --tickers SPY --benchmark SPY \
  --start 2010-01-01 --end 2020-12-31 \
  --params '{"sma_fast":100,"sma_slow":300}' \
  --save-results results/run_100_300

# Compare side-by-side
backtester compare results/run_50_200 results/run_100_300
```

Saves config, metrics, equity curve, and trades as JSON/Parquet. The `compare` command prints a side-by-side metric table.

---

### 28. Stress testing across historical crises

```bash
backtester stress-test \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 1998-01-01 --end 2023-12-31 \
  --scenario dot_com_crash \
  --scenario gfc_2008 \
  --scenario covid_crash
```

Runs the strategy across specific crisis periods and reports per-scenario metrics. Built-in scenarios: `dot_com_crash`, `gfc_2008`, `flash_crash_2010`, `taper_tantrum_2013`, `china_deval_2015`, `volmageddon_2018`, `covid_crash`, `rate_hike_2022`.

---

### 29. Monthly rebalance schedule

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN \
  --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --rebalance-schedule monthly
```

Generates signals only on month boundaries. On non-rebalance days, the engine still processes fills and updates prices but skips signal generation. Options: `daily` (default), `weekly`, `monthly`, `quarterly`.

---

### 30. LIFO lot accounting

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 \
  --lot-method lifo
```

Sells the most recently purchased lots first. Options: `fifo` (default), `lifo`, `highest_cost`, `lowest_cost`.

---

### 31. TOML config file

```bash
backtester run --config-file my_backtest.toml --tickers AAPL
```

Load all settings from a TOML file. CLI arguments override file values. Example `my_backtest.toml`:

```toml
strategy = "sma_crossover"
benchmark = "SPY"
start = "2010-01-01"
end = "2023-12-31"
cash = 50000
max_positions = 10
stop_loss_pct = 0.05
take_profit_pct = 0.20
fee_model = "composite_us"
```

---

## Advanced Examples

The examples below cover Python-API-only features (limit orders, multi-timeframe strategies, multi-strategy portfolios, cross-sectional strategies, fundamental data).

### 32. Short selling with stops and regime filter

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

### 33. Limit orders — use Signal dataclass in custom strategies

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

### 34. Stop orders via Signal dataclass

```python
from backtester.strategies.base import Signal
from backtester.types import SignalAction, OrderType

class MyStrategy(Strategy):
    def generate_signals(self, row, portfolio_state):
        return Signal(
            action=SignalAction.SELL,
            order_type=OrderType.STOP,
            stop_price=row["Close"] * 0.95,  # trigger at 5% below close
        )
```

Stop orders trigger when the day's Low crosses the stop price (for sells) or High crosses it (for buys). `STOP_LIMIT` orders require both a trigger and a fill price.

---

### 35. Cross-sectional strategy — rank and select from universe

```python
from backtester.strategies.base import CrossSectionalStrategy
from backtester.types import SignalAction

class MomentumRank(CrossSectionalStrategy):
    name = "momentum_rank"

    def compute_indicators(self, df, timeframe_data=None):
        df = df.copy()
        df["return_60d"] = df["Close"].pct_change(60)
        return df

    def rank_universe(self, bar_data, positions, portfolio_state, benchmark_row=None):
        scores = {}
        for symbol, row in bar_data.items():
            ret = row.get("return_60d")
            if ret is not None and not pd.isna(ret):
                scores[symbol] = ret

        top = self.top_n(scores, 5)
        signals = [(s, SignalAction.BUY) for s in top]

        # Sell anything no longer in top 5
        for symbol in positions:
            if symbol not in top:
                signals.append((symbol, SignalAction.SELL))

        return signals
```

Cross-sectional strategies see all symbols at once and return a list of `(symbol, signal)` tuples. The engine dispatches to `rank_universe()` instead of the per-symbol `generate_signals()` loop.

---

### 36. Fundamental data in strategies

```python
class ValueStrategy(Strategy):
    name = "value"

    def generate_signals(self, row, portfolio_state):
        pe = self.get_fundamental(row.name if hasattr(row, 'name') else "AAPL", "PE", row["date"])
        if pe is not None and pe < 15:
            return SignalAction.BUY
        return SignalAction.HOLD
```

```bash
backtester run \
  --strategy value \
  --tickers AAPL,MSFT \
  --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --fundamental-data fundamentals.csv
```

The fundamental data CSV has columns `date,symbol,field,value`. Point-in-time lookups use binary search — data reported on March 15 is not visible before March 15.

---

### 37. Target-weight rebalancing strategy

```python
class EqualWeight(Strategy):
    name = "equal_weight"

    def target_weights(self, bar_data, portfolio_state, benchmark_row=None):
        symbols = list(bar_data.keys())
        if not symbols:
            return None
        w = 1.0 / len(symbols)
        return {s: w for s in symbols}

    def generate_signals(self, row, portfolio_state):
        return SignalAction.HOLD  # not used when target_weights returns non-None
```

When a strategy returns target weights, the engine computes delta orders to rebalance (sells first to free cash, then buys). Combine with `--rebalance-schedule monthly` for calendar-based rebalancing.

---

### 38. Realistic fee models — percentage, tiered, and regulatory fees

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

### 39. Multi-timeframe strategy — weekly trend + daily entry

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

### 40. Multi-strategy portfolio — run strategies side by side

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
| `--config-file` | Load config from TOML file (CLI overrides file values) | disabled |
| **Fees & Slippage** | | |
| `--fee-model` | Fee model: `per_trade`, `percentage`, `composite_us` | `per_trade` |
| `--fee` | Fee amount: dollars for `per_trade`, basis points for `percentage`/`composite_us` | `0.05` |
| `--slippage-model` | Slippage model: `fixed`, `volume`, `sqrt` | `fixed` |
| `--slippage-bps` | Slippage in basis points (for `fixed` model) | `10.0` |
| `--slippage-impact` | Impact factor for `volume`/`sqrt` slippage model | `0.1` |
| **Position Sizing** | | |
| `--position-sizing` | Sizing model: `fixed_fractional`, `atr`, `vol_parity`, `kelly`, `risk_parity` | `fixed_fractional` |
| `--risk-pct` | Risk per trade for `atr` sizer (e.g. 0.01 = 1%) | `0.01` |
| `--atr-multiple` | ATR multiple for stop distance in `atr` sizer | `2.0` |
| `--vol-target` | Target annualized volatility for `vol_parity` sizer | `0.10` |
| `--vol-lookback` | Lookback window (days) for `vol_parity` sizer | `20` |
| `--kelly-fraction` | Fraction of Kelly for `kelly` sizer (0.5 = half-Kelly) | `0.5` |
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
| **Data** | | |
| `--data-source` | Data source: `yahoo`, `csv`, `parquet` | `yahoo` |
| `--data-path` | Path to data directory for CSV/Parquet sources | disabled |
| `--universe-file` | CSV with `date,symbol` for historical universe membership | disabled |
| `--adjust-prices` | Price adjustment: `none`, `splits`, `splits_and_dividends` | `splits` |
| `--fundamental-data` | Path to fundamental data CSV | disabled |
| **Execution Realism** | | |
| `--fill-price` | Fill price model: `open`, `close`, `vwap`, `random` | `open` |
| `--max-volume-pct` | Max fraction of daily volume fillable | `0.10` |
| `--partial-fill-policy` | Unfilled remainder: `cancel`, `requeue` | `cancel` |
| `--lot-method` | Lot accounting: `fifo`, `lifo`, `highest_cost`, `lowest_cost` | `fifo` |
| **Risk Controls** | | |
| `--max-drawdown` | Max drawdown before halting (e.g. 0.10 = 10%) | disabled |
| `--max-sector-exposure` | Max weight per sector (e.g. 0.30 = 30%) | disabled |
| `--sector-map` | CSV with `symbol,sector` columns | disabled |
| `--max-gross-exposure` | Max gross exposure as fraction of equity | disabled |
| `--max-net-exposure` | Max net exposure as fraction of equity | disabled |
| `--target-portfolio-vol` | Target annualized portfolio volatility | disabled |
| `--portfolio-vol-lookback` | Lookback days for portfolio vol computation | `60` |
| **Portfolio** | | |
| `--drip` | Enable dividend reinvestment (flag) | `false` |
| `--rebalance-schedule` | Signal frequency: `daily`, `weekly`, `monthly`, `quarterly` | `daily` |
| **Output & Analytics** | | |
| `--export-log` | Export activity log to CSV file path | disabled |
| `--tearsheet` | Generate HTML tearsheet at file path | disabled |
| `--save-results` | Save results (config, metrics, equity, trades) to directory | disabled |
| `--report-regime` | Print regime performance breakdown (flag) | `false` |
| `--report-signal-decay` | Print signal decay analysis (flag) | `false` |
| `--report-correlation` | Print correlation matrix (flag) | `false` |
| `--report-concentration` | Print portfolio concentration / HHI (flag) | `false` |
| `--report-risk` | Print VaR/CVaR risk metrics (flag) | `false` |
| `--report-mae-mfe` | Print per-trade MAE/MFE analysis (flag) | `false` |
| `--report-tca` | Print transaction cost analysis (flag) | `false` |
| `--trials` | Number of trials for Deflated Sharpe Ratio | disabled |
| `--permutation-test` | Number of permutations for significance test | disabled |
| `--monte-carlo-runs` | Number of Monte Carlo simulations | `1000` |

### `backtester optimize`

Run a parameter grid search or Bayesian optimization.

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

Supports all common options from `backtester run`, plus:

| Option | Description | Default |
|---|---|---|
| `--grid` | Parameter grid as JSON (e.g. `{"sma_fast":[50,100]}`) | **required** |
| `--metric` | Metric to optimize | `sharpe_ratio` |
| `--optimize-method` | Method: `grid`, `bayesian` | `grid` |
| `--n-trials` | Number of trials for Bayesian optimization | `50` |
| `--workers` | Parallel workers for grid search | `1` |

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

Supports all common options from `backtester run`, plus:

| Option | Description | Default |
|---|---|---|
| `--grid` | Parameter grid as JSON | **required** |
| `--is-months` | In-sample window length in months | `12` |
| `--oos-months` | Out-of-sample window length in months | `3` |
| `--anchored` | Use expanding (anchored) in-sample window (flag) | `false` |
| `--metric` | Metric to optimize | `sharpe_ratio` |
| `--cv-method` | Cross-validation: `walkforward`, `purged_kfold` | `walkforward` |
| `--purge-days` | Purge gap in days for purged K-fold | `10` |
| `--embargo-days` | Embargo gap in days for purged K-fold | `5` |

### `backtester stress-test`

Run strategy across historical crisis periods.

```bash
backtester stress-test \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 1998-01-01 --end 2023-12-31 \
  --scenario dot_com_crash \
  --scenario gfc_2008
```

Supports all common options from `backtester run`, plus:

| Option | Description | Default |
|---|---|---|
| `--scenario` | Stress scenario name (repeatable) | all scenarios |

Built-in scenarios: `dot_com_crash`, `gfc_2008`, `flash_crash_2010`, `taper_tantrum_2013`, `china_deval_2015`, `volmageddon_2018`, `covid_crash`, `rate_hike_2022`.

### `backtester compare`

Compare saved backtest results side-by-side.

```bash
backtester compare results/run_a results/run_b
```

Prints a metric comparison table across two or more saved result directories.

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
  result.py           — BacktestResult with save/load/compare
  data/
    manager.py        — DataManager, resample_ohlcv (daily -> weekly/monthly)
    cache.py          — ParquetCache for local OHLCV storage
    calendar.py       — TradingCalendar (NYSE-aware)
    sources/          — DataSource ABC, YahooDataSource
    csv_source.py     — CSVDataSource (load from CSV files)
    parquet_source.py — ParquetDataSource (load from Parquet files)
    fundamental.py    — FundamentalDataManager (point-in-time lookups)
    universe.py       — Universe provider (S&P 500, TSX), HistoricalUniverse
  strategies/
    base.py           — Strategy ABC, CrossSectionalStrategy ABC, Signal dataclass
    registry.py       — Strategy auto-discovery and lookup
    indicators.py     — 18 vectorized indicator functions
    sma_crossover.py  — SMA crossover strategy
    rule_based.py     — Rule-based DSL strategy
  execution/
    broker.py         — SimulatedBroker (market/limit/stop/bracket fills, partial fills)
    slippage.py       — FixedSlippage, VolumeSlippage, SqrtImpactSlippage
    fees.py           — PerTradeFee, PercentageFee, TieredFee, SECFee, TAFFee, CompositeFee
    position_sizing.py — FixedFractional, ATRSizer, VolatilityParity, KellyCriterionSizer, RiskParitySizer
    stops.py          — StopManager (long + short stop/take-profit/trailing)
  portfolio/
    portfolio.py      — Portfolio state, margin tracking, rebalance order computation
    position.py       — Position with FIFO/LIFO/cost-based lots (long + short)
    order.py          — Order (market/limit/stop/bracket, DAY/GTC), Fill, Trade
  analytics/
    metrics.py        — CAGR, Sharpe, Sortino, drawdown, VaR, CVaR, Omega, Treynor
    montecarlo.py     — Monte Carlo simulation
    report.py         — Console report output
    calendar.py       — Monthly returns, drawdown periods
    tearsheet.py      — Self-contained HTML tearsheet with embedded charts
    correlation.py    — Correlation matrix, HHI, sector exposure
    signal_decay.py   — Signal return attribution, optimal holding period
    regime.py         — Bull/bear/sideways classification, per-regime metrics
    overfitting.py    — Deflated Sharpe Ratio, permutation significance test
    trade_analysis.py — Per-trade MAE/MFE analysis
    tca.py            — Turnover, cost attribution, capacity estimation
    stress.py         — Historical stress test scenarios
  research/
    optimizer.py      — Grid search, Bayesian optimization, parallel execution
    walk_forward.py   — Walk-forward IS/OOS analysis
    cross_validation.py — Purged K-fold cross-validation
    multi_strategy.py — Multi-strategy allocation, combined equity, attribution
```

## Testing

```bash
pytest tests/ -v          # full suite (588 tests)
```

Run a single test:

```bash
pytest tests/test_portfolio.py::TestPosition::test_sell_fifo -v
```
