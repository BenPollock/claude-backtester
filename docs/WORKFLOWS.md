# Backtesting Workflows

Practical step-by-step checklists with exact CLI commands. Each workflow answers a specific research question, ordered from simplest to most complex.

For the *why* behind each concept, see [CONCEPTS.md](CONCEPTS.md).

---

## Workflow 1: Run Your First Backtest

> *"I have a strategy idea — how do I run my first backtest?"*

### Steps

**1. See what strategies are available:**

```bash
backtester list-strategies
```

**2. Run a basic backtest with a single ticker:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --params '{"sma_fast": 50, "sma_slow": 200}'
```

**3. Read the output.** The standard report prints:
- **CAGR** — Compound annual growth rate (annualized return)
- **Sharpe ratio** — Return per unit of risk (> 1.0 is good)
- **Max drawdown** — Worst peak-to-trough decline
- **Trade count** — How active the strategy is

**4. Export the trade log to spot-check entries:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --export-log trades.csv
```

Open `trades.csv` and verify: Are entries happening at the open of the day *after* the signal? Are stop-outs happening at reasonable prices?

**5. Add multiple tickers to test portfolio behavior:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}'
```

**6. Generate an HTML tearsheet for a visual summary:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --tearsheet tearsheet.html
```

Open `tearsheet.html` in a browser. It shows equity curves, drawdown chart, monthly returns heatmap, and a trade table.

### What to look for

- Is the Sharpe ratio above 0.5? Below that, the strategy may not be worth pursuing.
- Is the max drawdown tolerable? A 40% drawdown means you need 67% to recover.
- Does the equity curve beat the benchmark? If not, you're adding complexity for no value.

---

## Workflow 2: Check If Your Result Is Real or Overfitted

> *"My backtest looks good — how do I know if it's real?"*

### Steps

**1. Record how many parameter combinations you tested.**
If you tried 5 values of sma_fast and 5 values of sma_slow, that's 25 combinations.

**2. Run with Deflated Sharpe Ratio and permutation test:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --trials 25 \
  --permutation-test 1000
```

**3. Run Monte Carlo simulation for confidence intervals:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --trials 25 \
  --permutation-test 1000 \
  --monte-carlo-runs 1000
```

### How to interpret the results

| Metric | Pass | Fail |
|--------|------|------|
| Deflated Sharpe Ratio | > 0 | ≤ 0 (likely noise) |
| Permutation p-value | < 0.05 | ≥ 0.05 (can't reject luck) |
| Monte Carlo P5 final equity | > starting cash | < starting cash (possible loss) |

**All three should pass.** If the Deflated Sharpe drops to zero after accounting for the 25 combinations you tested, your observed Sharpe is likely a product of selection bias, not genuine edge.

---

## Workflow 3: Validate a Strategy Rigorously

> *"How do I validate a strategy before trusting it?"*

### Steps

**1. Optimize parameters on in-sample data with grid search:**

```bash
backtester optimize \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2010-01-01 \
  --end 2018-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --grid '{"sma_fast": [20, 50, 100, 150], "sma_slow": [150, 200, 250, 300]}' \
  --metric sharpe_ratio
```

Note the best parameters from the output. **Do not look at 2019-2023 data yet.**

**2. Test the best parameters on out-of-sample data:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2019-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --trials 16
```

Replace the params with whatever the optimizer found. Set `--trials 16` (the 16 combinations in your 4×4 grid).

**3. Run walk-forward analysis for rolling IS/OOS validation:**

```bash
backtester walk-forward \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2010-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --grid '{"sma_fast": [20, 50, 100, 150], "sma_slow": [150, 200, 250, 300]}' \
  --is-months 12 \
  --oos-months 3 \
  --metric sharpe_ratio
```

**4. Run purged K-fold cross-validation for an additional check:**

```bash
backtester walk-forward \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2010-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --grid '{"sma_fast": [20, 50, 100, 150], "sma_slow": [150, 200, 250, 300]}' \
  --cv-method purged_kfold \
  --purge-days 10 \
  --embargo-days 5 \
  --metric sharpe_ratio
```

**5. Run stress tests across historical crises:**

```bash
backtester stress-test \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --scenario gfc_2008 \
  --scenario covid_crash
```

### What to look for

| Check | Healthy | Red flag |
|-------|---------|----------|
| OOS Sharpe vs IS Sharpe | Within 50% (degradation ratio > 0.5) | OOS Sharpe < 50% of IS |
| Walk-forward consistency | Positive OOS in most windows | Negative OOS in multiple windows |
| K-fold average | Positive average Sharpe | Negative average Sharpe |
| Stress test drawdown | Survivable (< 30%) | Would blow up the account (> 50%) |

---

## Workflow 4: Measure and Control Risk

> *"How do I measure and control risk in my strategy?"*

### Steps

**1. Run with risk reporting to see VaR and CVaR:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --report-risk \
  --save-results results/baseline
```

**2. Add stop-loss and take-profit:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --stop-loss 0.08 \
  --take-profit 0.20 \
  --trailing-stop 0.10 \
  --report-risk \
  --save-results results/with_stops
```

**3. Calibrate stops with MAE/MFE analysis:**

Run a backtest *without* stops to see the natural excursion of your trades:

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --report-mae-mfe
```

Use the MAE distribution to set stop-loss (just beyond where most winners dip) and MFE to set take-profit (near where most winners peak).

**4. Add a drawdown kill switch:**

```bash
  --max-drawdown 0.20
```

**5. Add exposure limits:**

```bash
  --max-sector-exposure 0.30 \
  --sector-map sectors.csv \
  --max-gross-exposure 1.0 \
  --max-net-exposure 0.80
```

**6. Add volatility targeting:**

```bash
  --target-portfolio-vol 0.15 \
  --portfolio-vol-lookback 60
```

**7. Add regime filtering:**

```bash
  --regime-benchmark SPY \
  --regime-fast 50 \
  --regime-slow 200 \
  --regime-condition fast_above_slow
```

**8. Compare all versions:**

```bash
backtester compare results/baseline results/with_stops
```

### Build up incrementally

Add one risk control at a time and compare. This lets you see the marginal impact of each control. Some controls improve risk-adjusted returns (regime filter, volatility targeting). Others reduce returns but reduce drawdowns more (stops, drawdown kill switch). The goal is finding the combination that matches your risk tolerance.

---

## Workflow 5: Test Strategy Viability at Scale

> *"How do I know if my strategy is viable at scale?"*

### Steps

**1. Run with realistic US equity costs:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --fee-model composite_us \
  --slippage-model sqrt \
  --slippage-impact 0.1 \
  --max-volume-pct 0.05 \
  --partial-fill-policy requeue \
  --report-tca \
  --save-results results/realistic_100k
```

**2. Run at different capital levels to find the capacity ceiling:**

```bash
# $10K
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 10000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --fee-model composite_us \
  --slippage-model sqrt \
  --max-volume-pct 0.05 \
  --report-tca \
  --save-results results/realistic_10k

# $1M
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 1000000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --fee-model composite_us \
  --slippage-model sqrt \
  --max-volume-pct 0.05 \
  --report-tca \
  --save-results results/realistic_1m

# $10M
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 10000000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --fee-model composite_us \
  --slippage-model sqrt \
  --max-volume-pct 0.05 \
  --report-tca \
  --save-results results/realistic_10m
```

**3. Compare across capital levels:**

```bash
backtester compare results/realistic_10k results/realistic_100k results/realistic_1m results/realistic_10m
```

**4. Check signal decay to understand alpha half-life:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --report-signal-decay
```

### What to look for

| Signal | Meaning |
|--------|---------|
| Sharpe drops significantly at $1M+ | Strategy has limited capacity |
| TCA shows turnover > 500% annual | Too much trading; costs will dominate |
| Partial fills increase at higher cash | Liquidity is a binding constraint |
| Signal decay shows alpha peaks at T+3 | Strategy should exit sooner; long holding periods waste the edge |

---

## Workflow 6: Compare Two Strategies

> *"How do I compare two strategies against each other?"*

### Steps

**1. Run each strategy with identical settings (except strategy and params):**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --fee-model composite_us \
  --slippage-model sqrt \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --save-results results/strategy_a

backtester run \
  --strategy rule_based \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --fee-model composite_us \
  --slippage-model sqrt \
  --params '{"rules": "rsi_oversold"}' \
  --save-results results/strategy_b
```

**2. Compare side-by-side:**

```bash
backtester compare results/strategy_a results/strategy_b
```

**3. Check correlation between the equity curves:**

Run each strategy with `--report-correlation` to see how correlated the holdings are. Low correlation between strategies means they diversify each other.

**4. Evaluate combining them in a multi-strategy portfolio:**

If the two strategies have low correlation, combining them should improve the overall Sharpe ratio. Use the Python API:

```python
from backtester.research.multi_strategy import (
    MultiStrategyConfig, StrategyAllocation, run_multi_strategy,
    print_multi_strategy_report,
)
from backtester.config import BacktestConfig

base_config = BacktestConfig(
    strategy_name="sma_crossover",  # placeholder, overridden per allocation
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"],
    benchmark="SPY",
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    starting_cash=100000,
    max_positions=5,
    max_alloc_pct=0.20,
    fee_model="composite_us",
    slippage_model="sqrt",
)

multi_config = MultiStrategyConfig(
    allocations=(
        StrategyAllocation("sma_crossover", {"sma_fast": 50, "sma_slow": 200}, weight=0.5),
        StrategyAllocation("rule_based", {"rules": "rsi_oversold"}, weight=0.5),
    ),
    base_config=base_config,
)

result = run_multi_strategy(multi_config)
print_multi_strategy_report(result)
```

### What to look for

| Metric | Better strategy wins on |
|--------|------------------------|
| Sharpe ratio | Risk-adjusted return |
| Max drawdown | Worst-case survivability |
| Calmar ratio | Return per unit of drawdown |
| Win rate + profit factor | Trade-level consistency |
| Combined Sharpe (multi-strategy) | Should exceed either individual Sharpe |

---

## Workflow 7: Optimize Parameters Without Overfitting

> *"How do I find the best parameters without fooling myself?"*

### Steps

**1. Define a sensible parameter grid — keep it small:**

```bash
# Good: 4 × 4 = 16 combos. Manageable, interpretable.
--grid '{"sma_fast": [20, 50, 100, 150], "sma_slow": [150, 200, 250, 300]}'

# Bad: 10 × 10 × 5 = 500 combos. Overfitting risk skyrockets.
--grid '{"sma_fast": [10,20,30,40,50,60,70,80,90,100], "sma_slow": [100,120,140,160,180,200,220,240,260,280], "threshold": [0.01,0.02,0.03,0.04,0.05]}'
```

**2. Run grid search on in-sample data only:**

```bash
backtester optimize \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2010-01-01 \
  --end 2018-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --grid '{"sma_fast": [20, 50, 100, 150], "sma_slow": [150, 200, 250, 300]}' \
  --metric sharpe_ratio
```

**3. For large search spaces, use Bayesian optimization** (fewer evaluations needed):

```bash
backtester optimize \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2010-01-01 \
  --end 2018-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --grid '{"sma_fast": [20, 150], "sma_slow": [150, 300]}' \
  --optimize-method bayesian \
  --n-trials 50 \
  --metric sharpe_ratio
```

**4. Validate best params with walk-forward analysis:**

```bash
backtester walk-forward \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2010-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --grid '{"sma_fast": [20, 50, 100, 150], "sma_slow": [150, 200, 250, 300]}' \
  --is-months 12 \
  --oos-months 3 \
  --metric sharpe_ratio
```

**5. Check degradation ratio:**

The walk-forward output reports the average IS Sharpe, average OOS Sharpe, and degradation ratio (OOS / IS).

| Degradation ratio | Interpretation |
|-------------------|----------------|
| > 0.7 | Excellent generalization |
| 0.5 – 0.7 | Normal. Some overfitting, but strategy has edge |
| 0.3 – 0.5 | Concerning. Consider simpler model |
| < 0.3 | Strategy memorized in-sample noise |

**6. Run permutation test on the final result:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2019-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --trials 16 \
  --permutation-test 1000
```

**7. Check parameter robustness — test neighboring values:**

If sma_fast=50 works but sma_fast=45 and sma_fast=55 don't, you've overfit to a specific value. Good parameters have a "plateau" — nearby values should produce similar results. The grid search results table shows this.

---

## Workflow 8: Pre-Flight Checklist Before Live Trading

> *"What should I check before considering a strategy ready?"*

### The Checklist

Run through each item. A strategy that passes all 10 checks has been rigorously validated.

- [ ] **1. Data integrity**
  - Using survivorship-bias-free universe (`--universe-file`)
  - Prices adjusted for splits at minimum (`--adjust-prices splits`)
  - No lookahead (built-in T/T+1 model enforces this)

- [ ] **2. Execution realism**
  - Realistic fee model (`--fee-model composite_us`)
  - Volume-aware slippage (`--slippage-model sqrt`)
  - Partial fill constraints (`--max-volume-pct 0.05`)

- [ ] **3. Statistical validity**
  - Deflated Sharpe Ratio > 0 (`--trials N`)
  - Permutation test p-value < 0.05 (`--permutation-test 1000`)

- [ ] **4. Walk-forward validation**
  - Degradation ratio > 0.5
  - Consistent OOS performance across windows (no single window carrying the result)

- [ ] **5. Risk management configured**
  - Stops calibrated via MAE/MFE analysis (`--report-mae-mfe`)
  - Drawdown kill switch set (`--max-drawdown`)
  - Exposure limits in place (`--max-alloc`, `--max-sector-exposure`)

- [ ] **6. Stress tested**
  - Survives GFC 2008 (`--scenario gfc_2008`)
  - Survives COVID crash (`--scenario covid_crash`)
  - Survives at least 2 other scenarios

- [ ] **7. Capacity verified**
  - TCA report shows strategy viable at your intended capital (`--report-tca`)
  - Sharpe doesn't collapse at your target cash level

- [ ] **8. Parameter robustness**
  - Neighboring parameter values produce similar results
  - Strategy doesn't depend on a single "magic" number

- [ ] **9. Signal decay understood**
  - Alpha persists through intended holding period (`--report-signal-decay`)
  - Not holding beyond the point where returns plateau

- [ ] **10. Portfolio construction complete**
  - Position sizing model chosen (`--position-sizing`)
  - Rebalance schedule set (`--rebalance-schedule`)
  - Volatility target configured if desired (`--target-portfolio-vol`)

### The comprehensive final run

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --fee-model composite_us \
  --slippage-model sqrt \
  --slippage-impact 0.1 \
  --max-volume-pct 0.05 \
  --partial-fill-policy requeue \
  --position-sizing atr \
  --risk-pct 0.01 \
  --atr-multiple 2.0 \
  --stop-loss-atr 2.0 \
  --take-profit-atr 3.0 \
  --trailing-stop 0.10 \
  --max-drawdown 0.20 \
  --max-sector-exposure 0.30 \
  --target-portfolio-vol 0.15 \
  --regime-benchmark SPY \
  --regime-fast 50 \
  --regime-slow 200 \
  --rebalance-schedule weekly \
  --adjust-prices splits_and_dividends \
  --trials 16 \
  --permutation-test 1000 \
  --monte-carlo-runs 1000 \
  --report-risk \
  --report-mae-mfe \
  --report-tca \
  --report-signal-decay \
  --report-concentration \
  --report-correlation \
  --report-regime \
  --tearsheet final_tearsheet.html \
  --export-log final_trades.csv
```

---

## Workflow 9: Build a Regime-Aware Strategy

> *"How do I make my strategy respect market conditions?"*

### Steps

**1. Run a baseline without regime filtering:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2005-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --save-results results/no_regime
```

**2. Add regime filter — only trade when SPY is in a bullish regime:**

```bash
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2005-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --regime-benchmark SPY \
  --regime-fast 50 \
  --regime-slow 200 \
  --regime-condition fast_above_slow \
  --report-regime \
  --save-results results/with_regime
```

**3. Compare with and without:**

```bash
backtester compare results/no_regime results/with_regime
```

**4. Experiment with different regime parameters:**

```bash
# Faster regime (more responsive, more whipsaws)
  --regime-fast 20 --regime-slow 100

# Slower regime (more lag, fewer false signals)
  --regime-fast 100 --regime-slow 300

# Inverted regime (trade only in bear markets — for short strategies)
  --regime-condition fast_below_slow
```

### What to look for

| Metric | With regime filter should show |
|--------|-------------------------------|
| Max drawdown | Smaller (avoid buying into crashes) |
| Sharpe ratio | Higher (cut losing periods) |
| Trade count | Lower (sitting out during bearish regime) |
| Exposure time | Lower (less time in market) |

The regime filter is most valuable for trend-following strategies that suffer from whipsaws in bear markets. If your strategy is mean-reverting, a regime filter might actually hurt (mean-reversion often works better during volatile markets).

---

## Workflow 10: Choose a Position Sizing Model

> *"How much should I bet on each trade?"*

### Steps

**1. Run with each sizing model, saving results for comparison:**

```bash
# Fixed Fractional (baseline) — same % of equity per trade
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --position-sizing fixed_fractional \
  --save-results results/sizing_fixed

# ATR — size based on stock volatility
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --position-sizing atr \
  --risk-pct 0.01 \
  --atr-multiple 2.0 \
  --save-results results/sizing_atr

# Volatility Parity — equalize risk contribution per position
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --position-sizing vol_parity \
  --vol-target 0.10 \
  --vol-lookback 20 \
  --save-results results/sizing_volparity

# Half-Kelly — edge-optimized sizing
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --position-sizing kelly \
  --kelly-fraction 0.5 \
  --save-results results/sizing_kelly

# Risk Parity — equal risk contribution from each position
backtester run \
  --strategy sma_crossover \
  --tickers AAPL,MSFT,GOOGL,AMZN,SPY \
  --benchmark SPY \
  --start 2015-01-01 \
  --end 2023-12-31 \
  --cash 100000 \
  --max-positions 5 \
  --max-alloc 0.20 \
  --params '{"sma_fast": 50, "sma_slow": 200}' \
  --position-sizing risk_parity \
  --save-results results/sizing_riskparity
```

**2. Compare all five:**

```bash
backtester compare \
  results/sizing_fixed \
  results/sizing_atr \
  results/sizing_volparity \
  results/sizing_kelly \
  results/sizing_riskparity
```

### Decision guide

| Model | Best for | Watch out for |
|-------|----------|---------------|
| **Fixed Fractional** | Simplicity, getting started | Ignores volatility differences between stocks |
| **ATR** | Strategies with stops | Requires ATR indicator; falls back to fixed if ATR unavailable |
| **Vol Parity** | Multi-asset portfolios | Lookback window choice affects results |
| **Half-Kelly** | Known-edge strategies | Overestimates sizing if edge estimate is wrong; use half-Kelly, not full Kelly |
| **Risk Parity** | Diversification-focused portfolios | Computationally heavier; needs sufficient history |

### Rules of thumb

- Start with **fixed fractional** to establish a baseline
- Move to **ATR** if you're using stop-losses (they pair naturally — ATR sizes the stop, stop sizes the position)
- Use **vol parity** when holding diverse assets (stocks + bonds, or large-cap + small-cap)
- Use **half-Kelly** only when you have strong confidence in your strategy's edge estimate
- Use **risk parity** for portfolio-level risk balancing across many positions
