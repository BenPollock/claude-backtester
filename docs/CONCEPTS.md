# Quantitative Backtesting Concepts

A guide for programmers who've built a backtesting system but want to think like a quant. Every concept connects to a feature you already have — the goal is understanding *why* it exists and *when* it matters.

Each section builds on the ones before it, so reading in order is recommended.

---

## Part I — Data Integrity

*What goes into your backtest determines the ceiling of what comes out. Garbage in, garbage out — and in finance, "garbage" often looks perfectly reasonable.*

---

### 1. Survivorship Bias

**The problem:** If you test a strategy on today's S&P 500 members going back to 2000, you're only testing on the *winners* — the companies that survived long enough to still be in the index. You're excluding every company that went bankrupt, got acquired, or was delisted. This systematically inflates your results because you'll never see the stocks that would have killed your returns.

**Why it matters:** Studies show survivorship bias can inflate annual returns by 1-2%. That doesn't sound like much until you realize many quant strategies only have 2-4% of genuine edge. Survivorship bias can make a losing strategy look like a winner.

**Example:** Imagine testing a "buy cheap stocks" strategy. In 2005, your strategy would have bought Lehman Brothers, Bear Stearns, and Washington Mutual — all of which went to zero. But if you test on today's stock list, those companies don't exist, so your backtest never takes those devastating losses.

**Your system features:**
- `--universe-file` — Load a historical universe CSV with `date,symbol` columns, so the backtest only sees stocks that were actually tradeable on each day
- `HistoricalUniverse` class — Manages point-in-time universe membership
- Delisting detection — Engine auto-closes positions after 5 consecutive absent trading days, simulating the real cost of holding a stock that stops trading

**Key question:** *"Why can't I just test on today's S&P 500 members back to 2000?"*

Because you'd be testing on a curated list of survivors. The honest test uses the actual members of the index *on each date*, including the ones that later failed.

---

### 2. Corporate Actions: Splits, Dividends, and Delistings

**The problem:** Companies change their share structure over time. A 2-for-1 stock split halves the price overnight — but nobody lost money. Dividends pay cash to holders, reducing the stock price by the dividend amount on the ex-date. If your data doesn't account for these events, your indicators and signals will be based on false price moves.

**Why it matters:** Apple has split 5 times since its IPO. Without adjustment, its price chart shows what looks like repeated 50-80% crashes on random days. Any moving average or momentum indicator computed on unadjusted data will generate phantom signals at every split.

**The three adjustment levels:**
1. **None** — Raw prices as-traded. Useful only for checking actual execution prices.
2. **Split-adjusted** — Prices adjusted backward for splits. Volume adjusted inversely. This is the minimum for correct indicator computation.
3. **Split-and-dividend-adjusted** — Prices further adjusted for dividend payments. Gives you true total-return data, but the historical prices no longer match what you'd have seen on your screen at the time.

**Your system features:**
- `--adjust-prices` — Choose `none`, `splits`, or `splits_and_dividends` (default: `splits`)
- `--drip` — Enable Dividend Reinvestment Plan (automatically reinvest cash dividends into more shares)
- Engine delisting detection — Auto-closes positions in stocks that stop trading

**Key question:** *"Why does my stock look like it crashed 50% on a random Tuesday?"*

That's almost certainly an unadjusted stock split. Use `--adjust-prices splits` at minimum.

---

### 3. Lookahead Bias

**The problem:** Using information that wouldn't have been available at the time of the trading decision. This is the single most common and most dangerous backtest error, because it's invisible — your code runs fine, your results look great, and you have no idea you're cheating.

**Common forms of lookahead:**
- **Same-day fills:** Seeing today's close, deciding to buy, and getting filled at today's open (which happened hours earlier)
- **Future indicators:** Computing a 20-day moving average that includes today's close, then using it to make today's trading decision
- **Leaking weekly data into daily signals:** Using the close of a weekly bar that won't be finalized until Friday to trade on Monday
- **Point-in-time violations:** Using a company's earnings report that was released after market close to trade that same day

**Your system features:**
- **T/T+1 fill model** — `fill_delay_days=1` in config. Signals are generated using day T's close; orders fill at day T+1's open. This is the most critical anti-lookahead protection.
- **Backward-looking indicators only** — `compute_indicators()` contract requires that indicators only use past and current data, never future rows
- **Frozen portfolio state** — Strategies receive immutable `PortfolioState` snapshots, preventing them from seeing or modifying the actual portfolio mid-day
- **Multi-timeframe forward-fill** — Weekly/monthly data is forward-filled onto the daily index, so Monday uses *last* Friday's weekly close, not *this* Friday's
- **Point-in-time fundamental lookups** — `get_fundamental()` uses binary search to find the most recent value available as of the query date

**Key question:** *"How does the system guarantee I'm not cheating?"*

The T/T+1 model is the main protection: you can only act on what you knew at yesterday's close, and you pay today's open price. The system enforces this structurally — it's not just a guideline, it's how the engine loop is ordered.

---

## Part II — Execution Realism

*A backtest that ignores trading costs is a fantasy. The gap between simulated and real performance is almost always negative, and it's almost always bigger than you expect.*

---

### 4. Slippage and Market Impact

**The problem:** When you submit a buy order, you push the price up slightly because you're adding demand. The bigger your order relative to the stock's daily volume, the more you move the price against yourself. This difference between your expected price and actual fill price is called slippage.

**The three models:**

| Model | Formula | When to use |
|-------|---------|-------------|
| **Fixed** | Price ± fixed basis points | Quick-and-dirty estimates, small orders |
| **Volume-proportional** | Slippage scales linearly with order size / volume | Medium orders, liquid stocks |
| **Square-root (Almgren-Chriss)** | Impact ∝ σ × √(order / volume) | Realistic institutional modeling |

The square-root model is the industry standard because market impact is empirically concave — doubling your order size doesn't double your slippage, it increases it by about √2 ≈ 41%.

**Your system features:**
- `--slippage-model` — Choose `fixed`, `volume`, or `sqrt`
- `--slippage-bps` — Basis points for the fixed model (default: 10 bps = 0.10%)
- `--slippage-impact` — Impact factor for volume and sqrt models (default: 0.1)

**Key question:** *"My backtest shows 15% CAGR — how much of that disappears once I trade real money?"*

Run the same backtest with `--slippage-model sqrt` and compare. For strategies that trade frequently or in less liquid stocks, slippage can easily consume 2-5% of annual returns.

---

### 5. Partial Fills and Liquidity Constraints

**The problem:** You can't always buy as many shares as you want. If a stock trades 100,000 shares per day and your strategy wants to buy 50,000, you'd be 50% of daily volume — which is unrealistic and would cause massive market impact. Real institutional traders typically stay below 5-10% of daily volume per order.

**Your system features:**
- `--max-volume-pct` — Maximum fraction of daily volume your order can fill (default: 0.10 = 10%). Anything above this limit is either cancelled or re-queued.
- `--partial-fill-policy` — What happens to the unfilled remainder:
  - `cancel` — Give up on the rest (conservative, avoids chasing)
  - `requeue` — Try again the next day (persistent, but may accumulate slippage)

**Key question:** *"What happens when my order is bigger than the market can handle?"*

The system caps your fill at `max_volume_pct` of the day's actual volume. The remainder follows your partial fill policy. This is critical for strategies trading small-cap or illiquid stocks.

---

### 6. Transaction Costs and Strategy Capacity

**The problem:** Every trade incurs costs. Explicit costs (commissions, regulatory fees) are obvious. Implicit costs (slippage, market impact) are harder to measure but usually larger. At some portfolio size, total trading costs exceed the strategy's alpha, and it stops being profitable.

**Cost components:**
- **Commissions** — Per-trade or per-share fees charged by your broker
- **SEC fee** — 0.00278% of sell proceeds (US equities, sell-side only)
- **TAF fee** — $0.000166 per share (FINRA Trading Activity Fee)
- **Slippage** — Implicit cost from market impact (see concept 4)

**Strategy capacity** is the maximum capital a strategy can deploy before costs overwhelm returns. A strategy that trades 100 times per year in large-cap stocks has much more capacity than one that trades 1,000 times per year in micro-caps.

**Your system features:**
- `--fee-model` — Choose `per_trade`, `percentage`, or `composite_us` (includes SEC + TAF + commission)
- `--fee` — Fee amount (dollars for per_trade, basis points for percentage/composite_us, default: $0.05)
- `--report-tca` — Transaction Cost Analysis: shows turnover rate, cost attribution breakdown, and estimates strategy capacity

**Key question:** *"How much capital can this strategy manage before costs eat the returns?"*

Use `--fee-model composite_us --report-tca` for realistic US equity costs. The TCA report shows you the capacity ceiling.

---

### 7. Order Types: Market, Limit, Stop, and Bracket

**The problem:** A market order says "fill me now at whatever price is available." A limit order says "fill me only at this price or better." Each order type trades off execution certainty against price improvement.

**The four types:**

| Type | Behavior | Use when |
|------|----------|----------|
| **Market** | Fill at next open. Guaranteed execution, uncertain price. | You want in/out immediately |
| **Limit** | Fill only if price reaches your level. Uncertain execution, guaranteed price. | You want price improvement and can wait |
| **Stop** | Becomes a market order when price hits trigger. | Protective exits (stop-loss) |
| **Stop-limit** | Becomes a limit order when price hits trigger. | Protective exits with price control |

**Time-in-force:**
- **DAY** — Order expires at end of day if not filled
- **GTC** (Good-Til-Cancelled) — Order persists across days until filled or explicitly cancelled (with optional expiry date)

**Fill logic for limit orders:** A limit BUY fills if the day's Low ≤ limit price. A limit SELL fills if the day's High ≥ limit price. This simulates the idea that at some point during the day, the price reached your level.

**Your system features:**
- `OrderType` enum — `MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT`
- `Signal` dataclass — Wraps `SignalAction` with `limit_price`, `stop_price`, `time_in_force`, `expiry_date`
- `--fill-price` — For market orders: `open` (default), `close`, `vwap`, `random`

**Key question:** *"When should I use a limit order vs. a market order in my strategy?"*

Market orders for time-sensitive signals (momentum breakouts). Limit orders for mean-reversion strategies where you want to buy on a dip and can afford to miss some fills.

---

### 8. Lot Accounting Methods

**The problem:** Suppose you buy 100 shares of AAPL at $150, then 100 more at $170. Later you sell 100 shares at $180. Did you make $30/share (selling the $150 lot) or $10/share (selling the $170 lot)? The answer depends on your lot selection method — and it changes your realized P&L.

**The four methods:**

| Method | Sells first | Effect on realized P&L |
|--------|-------------|----------------------|
| **FIFO** (First In, First Out) | Oldest lots | Tends to realize larger gains in rising markets |
| **LIFO** (Last In, First Out) | Newest lots | Tends to realize smaller gains (or losses) |
| **Highest Cost** | Most expensive lots | Minimizes realized gains (tax-efficient) |
| **Lowest Cost** | Cheapest lots | Maximizes realized gains |

**Why it matters for backtesting:** Lot method affects your realized P&L stream, which affects win rate, profit factor, and trade-level statistics. It doesn't change your total P&L (that's determined by your entries and exits), but it changes *when* gains and losses are recognized.

**Your system features:**
- `--lot-method` — Choose `fifo`, `lifo`, `highest_cost`, or `lowest_cost` (default: `fifo`)
- FIFO lot tracking in `Position` class — Each position maintains a list of lots with individual entry prices and quantities

**Key question:** *"Why does lot selection method change my backtest results even though I'm trading the same signals?"*

Total P&L is identical regardless of lot method. But per-trade realized P&L changes, which affects trade statistics like win rate and profit factor. Use the method that matches your real broker's default (usually FIFO).

---

## Part III — Risk Management

*Returns are what you hope for. Risk management is what keeps you alive long enough to get them.*

---

### 9. Drawdown and the Kill Switch

**The problem:** Drawdown is how far your portfolio has fallen from its peak. A 10% drawdown means your $100K portfolio is now worth $90K. The psychological and financial danger of drawdowns is nonlinear — a 50% drawdown requires a 100% return to recover, and few strategies (or traders) survive that.

**Drawdown math:**

| Drawdown | Return needed to recover |
|----------|------------------------|
| 10% | 11.1% |
| 20% | 25.0% |
| 30% | 42.9% |
| 50% | 100.0% |
| 75% | 300.0% |

A **kill switch** (maximum drawdown limit) automatically halts trading when losses exceed a threshold. It's the circuit breaker that prevents catastrophic loss, even if your strategy is theoretically sound but going through an anomalous bad patch.

**Your system features:**
- `--max-drawdown` — Halt all trading when portfolio drawdown exceeds this fraction (e.g., 0.20 = 20%)
- `max_drawdown` metric — Reports worst peak-to-trough decline
- `max_drawdown_duration` metric — Reports longest time (in days) spent below previous peak
- Equity curve tracking — Portfolio records daily equity snapshots for drawdown computation

**Key question:** *"How do I prevent a bad streak from wiping out my account?"*

Set `--max-drawdown 0.20` to kill the strategy at a 20% drawdown. Choose a level that's uncomfortable but survivable — tight enough to protect capital, loose enough that normal strategy volatility doesn't trigger it.

---

### 10. Stop-Loss, Take-Profit, and Trailing Stops

**The problem:** Without pre-defined exit rules, every losing position becomes a judgment call. "Should I hold? It might come back." Systematic stops remove this decision by defining exit rules before the trade begins.

**The three stop types:**

| Stop | Triggers when | Purpose |
|------|--------------|---------|
| **Stop-loss** | Price falls X% from entry | Limit losses per position |
| **Take-profit** | Price rises X% from entry | Lock in gains at target |
| **Trailing stop** | Price falls X% from *highest point since entry* | Let winners run, then exit when trend reverses |

**ATR-based stops:** Instead of fixed percentages, you can size stops relative to the stock's recent volatility (ATR = Average True Range). A stock that moves 3% per day needs a wider stop than one that moves 0.5%. ATR stops automatically adapt to the security's volatility.

**How stops work in the system:** Stops bypass the normal T/T+1 order flow. They execute on the same day using intraday High/Low prices. If a stock hits your stop-loss intraday, you're out immediately — you don't wait until tomorrow. This is more realistic because real stop orders execute in real-time.

**Your system features:**
- `--stop-loss` / `--take-profit` / `--trailing-stop` — Fixed percentage stops
- `--stop-loss-atr` / `--take-profit-atr` — ATR-based stops (e.g., `--stop-loss-atr 2.0` = 2× ATR below entry)
- `StopConfig` dataclass — Bundles all stop parameters
- Same-day execution — Stops check the intraday High/Low range, not just the close

**Key question:** *"How tight should my stops be?"*

Too tight and you get "stopped out" constantly by normal volatility. Too loose and they don't protect you. The answer comes from MAE/MFE analysis (next concept).

---

### 11. MAE / MFE and Stop Calibration

**The problem:** Most traders set stops based on gut feel ("5% seems reasonable"). MAE/MFE analysis gives you an empirical basis for stop placement by examining what actually happened during your historical trades.

**Definitions:**
- **Maximum Adverse Excursion (MAE)** — The worst unrealized loss during a trade, measured from entry to the lowest point before exit
- **Maximum Favorable Excursion (MFE)** — The best unrealized gain during a trade, measured from entry to the highest point before exit

**How to use them:**

If your MAE analysis shows that 90% of your winning trades never dip more than 3% below entry, then a 3% stop-loss will protect you from losers while rarely stopping out winners.

If your MFE analysis shows that winning trades typically peak 8% above entry before pulling back, then an 8% take-profit or a trailing stop around that level captures most of the available gain.

**Efficiency ratio** (MFE/MAE) tells you how much of the available profit your strategy captures. A ratio near 1.0 means you're exiting near the peak. A ratio well below 1.0 means you're leaving profits on the table.

**Your system features:**
- `--report-mae-mfe` — Print per-trade MAE/MFE analysis
- `compute_mae_mfe()` — Computes MAE, MFE, and efficiency ratio for each closed trade

**Key question:** *"My stop-loss keeps getting triggered right before the stock rebounds — how do I calibrate it?"*

Run `--report-mae-mfe` on a backtest *without* stops. Look at the distribution of MAE for winning trades. Set your stop-loss just beyond where most winners' worst dip occurred.

---

### 12. Value-at-Risk (VaR) and CVaR

**The problem:** Standard deviation tells you about *average* risk, but you care most about *tail* risk — how bad things get on the worst days. VaR and CVaR quantify this.

**Definitions:**
- **Value-at-Risk (VaR) at 95%** — "On 95% of days, I lose no more than X%." This is the 5th percentile of your daily return distribution.
- **Conditional VaR (CVaR), a.k.a. Expected Shortfall** — "On the 5% of days that are *worse* than VaR, I lose an average of Y%." This is the mean of the returns in the worst 5% tail.

**Why CVaR is better than VaR:** VaR tells you the boundary of the worst 5%, but says nothing about how bad it gets beyond that boundary. CVaR tells you the *average* of the catastrophic days. A strategy with a 2% VaR and 3% CVaR is very different from one with a 2% VaR and 8% CVaR — the second one has a much fatter tail.

**Your system features:**
- `--report-risk` — Print VaR and CVaR at 95% confidence
- `historical_var()` — Computes VaR from the empirical return distribution
- `cvar()` — Computes Expected Shortfall (mean of returns below VaR threshold)

**Key question:** *"How do I quantify the worst-case scenario for my strategy?"*

Use `--report-risk`. If your CVaR is -4%, it means on the worst ~13 trading days per year (5% of 252), you lose an average of 4% per day. Decide if you can stomach that.

---

### 13. Exposure Limits and Volatility Targeting

**The problem:** Even with good signals, your portfolio can become accidentally concentrated. Maybe your strategy loves tech stocks this month, so 80% of your capital ends up in one sector. Or maybe it's fully invested during a high-volatility period when it should be holding more cash.

**Types of exposure control:**

| Control | What it limits | Flag |
|---------|---------------|------|
| **Max positions** | Number of simultaneous holdings | `--max-positions` |
| **Max allocation** | Capital per position | `--max-alloc` |
| **Sector cap** | Weight in any one sector | `--max-sector-exposure` |
| **Gross exposure** | Total long + short as % of equity | `--max-gross-exposure` |
| **Net exposure** | Long minus short as % of equity | `--max-net-exposure` |
| **Volatility target** | Scale total exposure up/down to hit a target annualized volatility | `--target-portfolio-vol` |

**Volatility targeting** is especially powerful. Instead of always being fully invested, the system scales your exposure inversely with recent market volatility. In calm markets, you're fully invested; in turbulent markets, you automatically reduce exposure. This smooths your return stream and often improves risk-adjusted returns.

**Your system features:**
- `--max-positions`, `--max-alloc` — Basic position limits
- `--max-sector-exposure`, `--sector-map` — Sector-level concentration limits
- `--max-gross-exposure`, `--max-net-exposure` — Portfolio-level exposure limits
- `--target-portfolio-vol`, `--portfolio-vol-lookback` — Dynamic volatility targeting

**Key question:** *"How do I keep my portfolio from being accidentally concentrated in one sector or one bet?"*

Combine multiple layers: `--max-alloc 0.05` (no single position > 5%), `--max-sector-exposure 0.30` (no sector > 30%), and `--target-portfolio-vol 0.15` (scale to 15% annualized vol).

---

### 14. Regime Filtering

**The problem:** Many strategies work well in bull markets but lose money in bear markets. Rather than trying to build a strategy that works in all conditions, you can simply *turn it off* when conditions are unfavorable. This is regime filtering.

**How it works:** A regime filter uses a simple indicator on a broad benchmark (e.g., 50-day SMA vs. 200-day SMA on SPY) to classify the market as "risk-on" or "risk-off." When the regime is off, the engine suppresses new BUY and SHORT signals — existing positions are held, but no new ones are opened.

**Why it lives in the engine, not the strategy:** Regime filtering is a portfolio-level decision, not a stock-level signal. By placing it in the engine, every strategy benefits from it without any strategy needing to implement benchmark logic. This separation of concerns keeps strategies focused on stock selection.

**Your system features:**
- `--regime-benchmark` — Benchmark ticker for regime classification (e.g., `SPY`)
- `--regime-fast` / `--regime-slow` — SMA periods for the regime indicator (default: 100 / 200)
- `--regime-condition` — When to allow trading: `fast_above_slow` (bull filter) or `fast_below_slow` (bear filter)
- `--report-regime` — Performance breakdown by bull/bear/sideways periods

**Key question:** *"Should my strategy behave differently in a bull market vs. a bear market?"*

Test it empirically. Run with and without `--regime-benchmark SPY` and compare. Many trend-following strategies see a significant improvement with regime filtering because they avoid whipsaws in bear markets.

---

### 15. Short Selling Mechanics

**The problem:** Going long means you buy shares hoping the price rises. Going short means you *borrow* shares, sell them, and hope to buy them back cheaper later. Shorting reverses the P&L direction and introduces several complications that don't exist with long positions.

**Key differences from going long:**

| Aspect | Long | Short |
|--------|------|-------|
| **P&L** | Profit when price rises | Profit when price falls |
| **Maximum loss** | 100% (stock goes to zero) | Unlimited (price can rise forever) |
| **Carrying cost** | None (you own the shares) | Borrow cost (annualized rate, charged daily) |
| **Margin** | Pay full price | Must post margin (typically 150% of position value) |
| **Stop-loss direction** | Triggers on price *drop* | Triggers on price *rise* |
| **Dividends** | You receive them | You *pay* them |

**Your system features:**
- `--allow-short` — Enable short selling (disabled by default)
- `--short-borrow-rate` — Annualized cost of borrowing shares (default: 2%)
- `--margin-requirement` — Initial margin requirement (default: 1.5 = 150%)
- `SignalAction.SHORT` / `SignalAction.COVER` — Strategy signals for short entry/exit
- Inverted stop logic — For short positions, stop-loss triggers on price *rise* and take-profit triggers on price *fall*
- Margin tracking — Portfolio tracks margin usage for short positions

**Key question:** *"What's different about shorting vs. going long, and what extra costs does it carry?"*

Beyond the reversed P&L, the borrow cost is a constant drag. At a 2% annual rate, you lose about 0.8 bps per day on every short position. For strategies that hold shorts for weeks or months, this adds up significantly.

---

## Part IV — Measuring Performance

*Raw returns are meaningless without context. "I made 12%" tells you nothing about whether a strategy is good, bad, or lucky.*

---

### 16. Risk-Adjusted Returns: Sharpe, Sortino, Calmar, Omega, Treynor

**The problem:** Two strategies both return 12% per year. Strategy A does it with 5% volatility (steady gains). Strategy B does it with 25% volatility (wild swings). Strategy A is clearly better, but raw return treats them as equal. Risk-adjusted ratios fix this by dividing return by some measure of risk.

**The five ratios:**

| Ratio | Formula | Risk measure | Best for |
|-------|---------|-------------|----------|
| **Sharpe** | (Return - Rf) / σ | Total volatility | General-purpose comparison |
| **Sortino** | (Return - Rf) / σ_down | Downside volatility only | Strategies with asymmetric returns |
| **Calmar** | CAGR / \|Max Drawdown\| | Worst drawdown | Drawdown-sensitive investors |
| **Omega** | Σ gains above threshold / Σ losses below | Full return distribution | Non-normal return distributions |
| **Treynor** | (Return - Rf) / β | Market risk (beta) | Comparing within a broader portfolio |

**Why Sharpe alone isn't enough:** Sharpe penalizes *upside* volatility equally to *downside* volatility. If your strategy has huge winning months and small losing months (positive skew), Sharpe undervalues it. Sortino is better for asymmetric strategies because it only penalizes downside deviation.

**Rough Sharpe benchmarks:**
- < 0.5 — Probably not worth trading
- 0.5–1.0 — Acceptable, typical of buy-and-hold
- 1.0–2.0 — Good; institutional quality
- \> 2.0 — Excellent, but verify it's not overfitted (see concept 18)

**Your system features:**
- `sharpe_ratio()`, `sortino_ratio()`, `calmar_ratio()`, `omega_ratio()`, `treynor_ratio()` — All computed in `metrics.py` and included in the standard report

**Key question:** *"Two strategies both returned 12% — which one was actually better?"*

Compare their Sharpe ratios first, then check Sortino (for asymmetry), Calmar (for drawdown tolerance), and Omega (for the full picture). The one with the highest risk-adjusted return on the metric that matches your risk preference is the better strategy.

---

### 17. Benchmark-Relative Metrics: Alpha, Beta, Information Ratio, Tracking Error

**The problem:** If the S&P 500 returned 10% and your strategy returned 12%, how much of that 12% is your skill and how much is just the market going up? Benchmark-relative metrics decompose your returns into market exposure (beta) and genuine value-add (alpha).

**The key metrics:**

| Metric | What it measures | Interpretation |
|--------|-----------------|----------------|
| **Beta** | Market sensitivity | β=1.0 means you move with the market; β=0.5 means half as sensitive |
| **Alpha** | Return unexplained by beta | Positive α = genuine skill; negative α = you'd be better off indexing |
| **Information Ratio** | Alpha / Tracking Error | Consistency of outperformance; higher is better |
| **Tracking Error** | Std of (your returns - benchmark returns) | How much you deviate from benchmark |
| **Up/Down Capture** | Return in up markets / return in down markets | Ideally: capture > 100% up, < 100% down |

**Why this matters:** A strategy with β=1.5 and 15% returns isn't impressive — it's just leveraged beta. In a 10% market, a β=1.5 strategy "should" return 15%. The alpha is near zero. You'd get similar performance cheaper with a leveraged index fund.

**Your system features:**
- `--benchmark` — Required benchmark ticker for relative metrics
- `alpha()`, `beta()`, `information_ratio()`, `tracking_error()`, `capture_ratio()` — All in `metrics.py`
- Up and down capture computed separately for bull/bear market days

**Key question:** *"Am I actually adding value, or am I just riding the market?"*

Check your alpha. If it's close to zero, your strategy might just be beta exposure with extra steps (and extra costs).

---

### 18. Deflated Sharpe Ratio

**The problem:** You tested 500 parameter combinations and found one with a Sharpe of 2.5. Should you trust it?

No — and the Deflated Sharpe Ratio (DSR) by Bailey & López de Prado tells you exactly why. When you test many combinations, you're effectively running a selection tournament. Even with purely random strategies, the *best* of 500 will have a misleadingly high Sharpe by pure chance. This is the multiple comparisons problem applied to finance.

**The intuition:** If you flip a fair coin 10 times, getting 7 heads isn't remarkable. But if you flip 1,000 different coins 10 times each and then report only the best one, you'll find one with 9 or 10 heads — and it doesn't mean that coin is special.

**What DSR does:** It adjusts your observed Sharpe downward based on:
1. How many parameter combinations you tested
2. The variance of Sharpe ratios across those combinations
3. The skewness and kurtosis of your return distribution

If DSR is still positive after adjustment, your result is more likely genuine.

**Your system features:**
- `--trials` — Tell the system how many parameter combinations you tested
- `deflated_sharpe_ratio()` — Computes the DSR adjustment
- `--permutation-test` — Runs N random permutations of your return stream to compute a p-value: what fraction of random shuffles produce a Sharpe as good as yours?

**Key question:** *"I found a Sharpe of 2.5 after testing 500 parameter combos — should I trust it?"*

Compute the DSR with `--trials 500`. If the DSR drops below zero, your result is likely noise. Also run `--permutation-test 1000` — if the p-value is > 0.05, you can't reject the hypothesis that your result is luck.

---

### 19. Signal Decay and Optimal Holding Period

**The problem:** Every trading signal has a half-life. The information that made you buy the stock erodes over time as the market incorporates it. Signal decay analysis measures how quickly your edge disappears after entry.

**What it shows:** For each completed trade, you measure the cumulative return at T+1, T+2, ..., T+20 days after entry. Averaging across all trades reveals the *decay curve*:
- If returns peak at T+5 and then flatten, your optimal holding period is ~5 days
- If returns peak at T+3 and then decline, you're holding too long and giving back profits
- If returns keep climbing through T+20, you might be exiting too early

**Your system features:**
- `--report-signal-decay` — Print signal decay analysis with cumulative returns at each horizon
- `compute_signal_returns()` — Computes per-trade returns at multiple horizons
- `signal_decay_summary()` — Identifies the optimal holding period (horizon where average return peaks)

**Key question:** *"Am I holding my positions too long and giving back profits?"*

Run `--report-signal-decay`. If the optimal holding period is 5 days but your average trade lasts 20 days, your signals are stale by the time you exit. Consider tighter take-profits or shorter holding windows.

---

### 20. Performance Attribution

**The problem:** Your portfolio returned 15% this year. Great. But was it because your momentum strategy crushed it, or because one lucky NVDA trade made 200% and everything else was flat? Attribution tells you *where* your returns came from.

**Levels of attribution:**
- **Per-strategy** — In a multi-strategy portfolio, how much did each strategy contribute?
- **Per-position** — Was your return broadly distributed or concentrated in a few lucky picks?
- **Per-time-period** — Did you make all your money in January and bleed slowly the rest of the year?

**Concentration risk** is measured by the Herfindahl-Hirschman Index (HHI): the sum of squared position weights. An HHI of 0.10 means a well-diversified portfolio; an HHI above 0.25 suggests dangerous concentration.

**Your system features:**
- Multi-strategy attribution — `MultiStrategyResult.attribution` breaks down CAGR, Sharpe, and P&L contribution by strategy
- `--report-concentration` — Prints HHI, effective N (number of equally-weighted positions that would produce the same HHI), and max position weight
- `--report-correlation` — Correlation matrix of returns across holdings

**Key question:** *"Is my portfolio return coming from one lucky bet, or is it broadly distributed?"*

Run with `--report-concentration --report-correlation`. If HHI is above 0.25 or your top position is more than 20% of the portfolio, you have concentration risk.

---

## Part V — Avoiding Self-Deception

*The human brain is a pattern-recognition machine. It will find patterns in pure noise, and it will convince you they're real. These tools are your defense against your own cognitive biases.*

---

### 21. Overfitting and the Multiple Comparisons Problem

**The problem:** Given enough parameters and enough testing, you can make *any* strategy look profitable on historical data. This is overfitting — your strategy has learned the noise in the historical data rather than any genuine signal.

**Why it happens:** A strategy with 5 free parameters, each tested at 10 values, produces 100,000 combinations. At least one will look great by chance alone. The probability of finding a spurious result grows exponentially with the number of combinations tested.

**Red flags for overfitting:**
- Results are highly sensitive to small parameter changes (if SMA fast=97 works but SMA fast=103 doesn't, you've likely overfit)
- Out-of-sample performance drops dramatically from in-sample
- The strategy has more parameters than necessary for its core idea
- Results that seem "too good" (Sharpe > 3, zero losing months, etc.)

**Your system features:**
- `--trials` — Declares how many combinations you tested, used by DSR
- `deflated_sharpe_ratio()` — Adjusts Sharpe for multiple testing
- `--permutation-test` — Statistical test for significance
- Grid search — Makes it easy to test many combos, but also makes overfitting easy

**Key question:** *"How do I know if my backtest result is real or lucky?"*

Use the trinity of validation: (1) Deflated Sharpe > 0, (2) permutation p-value < 0.05, (3) out-of-sample performance within 50% of in-sample. If all three pass, your result is more likely genuine.

---

### 22. In-Sample vs. Out-of-Sample

**The problem:** If you develop a strategy on data from 2010-2020 and test it on the same 2010-2020 data, you haven't validated anything — you've just confirmed that you can fit a curve to historical data. In-sample data is your training set; out-of-sample data is your honest test.

**The critical rule:** Once you look at your out-of-sample results and go back to change your strategy, the OOS data is now contaminated. It has become in-sample. True OOS means you run it *once* and accept the result, good or bad.

**Typical split:** 70% in-sample for development, 30% out-of-sample for validation. For a 2010-2023 range, that might be 2010-2019 IS and 2019-2023 OOS.

**Your system features:**
- `backtester optimize` — Grid search on in-sample data to find best parameters
- `backtester walk-forward` — Automatically manages IS/OOS splits in rolling windows
- `--is-months` / `--oos-months` — Control window sizes for walk-forward
- `--metric` — Which metric to optimize (default: `sharpe_ratio`)

**Key question:** *"How do I split my data to get honest performance estimates?"*

Use `backtester walk-forward` instead of a single split. It rolls through your data in IS/OOS windows, giving you multiple OOS tests instead of just one. This is much more robust than a single train/test split.

---

### 23. Walk-Forward Validation

**The problem:** A single IS/OOS split gives you one data point. Maybe your OOS period happened to be easy, or maybe it was unusually hard. Walk-forward analysis gives you *multiple* OOS tests by rolling through the data.

**How it works:**
1. Define an IS window (e.g., 12 months) and OOS window (e.g., 3 months)
2. Optimize parameters on the IS window
3. Test those parameters on the next OOS window
4. Slide the window forward and repeat
5. Stitch together all OOS results for an aggregate performance picture

**Anchored vs. rolling:**
- **Rolling** — IS window stays the same size, moves forward. Each optimization sees the same amount of recent history.
- **Anchored** — IS window expands from a fixed start date. Each optimization sees more data, which can be more stable but also slower to adapt.

**The degradation ratio** (OOS Sharpe / IS Sharpe) is your key diagnostic:
- Near 1.0 — Strategy generalizes well. OOS performance matches IS.
- 0.5–1.0 — Some degradation is normal. Strategy likely has real edge but with some overfitting.
- Below 0.5 — Significant overfitting. The strategy memorized in-sample noise.
- Below 0.0 — The strategy *loses money* out of sample. Almost certainly overfit.

**Your system features:**
- `backtester walk-forward` — Full walk-forward analysis
- `--is-months` / `--oos-months` — Window sizes (default: 12 IS, 3 OOS)
- `--anchored` — Use expanding IS window instead of rolling
- `--grid` — Parameter grid to optimize on each IS window
- `--metric` — Optimization target metric
- Degradation ratio computed and reported automatically

**Key question:** *"How do I test whether my strategy adapts to changing markets?"*

`backtester walk-forward --is-months 12 --oos-months 3 --grid '{"sma_fast":[50,100,150],"sma_slow":[200,250,300]}'`. If the degradation ratio is above 0.5 and OOS performance is consistent across windows, your strategy is adaptive.

---

### 24. Purged K-Fold Cross-Validation

**The problem:** Standard K-fold cross-validation randomly shuffles data into K folds. This is fine for independent observations (like classifying images), but financial time series have autocorrelation — today's return is related to yesterday's. Random shuffling breaks this time structure and leaks information between folds.

**How purged K-fold fixes it:**
1. Data is split into K contiguous time-blocks (not shuffled)
2. A **purge gap** is inserted between each training and test fold, removing days that could leak information across the boundary
3. An **embargo period** after each test fold prevents the training data from peeking at returns just after the test window

**Example:** With purge_days=10 and embargo_days=5, there's a 15-day buffer zone between any training and test data, ensuring no autocorrelation leakage.

**Your system features:**
- `--cv-method purged_kfold` — Use purged K-fold instead of standard walk-forward
- `--purge-days` — Gap between train and test folds (default: 10)
- `--embargo-days` — Gap after test fold before next train (default: 5)

**Key question:** *"Why can't I just use normal K-fold cross-validation on time-series data?"*

Because financial returns are autocorrelated. Today's volatility predicts tomorrow's. A random split lets the model "see" the future via correlated observations that leaked into training data. Purged K-fold prevents this.

---

### 25. Monte Carlo Simulation

**The problem:** Your backtest is one realization of history. The specific sequence of returns matters — the same set of daily returns in a different order can produce dramatically different outcomes (an early drawdown vs. a late one changes your compounding path).

**How Monte Carlo works:**
1. Take your actual daily returns
2. Randomly shuffle (bootstrap) them into a new order
3. Compute equity curve from the shuffled returns
4. Repeat 1,000+ times
5. Look at the distribution of outcomes

**What it tells you:**
- **Median outcome** — What you'd expect on average regardless of luck in sequencing
- **P5/P95 bands** — The range of outcomes you should realistically expect
- **Worst case (P5)** — How bad things could get if the bad days cluster together
- **Best case (P95)** — How good things could get with favorable sequencing

If your actual result is near the P95 band, you may have gotten lucky with the specific ordering. If it's near the P5 band, you may have been unlucky.

**Your system features:**
- `--monte-carlo-runs` — Number of simulations (default: 1000)
- `run_monte_carlo()` — Generates N shuffled equity paths
- `monte_carlo_percentiles()` — Computes P5, P25, P50, P75, P95 bands and final equity distribution

**Key question:** *"What's the range of outcomes I should realistically expect?"*

If your Monte Carlo P50 shows 10% CAGR but P5 shows -2%, you should plan for the possibility of losing money even with a strategy that's positive in expectation. The spread between P5 and P95 tells you how much of your result depends on luck vs. skill.

---

### 26. Stress Testing and Scenario Analysis

**The problem:** Backtests that span calm markets look great. But markets periodically have crises that are unlike anything in the recent past. Stress testing runs your strategy through specific historical crisis periods to see if it survives.

**Available scenarios:**

| Scenario | Period | Character |
|----------|--------|-----------|
| Dot-com crash | 2000-2002 | Slow grinding bear, tech collapse |
| Global Financial Crisis | 2007-2009 | Systemic panic, correlations spike to 1.0 |
| Flash crash | May 2010 | Intraday 9% crash and recovery |
| Taper tantrum | 2013 | Bond/equity selloff on Fed talk |
| China devaluation | Aug 2015 | Global contagion from currency shock |
| Volmageddon | Feb 2018 | Volatility spike, short-vol blowups |
| COVID crash | Feb-Mar 2020 | Fastest 30% drop in history |
| Rate hike cycle | 2022 | Sustained selloff from monetary tightening |

**What to look for:**
- Does the strategy survive (final equity > starting equity)?
- How deep is the drawdown during the crisis?
- Does the regime filter help?
- Do stops limit losses effectively?

**Your system features:**
- `backtester stress-test` — Run strategy through crisis scenarios
- `--scenario` — Choose one or more scenarios by name (can be specified multiple times)

**Key question:** *"Would my strategy have survived the 2008 financial crisis?"*

`backtester stress-test --strategy sma_crossover --scenario gfc_2008`. If it would have suffered a 60% drawdown, that's a real possibility in the next crisis, regardless of what your calm-period backtest shows.

---

## Part VI — Portfolio Construction

*Individual trades are tactics. Portfolio construction is strategy. How you size, combine, and rebalance your positions matters at least as much as which stocks you pick.*

---

### 27. Position Sizing: Fixed Fractional, ATR, Vol Parity, Kelly, Risk Parity

**The problem:** You've decided to buy AAPL. But how much? $1,000? $10,000? Your entire portfolio? The same signals with different position sizing produce wildly different results. Sizing is arguably the most under-appreciated lever in a trading system.

**The five models:**

| Model | How it sizes | When to use |
|-------|-------------|-------------|
| **Fixed Fractional** | Same % of equity per trade | Simple, easy to understand |
| **ATR** | Risk budget / (ATR × multiplier) | Size based on stock volatility |
| **Vol Parity** | Scale to equalize volatility contribution | Equal risk from each position |
| **Kelly** | edge / odds (or half-Kelly) | Mathematically optimal for known edge |
| **Risk Parity** | Equal risk contribution from each position | Portfolio-level risk balancing |

**Kelly Criterion deep dive:** The Kelly formula says to bet a fraction of your capital equal to your *edge / odds*. For a strategy with a 55% win rate and 1:1 payoff, Kelly says bet 10% per trade. But Kelly is very aggressive and assumes you know your exact edge — which you don't. **Half-Kelly** (betting half the Kelly amount) is the industry standard: it captures ~75% of the growth rate with ~50% of the volatility.

**ATR sizing:** Instead of betting a fixed dollar amount, you bet a fixed *risk* amount. If ATR (average daily range) is $2, and you want to risk $200 per position, you buy 100 shares. If ATR is $5, you buy 40 shares. This automatically sizes smaller positions in volatile stocks and larger positions in calm ones.

**Your system features:**
- `--position-sizing` — Choose `fixed_fractional`, `atr`, `vol_parity`, `kelly`, or `risk_parity`
- `--risk-pct` — Risk per trade for ATR sizer (default: 1%)
- `--atr-multiple` — ATR multiplier for stop distance (default: 2.0)
- `--vol-target` / `--vol-lookback` — Volatility parity parameters
- `--kelly-fraction` — Kelly fraction (default: 0.5 = half-Kelly)

**Key question:** *"How much should I bet on each trade?"*

Start with fixed fractional for simplicity. Graduate to ATR sizing when you want volatility-adjusted positions. Use vol parity when you want each position to contribute equal risk. Only use Kelly if you have a well-calibrated estimate of your strategy's edge.

---

### 28. Portfolio Rebalancing and Target Weights

**The problem:** You start with 10 positions, each at 10% of your portfolio. One doubles, another halves. Now you're 18% in the winner and 5% in the loser. Your portfolio has *drifted* from its target allocation. Without rebalancing, winners grow and losers shrink, which sounds good but actually concentrates your risk in whatever happened to go up recently.

**Rebalancing trade-offs:**

| Frequency | Pros | Cons |
|-----------|------|------|
| **Daily** | Tight tracking of target weights | High turnover, high costs |
| **Weekly** | Good balance | Moderate costs |
| **Monthly** | Common institutional frequency | Some drift between rebalances |
| **Quarterly** | Low costs | Significant drift possible |

**Target-weight strategies** skip the signal-based approach entirely. Instead of generating BUY/SELL signals, they return a dictionary of desired portfolio weights (e.g., {"AAPL": 0.10, "GOOGL": 0.10, ...}). The engine computes the trades needed to reach those targets. This is the standard approach for factor-based and cross-sectional strategies.

**Your system features:**
- `--rebalance-schedule` — Choose `daily`, `weekly`, `monthly`, or `quarterly`
- `target_weights()` — Strategy method that returns desired portfolio weights
- `compute_rebalance_orders()` — Computes the trades needed to reach target weights from current positions

**Key question:** *"How often should I rebalance, and what are the trade-offs?"*

Monthly is a good starting point. Compare the results at different frequencies using `--save-results` and `backtester compare`. If the improvement from daily rebalancing is small, the extra costs aren't worth it.

---

### 29. Cross-Sectional Strategies and Universe Construction

**The problem:** Most beginner strategies ask "should I buy AAPL today?" Cross-sectional strategies ask a fundamentally different question: "of all the stocks in my universe, which N look best right now?" This relative ranking approach is how most institutional quant strategies actually work.

**How they work:**
1. Define a universe of stocks (e.g., S&P 500 members)
2. Compute a score for each stock (e.g., 12-month momentum)
3. Rank all stocks by score
4. Buy the top N, sell (or short) the bottom N
5. Rebalance periodically

**Universe construction matters:** The universe defines your opportunity set. Using today's S&P 500 members (survivorship bias — see concept 1) gives you a biased universe. A proper historical universe file that tracks index membership changes over time is essential for honest results.

**Your system features:**
- `CrossSectionalStrategy` ABC — Base class for ranking-based strategies
- `rank_universe()` — Override this to score and rank all stocks
- `top_n()` / `bottom_n()` — Helpers to select the N best/worst stocks
- `--universe-file` — Historical universe CSV for survivorship-free testing
- `--tickers` — Fixed ticker list for simple testing
- `--market` / `--universe` — Market scope and breadth for universe construction

**Key question:** *"How do I build a strategy that selects from a universe of stocks instead of trading a fixed list?"*

Subclass `CrossSectionalStrategy`, implement `rank_universe()` to score each stock, and use `top_n()` to select winners. Use `--universe-file` for honest backtesting with point-in-time universe membership.

---

### 30. Factor Models and Alpha Scores

**The problem:** Your strategy buys stocks with high momentum. It performs well. But is your edge *momentum itself* (a well-known factor that everyone can access cheaply via ETFs), or is it something unique to your implementation? If it's just momentum exposure, you're paying transaction costs to replicate something you could get for 0.03% per year in an ETF.

**What a factor is:** A factor is any measurable stock characteristic that predicts returns over time. The most established factors are:
- **Momentum** — Stocks that went up tend to keep going up (12-month return)
- **Value** — Cheap stocks (low P/E, P/B) tend to outperform expensive ones
- **Size** — Small-cap stocks tend to outperform large-cap (with more risk)
- **Low volatility** — Calm stocks tend to outperform wild ones (risk-adjusted)
- **Quality** — Profitable, low-leverage companies tend to outperform

**Alpha vs. factor exposure:** If your strategy's returns can be fully explained by exposure to known factors (high beta to momentum, for example), your alpha is zero. True alpha is the residual return *after* accounting for factor exposures.

**Your system features:**
- `--fundamental-data` — Path to fundamental data CSV (P/E, P/B, etc.)
- `get_fundamental()` — Point-in-time fundamental data lookup (no lookahead)
- Indicators library — 18 indicators including momentum (`roc`), RSI, MACD, and more
- `CrossSectionalStrategy` — Rank stocks by any factor score

**Key question:** *"How do I know if my strategy is capturing a real edge or just loading up on a known factor?"*

Compare your strategy's returns to a simple momentum factor (or whatever factor your signals are based on). If your strategy's alpha over that factor is near zero, you're paying trading costs for factor exposure you could get more cheaply.

---

### 31. Multi-Strategy Portfolios

**The problem:** Relying on a single strategy is like a single stock — concentrated risk. If your one strategy's edge decays, you're exposed. Running multiple strategies simultaneously diversifies across *edges*, not just across stocks.

**How it works:**
1. Define multiple strategies with their own parameters
2. Allocate capital across strategies (e.g., 40% momentum, 30% mean-reversion, 30% value)
3. Each strategy runs independently with its proportional cash
4. Combined equity curve is the sum of individual equity curves

**Key benefit — diversification of alpha:** If your momentum strategy and your mean-reversion strategy are uncorrelated, combining them reduces overall portfolio volatility without reducing expected return. This is the same diversification principle as holding multiple stocks, but applied at the strategy level.

**What to check:**
- **Per-strategy attribution:** Is one strategy carrying the portfolio while others drag?
- **Strategy correlation:** Low correlation between strategies = good diversification
- **Combined Sharpe:** Should be higher than any individual strategy's Sharpe (if strategies are uncorrelated)

**Your system features:**
- `MultiStrategyConfig` / `StrategyAllocation` — Define a multi-strategy portfolio with weights
- `run_multi_strategy()` — Run all strategies independently and combine results
- `MultiStrategyResult.attribution` — Per-strategy CAGR, Sharpe, P&L, and contribution percentage
- Per-strategy metrics — Full metric set for each strategy component

**Key question:** *"How do I combine multiple strategies into one portfolio?"*

Use the Python API to define a `MultiStrategyConfig` with `StrategyAllocation` entries specifying each strategy's name, parameters, and weight. The system handles cash allocation, independent execution, and combined reporting.

---

## Quick Reference: Concepts → CLI Flags

| Concept | Key flags |
|---------|-----------|
| Survivorship bias | `--universe-file` |
| Corporate actions | `--adjust-prices`, `--drip` |
| Lookahead prevention | (Built-in: T/T+1 fill model) |
| Slippage | `--slippage-model`, `--slippage-bps`, `--slippage-impact` |
| Partial fills | `--max-volume-pct`, `--partial-fill-policy` |
| Transaction costs | `--fee-model`, `--fee`, `--report-tca` |
| Order types | (Strategy-level: `Signal` dataclass) |
| Lot accounting | `--lot-method` |
| Drawdown kill switch | `--max-drawdown` |
| Stops | `--stop-loss`, `--take-profit`, `--trailing-stop`, `--stop-loss-atr`, `--take-profit-atr` |
| MAE/MFE | `--report-mae-mfe` |
| VaR/CVaR | `--report-risk` |
| Exposure limits | `--max-positions`, `--max-alloc`, `--max-sector-exposure`, `--max-gross-exposure`, `--max-net-exposure` |
| Volatility targeting | `--target-portfolio-vol`, `--portfolio-vol-lookback` |
| Regime filter | `--regime-benchmark`, `--regime-fast`, `--regime-slow`, `--regime-condition` |
| Short selling | `--allow-short`, `--short-borrow-rate`, `--margin-requirement` |
| Risk-adjusted returns | (Standard report output) |
| Benchmark metrics | `--benchmark` |
| Deflated Sharpe | `--trials`, `--permutation-test` |
| Signal decay | `--report-signal-decay` |
| Attribution | `--report-concentration`, `--report-correlation` |
| Overfitting | `--trials`, `--permutation-test` |
| IS/OOS | `backtester optimize`, `backtester walk-forward` |
| Walk-forward | `--is-months`, `--oos-months`, `--anchored` |
| Purged K-fold | `--cv-method purged_kfold`, `--purge-days`, `--embargo-days` |
| Monte Carlo | `--monte-carlo-runs` |
| Stress testing | `backtester stress-test`, `--scenario` |
| Position sizing | `--position-sizing`, `--risk-pct`, `--atr-multiple`, `--vol-target`, `--kelly-fraction` |
| Rebalancing | `--rebalance-schedule` |
| Cross-sectional | `CrossSectionalStrategy`, `--universe-file` |
| Factor models | `--fundamental-data`, indicators library |
| Multi-strategy | Python API: `MultiStrategyConfig` |
