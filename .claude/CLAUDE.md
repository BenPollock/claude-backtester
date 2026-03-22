# Claude-Backtester Technical Guide

## Module Map

| Module | Path | Purpose |
|--------|------|---------|
| CLI / Config / Types | `src/backtester/cli.py`, `config.py`, `types.py` | Click CLI, frozen config dataclasses, Side/SignalAction/OrderType/SHORT/COVER enums |
| Engine | `src/backtester/engine.py` | Central orchestrator: data load, day loop, regime filter, stop triggers, short selling, limit orders, multi-timeframe |
| Data | `src/backtester/data/` | Cache-first OHLCV loading (Parquet), NYSE calendar, Yahoo fetcher, universe scraper, `resample_ohlcv()` for multi-timeframe |
| EDGAR Data | `src/backtester/data/edgar_*.py`, `fundamental.py`, `fundamental_cache.py` | SEC EDGAR integration: financial statements (10-K/10-Q), insider trading (Form 4), institutional holdings (13F), material events (8-K). EdgarDataManager merges all onto daily DataFrames with `fund_`/`insider_`/`inst_`/`event_` prefixed columns |
| Alt Data | `src/backtester/data/market_data.py`, `fred_source.py`, `sentiment.py`, `analyst.py` | VIX term structure + intermarket (yfinance), FRED macro regime + yield curve (fredapi), CBOE put-call ratio, analyst revisions. All merged onto daily DataFrames with `vix_`/`intermarket_`/`fred_`/`yield_`/`sentiment_`/`analyst_` prefixed columns |
| Strategies | `src/backtester/strategies/` | Strategy ABC + registry, SMA crossover, rule-based DSL, value_quality, earnings_growth, fundamental_screener, insider_following, smart_money, macro_aware_value, sentiment_momentum, risk_regime, 18 indicator functions, `Signal` dataclass for limit orders, `timeframes` property |
| Execution | `src/backtester/execution/` | SimulatedBroker (next-day fills, limit orders, short fills), slippage/fee models, position sizers |
| Portfolio | `src/backtester/portfolio/` | Mutable Portfolio state, FIFO Position/Lot tracking (long + short), Order/Fill/Trade models, margin tracking |
| Analytics | `src/backtester/analytics/` | CAGR/Sharpe/drawdown metrics, report output, Monte Carlo, calendar analytics, tearsheet, correlation/concentration, signal decay, regime analysis |
| Research | `src/backtester/research/` | Grid search parameter optimization, walk-forward IS/OOS analysis, multi-strategy portfolio |
| Tests | `tests/` | ~1554 tests, 56 files, all synthetic data, no network calls |

## Key Data Flows

**Main Backtest Pipeline:**
CLI parses args --> `BacktestConfig` (frozen) --> `BacktestEngine.run()` --> `DataManager.load_many()` (cache-first) --> `EdgarDataManager.merge_all_onto_daily()` (if EDGAR enabled, adds `fund_`/`insider_`/`inst_`/`event_` columns) --> `_merge_auxiliary_data()` (if alt-data enabled, adds `vix_`/`intermarket_`/`fred_`/`yield_`/`sentiment_`/`analyst_` columns) --> `strategy.compute_indicators()` (vectorized, once per ticker) --> day-by-day loop --> `BacktestResult` --> `print_report()`

**Day Loop (strict order -- do not reorder):**
1. `broker.process_fills()` -- fill yesterday's orders at today's Open
2. `_set_stops_for_fills()` -- attach stops to new positions
3. `_check_stop_triggers()` -- intraday H/L stop exits (bypasses broker)
4. Update position market prices to Close
5. `portfolio.record_equity(day)` -- snapshot equity curve
6. `strategy.generate_signals()` per symbol --> regime filter --> position limit check
7. Size orders --> `broker.submit_order()` (fills next day)

**Day Loop (additions for P3/P4):**
- Between steps 3 and 4: daily borrow cost accrual for short positions
- Step 6 now handles SHORT/COVER signals, unwraps `Signal` objects for limit orders
- Engine resamples multi-timeframe data and merges into daily DataFrames before the loop

**Day Loop (additions for alt-data):**
- Step 6 regime filter (`_is_regime_on()`) now also checks VIX ratio (< 1.0 = contango = allow) and FRED macro score (>= 0.5 = allow). Mode "supplement" ANDs with SMA; "replace" uses only FRED+VIX

**Order Lifecycle:**
`strategy.generate_signals()` returns `SignalAction` or `Signal` --> engine sizes order (BUY: PositionSizer; SELL/COVER: `strategy.size_order()` returning -1 sentinel; SHORT: negative qty) --> `broker.submit_order()` queues --> next day `process_fills()` determines fill price (Open for market, limit_price for limits if H/L range reached) --> applies slippage/fees --> routes by `order.reason` (short_entry/cover) --> mutates Portfolio. Unfilled DAY limit orders expire; GTC persist.

**Research Pipeline:**
`grid_search()` builds Cartesian param sweep --> per-combo `BacktestEngine.run()` --> `compute_all_metrics()` --> rank by metric. `walk_forward()` splits into rolling IS/OOS windows, uses `grid_search` on IS, validates best params on OOS.

## Shared Interfaces

| Interface | Location | Implementations |
|-----------|----------|-----------------|
| `Strategy` ABC | `strategies/base.py` | `SmaCrossover`, `RuleBasedStrategy`, `ValueQuality`, `EarningsGrowth`, `FundamentalScreener`, `InsiderFollowing`, `SmartMoney`, `MacroAwareValue`, `SentimentMomentum`, `RiskRegime` |
| `@register_strategy` | `strategies/registry.py` | Decorator populates `_REGISTRY` dict; lookup via `get_strategy(name)` |
| `DataSource` ABC | `data/manager.py` | `YahooDataSource` |
| `SlippageModel` ABC | `execution/slippage.py` | `FixedSlippage`, `VolumeSlippage` |
| `FeeModel` ABC | `execution/fees.py` | `PerTradeFee`, `PercentageFee`, `TieredFee`, `SECFee`, `TAFFee`, `CompositeFee` |
| `PositionSizer` ABC | `execution/position_sizing.py` | `FixedFractional`, `ATRSizer`, `VolatilityParity` |
| `Signal` dataclass | `strategies/base.py` | Wraps `SignalAction` with `limit_price`, `time_in_force`, `expiry_date` |
| `StrategyAllocation` / `MultiStrategyConfig` | `research/multi_strategy.py` | Frozen dataclasses for multi-strategy portfolio definitions |
| `MarketDataManager` | `data/market_data.py` | VIX term structure + intermarket data via yfinance |
| `FredDataSource` | `data/fred_source.py` | FRED macro regime + Treasury yield curve (requires fredapi) |
| `CBOEPutCallSource` | `data/sentiment.py` | CBOE equity put-call ratio from free CSV endpoint |
| `AnalystRevisionSource` | `data/analyst.py` | Analyst earnings revisions via yfinance (per-symbol, snapshot-only) |

## Critical Invariants

- **T/T+1 Lookahead Prevention:** Signals use day T close; orders fill at day T+1 open. `fill_delay_days=1` in config. Fill at Open price (market) or limit price (limit orders), never Close.
- **FIFO Lot Accounting:** `Position.sell_lots_fifo()` for longs, `Position.close_lots_fifo()` for shorts. Both consume `lots[0]` first. Lot management lives in Position only.
- **-1 Sell/Cover Sentinel:** `strategy.size_order()` returns -1 meaning "sell/cover all"; broker resolves to `pos.total_quantity` or `abs(pos.total_quantity)`.
- **Frozen Configs:** All config dataclasses are `frozen=True`. Research uses `dataclasses.replace()` to create modified copies; base config is never mutated.
- **Strategy Auto-Discovery:** `discover_strategies()` in `registry.py` uses `pkgutil` to scan and import all strategy modules. Called at startup in `cli.py`, `conftest.py`, and `strategies/__init__.py`. New strategies are auto-discovered -- no manual imports needed.
- **Regime Filter is Engine-Level:** Strategies never handle benchmark logic. Engine suppresses BUY and SHORT signals when regime is "off".
- **DataFrame Column Contracts:** `compute_indicators()` must call `df.copy()` to avoid mutating caller data. Indicators must be backward-looking only.
- **PortfolioState Immutability:** Strategies receive frozen `PortfolioState` snapshots, never mutable `Portfolio`.
- **SignalAction vs Side:** `SignalAction` (BUY/SELL/HOLD/SHORT/COVER) is strategy-layer; `Side` (BUY/SELL) is execution-layer. Short routing uses `order.reason` field ("short_entry"/"cover").
- **Stop Triggers Bypass Broker:** Engine `_check_stop_triggers()` directly mutates portfolio (same-day execution). Short stops trigger on price rise (stop_loss) and price fall (take_profit).
- **252 Trading Days:** All annualization uses fixed 252-day assumption.
- **Limit Order Fill Logic:** Limit BUY fills if day's Low <= limit_price; limit SELL fills if day's High >= limit_price. DAY orders expire same day; GTC orders persist (with optional expiry_date).
- **Short Selling Guarded:** `allow_short=False` by default in BacktestConfig. SHORT signals rejected when disabled. Borrow costs tracked on Position but not auto-deducted from cash.
- **Multi-Timeframe Forward-Fill:** Resampled weekly/monthly data is forward-filled onto the daily index to prevent lookahead. Strategies access via prefixed columns (e.g., `weekly_Close`).
- **Multi-Strategy Isolation:** Each strategy allocation runs independently with its own proportional cash. No cross-strategy interaction. Combined equity = sum of individual curves.
- **EDGAR Point-in-Time:** All EDGAR data keyed on `filed_date`, not `period_end`. Financial data only visible after SEC filing. Insider trades visible after Form 4 filing (within 2 days of transaction).
- **EDGAR Column Prefixes:** `fund_` (10-K/10-Q financials), `insider_` (Form 4), `inst_` (13F holdings), `event_` (8-K events). Mirrors existing `weekly_`/`monthly_` pattern.
- **EDGAR Graceful Degradation:** All EDGAR strategies check for required columns and return HOLD if absent. Safe to run without `--use-edgar`.
- **EDGAR Cache Isolation:** Cached under `{cache_dir}/edgar/{type}/{SYMBOL}.parquet`. Separate from OHLCV cache.
- **edgartools Optional:** `pip install -e ".[edgar]"` to enable. All edgartools imports guarded with try/except ImportError.
- **Alt-Data Column Prefixes:** `vix_` (VIX term structure), `intermarket_` (copper/gold/dollar), `fred_` (macro regime), `yield_` (Treasury curve), `sentiment_` (put-call ratio), `analyst_` (earnings revisions). Mirrors EDGAR `fund_`/`insider_`/`inst_`/`event_` pattern.
- **Alt-Data Opt-In Flags:** `use_vix`, `use_intermarket`, `use_fred`, `use_yield_curve`, `use_pcr`, `use_analyst` in BacktestConfig. All `False` by default — zero impact on existing behavior.
- **Alt-Data Auxiliary Merge:** Engine `_merge_auxiliary_data()` reindexes aux data to daily index + forward-fill. No lookahead — data only appears on or after its publication date.
- **VIX + FRED Engine-Level Regime:** VIX backwardation (ratio > 1.0) and FRED macro score (< 0.5) suppress BUY/SHORT signals via `_is_regime_on()`. `fred_regime_mode` controls "supplement" (AND with SMA) vs "replace" (FRED/VIX only).
- **fredapi Optional:** `pip install -e ".[fred]"` to enable. `fredapi` import guarded with try/except ImportError. Requires `FRED_API_KEY` env var or `fred_api_key` config.
- **Analyst Data Point-in-Time Limitation:** yfinance provides current snapshot only, not historical estimates. `AnalystRevisionSource` places values on last date only; all other dates NaN.
- **Alt-Data Graceful Degradation:** All alt-data strategies (macro_aware_value, sentiment_momentum, risk_regime) check for required columns and return HOLD if absent. Safe to run without alt-data flags.

## Patterns

- **Frozen Dataclasses:** Config, Fill, Trade, BacktestResult, PortfolioState -- immutability prevents accidental mutation
- **ABC + Implementation:** Strategy, DataSource, SlippageModel, FeeModel, PositionSizer -- all use abstract base with concrete subclasses
- **Cache-First Loading:** ParquetCache checked before Yahoo fetch; stale-cache fallback for universe scraping
- **Vectorized-Then-Iterative:** `compute_indicators()` runs once vectorized per ticker; `generate_signals()` runs per-day iteratively
- **Silent Fallbacks:** ATR sizers fall back to FixedFractional when ATR unavailable; TradingCalendar falls back to bdate_range pre-1960s; ffill(limit=5) for gaps
- **Error-Tolerant Sweeps:** Grid search logs failed runs as error rows rather than raising exceptions

## Test Approach

```bash
pytest tests/ -v                                          # full suite (1554 tests)
pytest tests/test_e2e.py tests/test_edgar_e2e.py tests/test_alt_data_e2e.py -v  # E2E integration tests (148 tests)
pytest tests/test_portfolio.py::TestPosition::test_sell_fifo -v  # single unit test
```

**Two test tiers:**
1. **Unit tests** (`tests/test_*.py` excluding `test_e2e.py`) — test individual modules in isolation
2. **E2E integration tests** (`tests/test_e2e.py`) — run full backtests through the complete pipeline (engine, strategy, broker, portfolio, analytics) with only the data source mocked

**When to write E2E tests:** Always consider adding an E2E test when a change affects cross-module data flow (engine → broker → portfolio), order execution, signal generation, portfolio accounting, or configuration options that alter engine behavior (stops, regime filter, fees, sizing). E2E tests use `MockDataSource` + `make_controlled_df()` for deterministic price data and run full backtests via `BacktestEngine.run()`. See `tests/test_e2e.py` for patterns.

**Key Fixtures** (`conftest.py`): `make_price_df(seed=42)` generates deterministic OHLCV; `MockDataSource` serves pre-loaded DataFrames; `basic_config` provides standard BacktestConfig; `sample_df` is 252-day synthetic data; `portfolio` is fresh $100k Portfolio.

**Do not change:** conftest `discover_strategies()` call (registration), `make_price_df` seed=42, `basic_config` defaults, `MockDataSource` inclusive date filtering.

**Coverage gaps:** Multi-timeframe strategies lack end-to-end integration tests with real indicator logic. Tearsheet visual correctness is not tested (only structural HTML checks). Analyst revision data is snapshot-only (no historical point-in-time).

**Test counts by area:** E2E integration (69), alt-data E2E (54), EDGAR E2E (25), fundamental/financial (66), insider (24), institutional (23), Piotroski F-Score (32), Altman Z-Score (18), shareholder yield (19), dividend growth (16), VIX term structure (18), intermarket (17), macro regime (22), yield curve (18), put-call ratio (17), earnings revisions (20), strategies (97), short selling (47), metrics (64), limit orders (39), fees extended (34), gap features (64), portfolio (73), broker (42), position sizing (41), correlation (26), tearsheet (21), regime (21), multi-strategy (19), signal decay (17), multi-timeframe (16), stops (16), calendar analytics (26), CLI (63), report (12), optimizer (13), engine (22), data (25), slippage (11), Monte Carlo (10), calendar (18), universe (7), activity log (12), EDGAR utils (33), integration (28), overfitting (26), blackbox (70), TCA (17), stress (13), cross-validation (11), phase1/2/4 coverage (42).
