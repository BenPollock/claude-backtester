# engine.py Analysis

## Module Purpose

Orchestrates the full backtest lifecycle: loading data, computing indicators, iterating
trading days, and collecting results. Acts as the central coordinator between data,
strategy, broker, portfolio, and analytics subsystems.

## Key Classes/Functions

- **BacktestResult** -- Immutable container holding config, portfolio, and benchmark equity; exposes `equity_series` and `benchmark_series` as pd.Series properties.
- **BacktestEngine.__init__** -- Configures execution models (slippage, fees, position sizer) from `BacktestConfig`.
- **BacktestEngine.run** -- Main entry point: loads data, computes indicators, runs day-by-day loop, force-closes positions, returns `BacktestResult`.
- **_compute_regime_indicators** -- Adds `regime_fast`/`regime_slow` SMA columns to benchmark DataFrame.
- **_check_regime** -- Evaluates regime filter on a single benchmark row; returns bool (True = BUY signals allowed).
- **_set_stops_for_fills** -- Attaches stop-loss, take-profit, and trailing-stop levels to newly opened positions after BUY fills.
- **_check_stop_triggers** -- Intraday check using High/Low to trigger stop exits; executes same-day (no T+1 delay).
- **_force_close_all** -- Closes all open positions at last known market price on the final trading day.

## Critical Data Flows

1. `run()` loads universe data via `DataManager.load_many()` -> dict[symbol, DataFrame].
2. `strategy.compute_indicators()` mutates each ticker DataFrame in-place (vectorized, pre-loop).
3. Benchmark DataFrame gets regime indicator columns added separately by `_compute_regime_indicators`.
4. Day-by-day loop (indexed by `TradingCalendar.trading_days`):
   - `today_data` is built by slicing each ticker DataFrame at timestamp `ts`.
   - `broker.process_fills()` fills yesterday's pending orders at today's open prices.
   - `_set_stops_for_fills()` sets stop levels on newly filled BUY positions.
   - `_check_stop_triggers()` evaluates intraday H/L against stop levels; triggers immediate same-day exits.
   - Position market prices updated to today's close.
   - `portfolio.record_equity(day)` snapshots equity curve.
   - `strategy.generate_signals()` called per-symbol; output filtered by regime and position limits.
   - BUY sizing delegated to `PositionSizer`; SELL sizing delegated to `strategy.size_order()`.
   - Orders submitted to broker (fill next day at open).
5. After loop, `_force_close_all()` liquidates remaining positions.

## External Dependencies

**Internal backtester modules:**
- `config` (BacktestConfig, RegimeFilter)
- `data.manager` (DataManager), `data.calendar` (TradingCalendar)
- `strategies.base` (Strategy), `strategies.registry` (get_strategy), `strategies.indicators` (sma)
- `execution.broker` (SimulatedBroker), `execution.slippage` (FixedSlippage, VolumeSlippage), `execution.fees` (PerTradeFee)
- `execution.position_sizing` (PositionSizer, FixedFractional, ATRSizer, VolatilityParity)
- `portfolio.portfolio` (Portfolio), `portfolio.order` (Order, TradeLogEntry), `portfolio.position` (StopState)
- `types` (Side, OrderType, SignalAction)

**Third-party:** `pandas`, `logging` (stdlib), `datetime` (stdlib)

## "Do Not Touch" Warnings

1. **Lookahead invariant:** Signals use day-T close; orders fill at day-T+1 open. Reordering steps (a) through (f) in the day loop will break this. Stops are the one exception -- they execute same-day using intraday H/L.
2. **Regime filter is engine-level only.** Strategies must never contain benchmark/regime logic; the engine suppresses BUY->HOLD when regime is off (line 205-206).
3. **`_check_stop_triggers` bypasses the broker.** It directly modifies portfolio cash, trade_log, and activity_log. Changes to broker fill logic or fee computation must be mirrored here.
4. **`_force_close_all` uses `pos._market_price` (private attr)** with zero commission. Any refactor of Position's price tracking will break this.
5. **BUY vs SELL sizing asymmetry:** BUY orders use the engine's `PositionSizer`; SELL orders use `strategy.size_order()`. Do not unify without updating all strategies.
6. **Benchmark shares initialized lazily** (line 178-179) on first non-NaN close. Moving benchmark equity tracking earlier will produce incorrect buy-and-hold comparison.
