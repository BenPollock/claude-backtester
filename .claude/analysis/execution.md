# Execution Module Analysis

## Module Purpose

The `execution/` package handles simulated order execution: accepting orders, filling them at
next-day open prices with configurable slippage and fee models, and determining position sizes.
It is the sole bridge between strategy signals and portfolio state mutations.

## Key Classes/Functions

### broker.py
- **SimulatedBroker** -- Queues orders via `submit_order()`, fills them in `process_fills()` at the next day's open price. Directly mutates Portfolio cash, positions, and logs.

### slippage.py
- **SlippageModel** (ABC) -- Interface for price-adjustment models; `compute()` returns adjusted fill price.
- **FixedSlippage** -- Applies a fixed basis-point spread (default 10 bps). BUY pays more, SELL receives less.
- **VolumeSlippage** -- Slippage proportional to `order.quantity / volume`, scaled by an impact factor (default 0.1).

### fees.py
- **FeeModel** (ABC) -- Interface for commission models; `compute()` returns fee amount.
- **PerTradeFee** -- Flat fee per trade (default $0.00).

### position_sizing.py
- **PositionSizer** (ABC) -- Interface for share-count calculation given price, equity, cash, and max allocation.
- **FixedFractional** -- Allocates `max_alloc_pct` of equity per position, capped by available cash.
- **ATRSizer** -- Sizes by risking a fixed equity fraction per ATR-based stop distance; falls back to FixedFractional when ATR is unavailable.
- **VolatilityParity** -- Weights inversely to annualized volatility (ATR proxy); falls back to FixedFractional when ATR is unavailable.

## Critical Data Flows

1. **Order submission:** Engine calls `broker.submit_order(order)` -> order appended to `_pending_orders` list.
2. **Fill processing (called next trading day):** `broker.process_fills(date, market_data, portfolio)` iterates pending orders:
   - Reads the day's **Open** price from `market_data[symbol]`.
   - Computes adjusted price via `SlippageModel.compute()`.
   - Computes commission via `FeeModel.compute()`.
   - For BUY: validates cash sufficiency; auto-reduces quantity if short on cash; cancels if zero shares affordable.
   - For SELL: resolves sentinel quantity `-1` to `pos.total_quantity` (sell-all); calls `pos.sell_lots_fifo()`.
   - Mutates `portfolio.cash`, creates `Fill`, appends to `portfolio.activity_log` and `portfolio.trade_log`.
   - Closes position if remaining quantity is zero.
3. **Position sizing (called by engine before order creation):** `PositionSizer.compute()` returns share count; ATRSizer and VolatilityParity read the `"ATR"` column from the indicator row.

## External Dependencies

### Internal (backtester modules)
- `backtester.portfolio.order` -- Order, Fill, TradeLogEntry
- `backtester.portfolio.portfolio` -- Portfolio (cash, positions, logs)
- `backtester.types` -- Side, OrderStatus enums

### Third-party
- `pandas` -- pd.Series for market data rows and indicator access; `pd.isna()` for null checks
- `abc` -- ABC/abstractmethod for model interfaces
- `logging` -- Debug-level fill/cancel logging in broker

## "Do Not Touch" Warnings

1. **Fill at Open price:** `process_fills` reads `row["Open"]`, not Close. Changing this breaks the lookahead-prevention contract (signals on day T close, fills on day T+1 open).
2. **SELL sentinel value `-1`:** A quantity of `-1` means "sell entire position". Code in `process_fills` resolves this before execution. Do not change this convention without updating all strategy signal generators.
3. **Cash mutation ordering:** For BUY, cost is `price * qty + commission`. For SELL, proceeds are `price * qty - commission`. These are asymmetric by design. Altering arithmetic breaks P&L tracking.
4. **FIFO sell via `pos.sell_lots_fifo()`:** The broker delegates lot-level accounting to `Position`. Do not add lot logic here; it lives in `portfolio/position.py`.
5. **`_pending_orders` cleared per cycle:** After `process_fills`, only orders without market data survive. All others are filled or cancelled. The engine must not call `process_fills` twice per day.
6. **ATR fallback in sizers:** Both `ATRSizer` and `VolatilityParity` silently fall back to `FixedFractional` logic when the `"ATR"` column is missing or invalid. Removing this fallback will cause zero-share orders for strategies that do not compute ATR.
