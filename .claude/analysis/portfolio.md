# Portfolio Module Analysis

**Path:** `src/backtester/portfolio/`

## Module Purpose

Manages all portfolio state: cash balance, open positions (with FIFO lot tracking), equity history, and completed trade records. Provides a read-only `PortfolioState` snapshot for safe strategy consumption, and defines the `Order`, `Fill`, `Trade`, and `TradeLogEntry` data structures used throughout the execution pipeline.

## Key Classes/Functions

### `portfolio.py`
| Class/Member | Description |
|---|---|
| `Portfolio` | Mutable portfolio state: cash, positions dict, equity history, trade/activity logs |
| `Portfolio.total_equity` | Property: cash + sum of all position market values |
| `Portfolio.snapshot()` | Returns a frozen `PortfolioState` for safe read-only access by strategies |
| `Portfolio.open_position(symbol)` | Get-or-create a `Position` for a symbol |
| `Portfolio.close_position(symbol)` | Removes position from dict only if total quantity is zero |
| `Portfolio.position_weight(symbol)` | Returns position market value as fraction of total equity |
| `PortfolioState` | Frozen dataclass snapshot: cash, total_equity, num_positions, position_symbols |

### `order.py`
| Class | Description |
|---|---|
| `Order` | Mutable order with symbol, side, quantity, type, status, optional stop/limit prices, parent_id for brackets |
| `Fill` | Frozen record of an executed fill: price (post-slippage), commission, fill_date (T+1) |
| `TradeLogEntry` | Per-fill activity log entry for BUY/SELL with cost basis, fees, slippage |
| `Trade` | Frozen round-trip trade record: entry/exit dates+prices, PnL, holding days, total fees |

### `position.py`
| Class | Description |
|---|---|
| `Lot` | Single purchase lot: quantity, entry_price, entry_date, allocated commission |
| `StopState` | Per-position stop-loss, take-profit, and trailing-stop levels with high-water mark |
| `Position` | Holds a list of `Lot` objects for one symbol; sells via FIFO; tracks market price |
| `Position.sell_lots_fifo()` | Core FIFO sell: consumes lots front-to-back, allocates commissions proportionally, returns `Trade` list |

## Critical Data Flows

1. **BUY fill** -> `Portfolio.open_position(symbol)` -> `Position.add_lot(qty, price, date, commission)` -> appends `Lot` to `position.lots`
2. **SELL fill** -> `Position.sell_lots_fifo(qty, exit_price, exit_date, commission)` -> walks `lots[0]` first (FIFO), creates `Trade` per lot consumed, mutates/removes lots in-place -> returns `list[Trade]` -> appended to `Portfolio.trade_log`
3. **Market price update** -> `Position.update_market_price(price)` -> sets `_market_price` used by `market_value` property -> feeds into `Portfolio.total_equity`
4. **Equity recording** -> `Portfolio.record_equity(date)` -> appends `(date, total_equity)` tuple to `equity_history` (consumed by analytics)
5. **Strategy access** -> `Portfolio.snapshot()` -> returns frozen `PortfolioState` (strategies never mutate portfolio directly)

## External Dependencies

### Internal (backtester modules)
- `backtester.types` — `Side`, `OrderType`, `OrderStatus` enums (used by `order.py`)
- `backtester.portfolio.order` — `Trade`, `TradeLogEntry` (used by `portfolio.py`)
- `backtester.portfolio.position` — `Position` (used by `portfolio.py`)

### Third-party / stdlib
- `dataclasses` (dataclass, field)
- `datetime` (date)
- `uuid` (hex IDs for orders)

No pandas, numpy, or heavy library dependencies -- this module is pure Python dataclasses.

## "Do Not Touch" Warnings

1. **FIFO lot ordering is load-bearing.** `sell_lots_fifo` always consumes `self.lots[0]` first. Sorting or reordering `lots` breaks PnL calculations and trade records.
2. **Partial lot mutation in `sell_lots_fifo`.** When a lot is partially sold, both `lot.quantity` and `lot.commission` are decremented in-place. The commission adjustment (`lot.commission -= entry_comm`) preserves proportional fee allocation for future sells of the remaining lot.
3. **`close_position` guard.** Only deletes from `positions` dict when `total_quantity == 0`. Callers must sell all lots before calling close; otherwise the position silently persists.
4. **`PortfolioState` is frozen for a reason.** Strategies receive snapshots, not the live `Portfolio`. Passing the mutable `Portfolio` to strategies would break the lookahead-prevention model.
5. **`Order.signal_date` vs `Fill.fill_date` separation.** Signals generate on day T close; fills execute on T+1 open. Collapsing these into a single date breaks the core anti-lookahead invariant.
6. **`_market_price` defaults to 0.0.** If `update_market_price` is not called before reading `market_value`, positions report zero value. The engine must update prices before equity snapshots.
7. **`pnl_pct` in `sell_lots_fifo` is per-lot, not fee-adjusted.** It is `(exit_price / entry_price - 1)`, while `pnl` (dollar) includes fees. These are intentionally inconsistent -- do not "fix" one to match the other without updating analytics consumers.
