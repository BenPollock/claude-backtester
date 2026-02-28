# Data Module Analysis: data/

## Module Purpose

Cache-first OHLCV data loading pipeline with NYSE calendar alignment. `DataManager` orchestrates fetching from pluggable sources (default: Yahoo Finance), caching via Parquet files, and preparing data aligned to trading days with forward-fill for small gaps. `UniverseProvider` resolves ticker lists from index constituents (S&P 500, TSX).

## Key Classes/Functions

### manager.py
| Name | Description |
|------|-------------|
| `DataManager` | Orchestrates cache-first loading; main entry point for all data access |
| `DataManager.load()` | Load single symbol: check cache → fetch if needed → prepare (reindex + ffill) |
| `DataManager.load_many()` | Load multiple symbols sequentially; errors logged and skipped |
| `DataManager._prepare()` | Reindex to trading calendar days, forward-fill gaps up to `MAX_FFILL_DAYS=5` |

### cache.py
| Name | Description |
|------|-------------|
| `ParquetCache` | One Parquet file per ticker under `~/.backtester/cache/` |
| `ParquetCache.merge_and_save()` | Concat new + existing data, deduplicate (keep last), sort, save |
| `ParquetCache.date_range()` | Return (first, last) dates of cached data for cache-hit detection |

### calendar.py
| Name | Description |
|------|-------------|
| `TradingCalendar` | NYSE session calendar wrapper; falls back to business days outside supported range |
| `TradingCalendar.trading_days()` | Return DatetimeIndex of trading days in [start, end] |
| `TradingCalendar.next_trading_day()` | Next session after a given date (used by broker for T+1 fills) |

### sources/base.py
| Name | Description |
|------|-------------|
| `DataSource` | ABC — `fetch(symbol, start, end) -> DataFrame` with OHLCV columns |

### sources/yahoo.py
| Name | Description |
|------|-------------|
| `YahooDataSource` | Fetches split-adjusted OHLCV from yfinance with retry/backoff |
| `YahooDataSource._normalize()` | Strips tz, deduplicates, ensures required columns |

### universe.py
| Name | Description |
|------|-------------|
| `UniverseProvider` | Scrapes S&P 500 / TSX constituents from Wikipedia; JSON-cached for 7 days |
| `UniverseProvider._get_or_fetch()` | Cache-first with stale fallback on fetch failure |

## Critical Data Flows

1. **Load request**: `DataManager.load(symbol, start, end)` → check `ParquetCache.date_range()` → cache hit: load + prepare; cache miss: fetch from source → `merge_and_save` → prepare
2. **Prepare step**: Reindex DataFrame to `TradingCalendar.trading_days()` → forward-fill up to 5 days → return
3. **Multi-symbol**: `load_many()` calls `load()` sequentially (no parallelism); errors are logged and the symbol is skipped
4. **Universe resolution**: CLI calls `UniverseProvider.get_tickers(market, universe)` → checks JSON cache → scrapes Wikipedia if stale/missing

## External Dependencies

### Third-party
- `pandas` — DataFrames, DatetimeIndex (all files)
- `pyarrow` — Parquet engine (cache.py)
- `yfinance` — Yahoo Finance API (yahoo.py)
- `exchange_calendars` — NYSE session calendar (calendar.py)
- `urllib.request` — Wikipedia scraping (universe.py)

### Internal
- No imports from other backtester modules (data/ is a leaf dependency)

## "Do Not Touch" Warnings

1. **Timezone-naive DatetimeIndex**: The entire system assumes `df.index` is tz-naive dates. `YahooDataSource._normalize()` strips tz — do not remove this.
2. **`auto_adjust=True`** in `yf.Ticker.history()`: Returns split-adjusted prices. Changing to `False` breaks all price-based calculations.
3. **yfinance exclusive end date**: `end + timedelta(days=1)` compensates for yfinance's exclusive end parameter. Do not "fix" this.
4. **Duplicate handling order**: `keep="last"` in both cache and yahoo normalize. New data overwrites old data — intentional for corrections.
5. **`MAX_FFILL_DAYS = 5`**: Forward-fill limit. Larger values mask bad data; smaller values create spurious NaN gaps.
6. **Calendar fallback to business days**: Dates outside `exchange_calendars` range use `pd.bdate_range`. This is intentional for historical data before 1960s.
7. **Sequential fetching in `load_many()`**: No parallelism by design — avoids rate-limiting from Yahoo Finance.
8. **Wikipedia scraping fragility**: `_fetch_sp500()` and `_fetch_tsx()` depend on Wikipedia table structure. Stale-cache fallback is critical.
