# Strategies Module Analysis: strategies/

## Module Purpose

Pluggable strategy framework with an ABC, decorator-based registry, a library of vectorized indicators, and two concrete strategies (SMA crossover, rule-based DSL). Strategies compute indicators once per ticker (vectorized), then generate per-day signals during the backtest loop.

## Key Classes/Functions

### base.py
| Name | Description |
|------|-------------|
| `Strategy` | ABC defining the strategy lifecycle: `configure()` → `compute_indicators()` → `generate_signals()` → `size_order()` |
| `Strategy.compute_indicators()` | Abstract — add indicator columns to DataFrame (vectorized, backward-looking only) |
| `Strategy.generate_signals()` | Abstract — return `SignalAction` (BUY/SELL/HOLD) for one symbol on one day |
| `Strategy.size_order()` | Default sizing: buy up to `max_alloc_pct` of equity; sell returns `-1` sentinel (sell all) |
| `Strategy.compute_benchmark_indicators()` | Optional — add indicators to benchmark DataFrame (default no-op) |

### registry.py
| Name | Description |
|------|-------------|
| `register_strategy(name)` | Decorator that registers a Strategy subclass by name in `_REGISTRY` dict |
| `get_strategy(name)` | Instantiate a registered strategy by name |
| `list_strategies()` | Return sorted list of registered strategy names |

### indicators.py (18 functions)
| Name | Description |
|------|-------------|
| `sma`, `ema` | Simple/Exponential moving averages |
| `rsi` | Relative Strength Index (Wilder's smoothing) |
| `atr` | Average True Range |
| `macd` | MACD line, signal line, histogram (tuple return) |
| `bollinger` | Bollinger Bands — upper, middle, lower (tuple return) |
| `stochastic` | Stochastic %K, %D (tuple return) |
| `adx` | Average Directional Index |
| `obv` | On-Balance Volume |
| `keltner`, `donchian` | Channel indicators (tuple returns) |
| `williams_r`, `cci`, `mfi`, `roc` | Oscillators/momentum |
| `psar` | Parabolic SAR (iterative numpy loop) |
| `ichimoku` | Ichimoku Cloud — 5 components (tuple return) |
| `vwap` | Rolling VWAP proxy for daily data |

### sma_crossover.py
| Name | Description |
|------|-------------|
| `SmaCrossover` | Long-only SMA crossover; params: `sma_fast` (default 50), `sma_slow` (default 200) |

### rule_based.py
| Name | Description |
|------|-------------|
| `RuleBasedStrategy` | JSON-configurable strategy — define indicators and buy/sell conditions as `[left, op, right]` rules |
| `INDICATOR_REGISTRY` | Maps indicator names → (function, takes_df_flag) for the DSL |
| `_apply_indicator()` | Applies an indicator spec to DataFrame, handling tuple vs scalar returns |
| `_evaluate_rules()` | Evaluates rule triples against a context dict (AND logic) |

## Critical Data Flows

1. **Registration**: `@register_strategy("name")` decorator adds class to `_REGISTRY` at import time. CLI imports strategy modules to trigger registration.
2. **Lifecycle per ticker**: `configure(params)` → `compute_indicators(df)` adds columns → engine iterates rows calling `generate_signals()` per day
3. **Rule-based DSL**: JSON params define `indicators` (computed in `compute_indicators`), `benchmark_indicators` (computed separately), `buy_when`/`sell_when` rules evaluated against merged context dict
4. **Signal context**: `generate_signals` receives: current row (with indicators), position state, portfolio snapshot, optional benchmark row

## External Dependencies

### Third-party
- `pandas` — DataFrames, Series (all files)
- `numpy` — numerical ops (indicators.py)
- `math`, `operator` — rule evaluation (rule_based.py)

### Internal backtester imports
- `backtester.types.SignalAction` — signal enum (base.py, sma_crossover.py, rule_based.py)
- `backtester.portfolio.portfolio.PortfolioState` — read-only portfolio snapshot (base.py)
- `backtester.portfolio.position.Position` — position state (base.py)

## "Do Not Touch" Warnings

1. **`-1` sell sentinel**: `size_order()` returns `-1` to mean "sell all". The broker interprets this. Do not change to an actual share count.
2. **Side-effect imports in cli.py**: Strategy modules must be imported in `cli.py` to trigger `@register_strategy`. Adding a new strategy requires adding its import there.
3. **`compute_indicators` must be backward-looking only**: Forward-looking indicators would introduce lookahead bias. The engine enforces day-by-day signal access but relies on this contract.
4. **Indicator tuple return conventions**: `macd` returns 3, `bollinger`/`keltner`/`donchian` return 3, `stochastic` returns 2, `ichimoku` returns 5. `_apply_indicator` depends on these exact shapes.
5. **Rule evaluation is AND logic only**: All rules in `buy_when`/`sell_when` must pass. No OR support — users combine via separate strategy instances.
6. **`df.copy()` in compute_indicators**: Both strategies copy the DataFrame before mutating. Removing this would mutate the caller's data.
