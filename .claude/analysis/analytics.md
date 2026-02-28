# Analytics Module Analysis: analytics/

## Module Purpose

Post-backtest performance measurement, reporting, and simulation. `metrics.py` computes all quantitative metrics (CAGR, Sharpe, drawdown, trade stats, benchmark-relative). `report.py` formats console output and matplotlib charts. `calendar.py` provides monthly/yearly breakdowns and heatmaps. `montecarlo.py` runs bootstrap equity path simulations.

## Key Classes/Functions

### metrics.py (20+ functions)
| Name | Description |
|------|-------------|
| `compute_all_metrics()` | Master function — computes all metrics and returns as dict; the primary API |
| `cagr()` | Compound Annual Growth Rate from equity series |
| `sharpe_ratio()` | Annualized Sharpe (daily returns, 252 trading days) |
| `sortino_ratio()` | Annualized Sortino (downside deviation only) |
| `max_drawdown()` | Max drawdown as negative fraction |
| `max_drawdown_duration()` | Max drawdown duration in calendar days |
| `total_return()` | Simple total return fraction |
| `calmar_ratio()` | CAGR / |MaxDD| |
| `win_rate()` | Fraction of trades with positive PnL |
| `profit_factor()` | Gross profit / gross loss |
| `beta()`, `alpha()` | CAPM beta and Jensen's alpha vs benchmark |
| `information_ratio()`, `tracking_error()` | Active return metrics |
| `capture_ratio()` | Up/down capture vs benchmark |
| `trade_expectancy()` | Average PnL per trade |
| `avg_win_loss()` | Average winner/loser and payoff ratio (returns dict) |
| `holding_period_stats()` | Avg/median holding days (returns dict) |
| `max_consecutive()` | Max consecutive wins/losses (returns dict) |
| `exposure_time()` | Fraction of days with open positions |

### report.py
| Name | Description |
|------|-------------|
| `print_report()` | Main console output — strategy perf, benchmark perf, relative metrics, trade stats; returns metrics dict |
| `plot_results()` | Matplotlib equity curve + drawdown chart + monthly heatmap |
| `export_activity_log_csv()` | Write per-fill activity log to CSV |
| `_print_activity_log()` | Print per-fill activity table to console |

### calendar.py
| Name | Description |
|------|-------------|
| `monthly_returns()` | Year x Month pivot table of returns |
| `yearly_summary()` | Yearly return, max drawdown, Sharpe per calendar year |
| `drawdown_periods()` | Top N drawdown periods by depth |
| `print_calendar_report()` | Console output of monthly/yearly/drawdown tables |
| `plot_monthly_heatmap()` | Matplotlib heatmap of monthly returns |

### montecarlo.py
| Name | Description |
|------|-------------|
| `run_monte_carlo()` | Bootstrap-resample daily returns → simulated equity paths (2D numpy array) |
| `monte_carlo_percentiles()` | Compute percentile bands from paths (p5, p25, p50, p75, p95) |

## Critical Data Flows

1. **Engine → Analytics**: `BacktestResult` contains `equity_series` (pd.Series), `trades` (list of Trade), `benchmark_series`, `config`, `activity_log`
2. **Metrics computation**: `compute_all_metrics(equity, trades, benchmark)` → dict of all metric values. Benchmark-relative metrics only computed when benchmark provided.
3. **Report output**: `print_report()` calls `compute_all_metrics()` then formats to console; also calls `print_calendar_report()` and `_print_activity_log()`
4. **Monte Carlo**: Independent of report — takes equity series, returns numpy arrays of simulated paths

## External Dependencies

### Third-party
- `numpy` — numerical computations (metrics.py, montecarlo.py, calendar.py)
- `pandas` — Series/DataFrame operations (all files)
- `matplotlib.pyplot` — charts (report.py, calendar.py)

### Internal
- `backtester.analytics.metrics` — imported by report.py
- `backtester.analytics.calendar` — lazy imported by report.py
- `backtester.engine.BacktestResult` — imported by report.py (creates circular-ish dependency)

## "Do Not Touch" Warnings

1. **252 trading days assumption**: Sharpe, Sortino, alpha, information ratio all annualize using √252. Changing this breaks all risk-adjusted metrics.
2. **`max_drawdown` returns negative fraction**: e.g., -0.25 for 25% drawdown. Calmar ratio uses `abs(dd)`. Sign convention is important.
3. **`compute_all_metrics()` dict-spread pattern**: Uses `**avg_win_loss(trades)` and `**holding_period_stats(trades)` to merge sub-dicts. Keys must not collide.
4. **Lazy imports in report.py**: `calendar.print_calendar_report` and `calendar.plot_monthly_heatmap` are imported inside functions to avoid circular imports.
5. **`exposure_time()` uses `pd.bdate_range`**: Not the trading calendar — slight approximation. Acceptable for reporting but not for trading logic.
6. **Monte Carlo is stateless**: Uses `np.random.default_rng(seed)` — reproducible when seed is provided. No global random state mutation.
