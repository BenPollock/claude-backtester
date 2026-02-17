"""Backtest configuration dataclasses."""

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class RegimeFilter:
    """Benchmark-based regime filter configuration."""

    benchmark: str  # e.g. "SPY"
    indicator: str  # e.g. "sma"
    fast_period: int  # e.g. 100
    slow_period: int  # e.g. 200
    condition: str = "fast_above_slow"


@dataclass(frozen=True)
class BacktestConfig:
    """Complete configuration for a backtest run."""

    strategy_name: str
    tickers: list[str]
    benchmark: str
    start_date: date
    end_date: date
    starting_cash: float
    max_positions: int
    max_alloc_pct: float  # 0.10 = 10%
    fee_per_trade: float = 0.05
    slippage_model: str = "fixed"
    slippage_bps: float = 10.0
    fill_delay_days: int = 1  # signal on close T, fill at open T+1
    data_source: str = "yahoo"
    data_cache_dir: str = "~/.backtester/cache"
    monte_carlo_runs: int = 1000
    strategy_params: dict = field(default_factory=dict)
    regime_filter: RegimeFilter | None = None
