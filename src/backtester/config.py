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
class StopConfig:
    """Stop-loss / take-profit / trailing-stop configuration."""

    stop_loss_pct: float | None = None      # e.g. 0.05 = sell if price drops 5% from entry
    take_profit_pct: float | None = None    # e.g. 0.20 = sell if price rises 20% from entry
    trailing_stop_pct: float | None = None  # e.g. 0.08 = 8% trail from high-water mark
    stop_loss_atr: float | None = None      # e.g. 2.0 = stop at entry - 2*ATR
    take_profit_atr: float | None = None    # e.g. 3.0 = target at entry + 3*ATR


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
    stop_config: StopConfig | None = None
    position_sizing: str = "fixed_fractional"  # "fixed_fractional", "atr", "vol_parity"
    sizing_risk_pct: float = 0.01  # for ATR sizer: risk 1% of equity per trade
    sizing_atr_multiple: float = 2.0  # for ATR sizer: stop distance in ATR units
