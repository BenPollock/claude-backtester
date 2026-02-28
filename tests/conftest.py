"""Shared test fixtures."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig
from backtester.portfolio.portfolio import Portfolio
from backtester.data.sources.base import DataSource
from backtester.strategies.registry import discover_strategies

# Auto-discover all strategy modules so they register with the registry
discover_strategies()


def make_price_df(
    start: str = "2020-01-02",
    days: int = 252,
    start_price: float = 100.0,
    daily_return: float = 0.0005,
    volume: int = 1_000_000,
) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame for testing."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    rng = np.random.default_rng(42)

    prices = [start_price]
    for _ in range(days - 1):
        ret = daily_return + rng.normal(0, 0.02)
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices)

    df = pd.DataFrame(
        {
            "Open": prices * (1 - rng.uniform(0, 0.01, days)),
            "High": prices * (1 + rng.uniform(0, 0.02, days)),
            "Low": prices * (1 - rng.uniform(0, 0.02, days)),
            "Close": prices,
            "Volume": np.full(days, volume),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )
    return df


class MockDataSource(DataSource):
    """Data source that returns pre-loaded DataFrames."""

    def __init__(self, data: dict[str, pd.DataFrame] | None = None):
        self._data = data or {}

    def add(self, symbol: str, df: pd.DataFrame) -> None:
        self._data[symbol] = df

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if symbol not in self._data:
            raise ValueError(f"No mock data for {symbol}")
        df = self._data[symbol]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]


@pytest.fixture
def sample_df():
    """One year of synthetic price data."""
    return make_price_df()


@pytest.fixture
def portfolio():
    """Fresh portfolio with $100k."""
    return Portfolio(cash=100_000.0)


@pytest.fixture
def mock_source():
    """Empty mock data source."""
    return MockDataSource()


@pytest.fixture
def basic_config():
    """Basic backtest config for testing."""
    return BacktestConfig(
        strategy_name="sma_crossover",
        tickers=["TEST"],
        benchmark="TEST",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 12, 31),
        starting_cash=100_000.0,
        max_positions=10,
        max_alloc_pct=0.10,
        strategy_params={"sma_fast": 20, "sma_slow": 50},
    )
