"""BacktestResult container for backtest outputs."""

from datetime import date

import pandas as pd

from backtester.config import BacktestConfig
from backtester.portfolio.portfolio import Portfolio


class BacktestResult:
    """Container for backtest outputs."""

    def __init__(self, config: BacktestConfig, portfolio: Portfolio,
                 benchmark_equity: list[tuple[date, float]] | None = None):
        self.config = config
        self.portfolio = portfolio
        self.benchmark_equity = benchmark_equity

    @property
    def equity_series(self) -> pd.Series:
        dates, values = zip(*self.portfolio.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="Date"), name="Equity")

    @property
    def benchmark_series(self) -> pd.Series | None:
        if not self.benchmark_equity:
            return None
        dates, values = zip(*self.benchmark_equity)
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="Date"), name="Benchmark")

    @property
    def trades(self):
        return self.portfolio.trade_log

    @property
    def activity_log(self):
        return self.portfolio.activity_log
