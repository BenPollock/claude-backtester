"""SMA Crossover strategy: buy when fast SMA > slow SMA, sell when below."""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("sma_crossover")
class SmaCrossover(Strategy):
    """Long-only SMA crossover strategy.

    Parameters:
        sma_fast: fast SMA period (default 50)
        sma_slow: slow SMA period (default 200)
    """

    def __init__(self):
        self.sma_fast = 50
        self.sma_slow = 200

    def configure(self, params: dict) -> None:
        self.sma_fast = params.get("sma_fast", self.sma_fast)
        self.sma_slow = params.get("sma_slow", self.sma_slow)

        if self.sma_fast >= self.sma_slow:
            raise ValueError(f"sma_fast ({self.sma_fast}) must be < sma_slow ({self.sma_slow})")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma_fast"] = sma(df["Close"], self.sma_fast)
        df["sma_slow"] = sma(df["Close"], self.sma_slow)
        return df

    def generate_signals(
        self,
        symbol: str,
        row: pd.Series,
        position: Position | None,
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> SignalAction:
        fast = row.get("sma_fast")
        slow = row.get("sma_slow")

        # Skip if indicators not yet computed (warmup period)
        if pd.isna(fast) or pd.isna(slow):
            return SignalAction.HOLD

        has_position = position is not None and position.total_quantity > 0

        if fast > slow and not has_position:
            return SignalAction.BUY
        elif fast < slow and has_position:
            return SignalAction.SELL

        return SignalAction.HOLD
