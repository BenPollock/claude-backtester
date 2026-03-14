"""Multi-Timeframe Trend Following Strategy.

Uses weekly moving average crossover to determine trend direction,
then enters on daily RSI pullbacks within the trend. Exits when
daily RSI becomes overbought.

Requires the engine to provide weekly timeframe data via the
multi-timeframe feature (columns prefixed with 'weekly_').
"""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma, rsi
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("multi_tf_trend")
class MultiTimeframeTrend(Strategy):
    """Weekly trend + daily RSI entry strategy."""

    def __init__(self):
        super().__init__()
        self.weekly_fast = 10
        self.weekly_slow = 40
        self.daily_rsi_period = 14
        self.rsi_entry = 40
        self.rsi_exit = 70

    @property
    def timeframes(self) -> list[str]:
        """Request weekly data from the engine."""
        return ["weekly"]

    def configure(self, params: dict) -> None:
        self.weekly_fast = params.get("weekly_fast", self.weekly_fast)
        self.weekly_slow = params.get("weekly_slow", self.weekly_slow)
        self.daily_rsi_period = params.get("daily_rsi_period", self.daily_rsi_period)
        self.rsi_entry = params.get("rsi_entry", self.rsi_entry)
        self.rsi_exit = params.get("rsi_exit", self.rsi_exit)

    def compute_indicators(self, df: pd.DataFrame, timeframe_data=None) -> pd.DataFrame:
        df = df.copy()

        # Daily indicators
        df["daily_rsi"] = rsi(df["Close"], period=self.daily_rsi_period)

        # Weekly indicators (from multi-timeframe data)
        if timeframe_data and "weekly" in timeframe_data:
            weekly_df = timeframe_data["weekly"]
            weekly_df = weekly_df.copy()
            weekly_df["weekly_sma_fast"] = sma(weekly_df["Close"], period=self.weekly_fast)
            weekly_df["weekly_sma_slow"] = sma(weekly_df["Close"], period=self.weekly_slow)

            # Forward-fill weekly data onto daily index
            # The engine handles the merge, but we compute indicators on the weekly frame
            df["weekly_sma_fast"] = weekly_df["weekly_sma_fast"].reindex(df.index, method="ffill")
            df["weekly_sma_slow"] = weekly_df["weekly_sma_slow"].reindex(df.index, method="ffill")
        elif "weekly_Close" in df.columns:
            # Engine already merged weekly data with prefix
            df["weekly_sma_fast"] = sma(df["weekly_Close"], period=self.weekly_fast)
            df["weekly_sma_slow"] = sma(df["weekly_Close"], period=self.weekly_slow)

        return df

    def generate_signals(
        self,
        symbol: str,
        row: dict,
        positions: dict[str, Position],
        portfolio_state: PortfolioState,
        benchmark_row: dict | None = None,
    ) -> SignalAction:
        """Generate signals based on weekly trend + daily RSI.

        Logic:
        - Weekly uptrend: weekly_sma_fast > weekly_sma_slow
        - BUY: In weekly uptrend AND daily RSI < rsi_entry (pullback)
        - SELL: Daily RSI > rsi_exit (overbought)
        """
        daily_rsi_val = row.get("daily_rsi")
        weekly_fast_val = row.get("weekly_sma_fast")
        weekly_slow_val = row.get("weekly_sma_slow")

        # Need all indicators to be available
        if any(pd.isna(v) for v in [daily_rsi_val, weekly_fast_val, weekly_slow_val]
               if v is not None):
            return SignalAction.HOLD

        if weekly_fast_val is None or weekly_slow_val is None or daily_rsi_val is None:
            return SignalAction.HOLD

        has_position = (
            symbol in positions and positions[symbol].total_quantity > 0
        )

        weekly_uptrend = weekly_fast_val > weekly_slow_val

        # Exit signal: RSI overbought
        if has_position and daily_rsi_val > self.rsi_exit:
            return SignalAction.SELL

        # Entry signal: weekly uptrend + daily RSI pullback
        if not has_position and weekly_uptrend and daily_rsi_val < self.rsi_entry:
            return SignalAction.BUY

        return SignalAction.HOLD

    def size_order(
        self,
        symbol: str,
        action: SignalAction,
        row: dict,
        positions: dict[str, Position],
        portfolio_state: PortfolioState,
    ) -> int:
        """Return -1 for SELL (sell all), defer to sizer for BUY."""
        if action == SignalAction.SELL:
            return -1
        return 0  # Let the position sizer handle BUY sizing
