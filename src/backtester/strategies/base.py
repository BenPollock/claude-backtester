"""Strategy abstract base class."""

from abc import ABC, abstractmethod

import pandas as pd

from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


class Strategy(ABC):
    """Base class for all trading strategies.

    Lifecycle:
    1. configure(params) — receive strategy-specific parameters
    2. compute_indicators(df) — called ONCE per ticker with full history (vectorized)
    3. generate_signals() — called per (symbol, day) during the backtest loop
    4. size_order() — determine position size for a signal
    """

    def configure(self, params: dict) -> None:
        """Override to accept strategy-specific parameters."""
        pass

    def required_columns(self) -> list[str]:
        """Columns that must exist in data before compute_indicators is called."""
        return ["Open", "High", "Low", "Close", "Volume"]

    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to the DataFrame. Must be backward-looking only.

        Called once per ticker with the full history. Return the modified DataFrame.
        """
        ...

    @abstractmethod
    def generate_signals(
        self,
        symbol: str,
        row: pd.Series,
        position: Position | None,
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> SignalAction:
        """Generate a trading signal for one symbol on one day.

        Args:
            symbol: Ticker symbol
            row: Today's OHLCV + computed indicators
            position: Current position (None if no position)
            portfolio_state: Read-only portfolio snapshot
            benchmark_row: Today's benchmark data (if available)

        Returns:
            SignalAction: BUY, SELL, or HOLD
        """
        ...

    def size_order(
        self,
        symbol: str,
        action: SignalAction,
        row: pd.Series,
        portfolio_state: PortfolioState,
        max_alloc_pct: float,
    ) -> int:
        """Determine number of shares to buy/sell.

        Default: buy up to max_alloc_pct of equity; sell entire position.
        """
        if action == SignalAction.BUY:
            # Buy up to max_alloc_pct of total equity
            target_value = portfolio_state.total_equity * max_alloc_pct
            price = row["Close"]
            if price <= 0:
                return 0
            # Don't exceed available cash
            max_from_cash = portfolio_state.cash
            target_value = min(target_value, max_from_cash)
            return int(target_value // price)

        elif action == SignalAction.SELL:
            # Sell entire position
            if symbol in portfolio_state.position_symbols:
                # Caller will look up actual quantity from position
                return -1  # sentinel: sell all

        return 0
