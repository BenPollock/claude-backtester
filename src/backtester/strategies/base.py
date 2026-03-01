"""Strategy abstract base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@dataclass(frozen=True)
class Signal:
    """Wraps a SignalAction with optional order parameters.

    Strategies can return either a plain ``SignalAction`` or a ``Signal``
    instance from ``generate_signals()``.  Returning a ``Signal`` allows
    specifying limit order parameters without breaking backward compatibility.

    Attributes:
        action: The trading signal (BUY, SELL, HOLD, SHORT, COVER).
        limit_price: If set, submits a limit order instead of a market order.
        time_in_force: ``"DAY"`` (expires end of day) or ``"GTC"``
            (good-til-cancelled, persists across days).  Default ``"DAY"``.
        expiry_date: Optional explicit expiry date for GTC orders.
    """

    action: SignalAction
    limit_price: float | None = None
    time_in_force: str = "DAY"
    expiry_date: date | None = None


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

    @property
    def timeframes(self) -> list[str]:
        """Timeframes this strategy needs.

        Default is ``['daily']``, meaning the strategy only uses daily bars.
        Override to request additional timeframes, e.g. ``['daily', 'weekly']``.
        The engine will resample daily data and pass it to
        ``compute_indicators()`` via the *timeframe_data* argument.
        """
        return ["daily"]

    @abstractmethod
    def compute_indicators(
        self,
        df: pd.DataFrame,
        timeframe_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Add indicator columns to the DataFrame. Must be backward-looking only.

        Called once per ticker with the full daily history.

        Args:
            df: Daily OHLCV DataFrame for one ticker.
            timeframe_data: Optional dict mapping timeframe names (e.g.
                ``'weekly'``, ``'monthly'``) to resampled OHLCV DataFrames
                that have been forward-filled to the daily index. Only
                provided when the strategy's ``timeframes`` property
                requests non-daily timeframes.

        Returns:
            DataFrame with indicator columns added.
        """
        ...

    def compute_benchmark_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to the benchmark DataFrame.

        Called once before the main loop. Default is a no-op.
        Override in strategies that need benchmark-derived indicators.
        """
        return df

    @abstractmethod
    def generate_signals(
        self,
        symbol: str,
        row: pd.Series,
        position: Position | None,
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> "SignalAction | Signal":
        """Generate a trading signal for one symbol on one day.

        Args:
            symbol: Ticker symbol
            row: Today's OHLCV + computed indicators
            position: Current position (None if no position)
            portfolio_state: Read-only portfolio snapshot
            benchmark_row: Today's benchmark data (if available)

        Returns:
            A ``SignalAction`` (BUY, SELL, HOLD, SHORT, or COVER) for market
            orders, or a ``Signal`` instance to specify limit order parameters
            such as ``limit_price`` and ``time_in_force``.
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
        For SHORT: return negative quantity (short up to max_alloc_pct).
        For COVER: return -1 sentinel (cover all).
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

        elif action == SignalAction.SHORT:
            # Short up to max_alloc_pct of total equity (returned as negative qty)
            target_value = portfolio_state.total_equity * max_alloc_pct
            price = row["Close"]
            if price <= 0:
                return 0
            max_from_cash = portfolio_state.cash
            target_value = min(target_value, max_from_cash)
            qty = int(target_value // price)
            return -qty  # negative indicates short

        elif action == SignalAction.COVER:
            # Cover entire short position
            if symbol in portfolio_state.position_symbols:
                return -1  # sentinel: cover all

        return 0
