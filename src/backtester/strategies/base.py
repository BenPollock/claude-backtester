"""Strategy abstract base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

from backtester.types import SignalAction, OrderType
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
        stop_price: If set, submits a stop or stop-limit order.
        order_type: Order type (MARKET, LIMIT, STOP, STOP_LIMIT).
        time_in_force: ``"DAY"`` (expires end of day) or ``"GTC"``
            (good-til-cancelled, persists across days).  Default ``"DAY"``.
        expiry_date: Optional explicit expiry date for GTC orders.
    """

    action: SignalAction
    limit_price: float | None = None
    stop_price: float | None = None
    order_type: OrderType = OrderType.MARKET
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

    def __init__(self):
        self._fundamental_data = None

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

    # --- Gap 9: Fundamental data access ---

    def set_fundamental_data(self, manager) -> None:
        """Inject a FundamentalDataManager. Called by the engine."""
        self._fundamental_data = manager

    def get_fundamental(self, symbol: str, field: str, as_of: date) -> float | None:
        """Look up a point-in-time fundamental value."""
        if self._fundamental_data is None:
            return None
        return self._fundamental_data.get(symbol, field, as_of)

    # --- Gap 15: Target-weight rebalancing ---

    def target_weights(
        self,
        bar_data: dict[str, pd.Series],
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> dict[str, float] | None:
        """Return target portfolio weights, or None for signal-based mode.

        Override in weight-based strategies to return a dict mapping
        symbols to target weights (0.0 to 1.0). When this returns
        non-None, the engine will compute rebalance orders instead of
        processing per-symbol signals.
        """
        return None


class CrossSectionalStrategy(Strategy):
    """Strategy that ranks the entire universe and selects top/bottom symbols.

    Subclasses implement ``rank_universe()`` instead of per-symbol
    ``generate_signals()``.  The engine detects this via ``isinstance``
    and calls ``rank_universe()`` once per day with all available bar
    data.
    """

    def generate_signals(self, symbol, row, position, portfolio_state, benchmark_row=None):
        """Not used for cross-sectional strategies — returns HOLD."""
        return SignalAction.HOLD

    @abstractmethod
    def rank_universe(
        self,
        bar_data: dict[str, pd.Series],
        positions: dict[str, "Position"],
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> list[tuple[str, SignalAction]]:
        """Rank all symbols and return a list of (symbol, signal) tuples.

        Called once per trading day with all available bar data.
        Return signals for the symbols you want to trade.
        """
        ...

    @staticmethod
    def top_n(scores: dict[str, float], n: int) -> list[str]:
        """Return top N symbols by score (descending)."""
        return [s for s, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]

    @staticmethod
    def bottom_n(scores: dict[str, float], n: int) -> list[str]:
        """Return bottom N symbols by score (ascending)."""
        return [s for s, _ in sorted(scores.items(), key=lambda x: x[1])[:n]]
