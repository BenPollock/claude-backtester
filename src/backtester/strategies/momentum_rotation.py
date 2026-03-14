"""Sector ETF Momentum Rotation Strategy.

Ranks a universe of sector ETFs by rate-of-change (ROC) over a
configurable lookback period, then buys the top N performers and
sells any holdings that fall out of the top N.

Designed for monthly rebalancing with a regime filter.
"""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import roc
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("momentum_rotation")
class MomentumRotation(Strategy):
    """Cross-sectional momentum rotation across sector ETFs."""

    def __init__(self):
        super().__init__()
        self.roc_period = 63  # ~3 months of trading days
        self.top_n_count = 3

    def configure(self, params: dict) -> None:
        self.roc_period = params.get("roc_period", self.roc_period)
        self.top_n_count = params.get("top_n", self.top_n_count)

    def compute_indicators(self, df: pd.DataFrame, timeframe_data=None) -> pd.DataFrame:
        df = df.copy()
        df["roc"] = roc(df["Close"], period=self.roc_period)
        return df

    def generate_signals(
        self,
        symbol: str,
        row: dict,
        positions: dict[str, Position],
        portfolio_state: PortfolioState,
        benchmark_row: dict | None = None,
    ) -> SignalAction:
        """Not used directly -- rank_universe handles cross-sectional logic.

        Falls back to HOLD for any per-symbol call.
        """
        return SignalAction.HOLD

    def rank_universe(
        self,
        bar_data: dict[str, dict],
        positions: dict[str, Position],
        portfolio_state: PortfolioState,
        benchmark_row: dict | None = None,
    ) -> list[tuple[str, SignalAction]]:
        """Rank all symbols by ROC and generate BUY/SELL signals.

        Args:
            bar_data: Dict mapping symbol -> current row (dict of column values).
            positions: Dict mapping symbol -> Position for current holdings.
            portfolio_state: Frozen snapshot of portfolio state.
            benchmark_row: Current benchmark data row (optional).

        Returns:
            List of (symbol, SignalAction) tuples.
        """
        # Score all symbols by their ROC value
        scores: dict[str, float] = {}
        for sym, row in bar_data.items():
            val = row.get("roc")
            if pd.notna(val):
                scores[sym] = val

        # Select top N by ROC
        sorted_symbols = sorted(scores, key=scores.get, reverse=True)
        top = set(sorted_symbols[: self.top_n_count])

        signals: list[tuple[str, SignalAction]] = []

        # Sell any current holdings NOT in the top N
        for sym in list(positions.keys()):
            if sym not in top and positions[sym].total_quantity > 0:
                signals.append((sym, SignalAction.SELL))

        # Buy any top N symbols we do NOT currently hold
        for sym in top:
            if sym not in positions or positions[sym].total_quantity == 0:
                signals.append((sym, SignalAction.BUY))

        return signals

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
