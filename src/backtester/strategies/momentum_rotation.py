"""Cross-Asset Momentum Rotation Strategy.

Ranks a universe of ETFs by rate-of-change (ROC) over a configurable
lookback period, then buys the top N performers and sells any holdings
that fall out of the top N.  Supports absolute momentum filter (only
buy when ROC > 0) and dual momentum (absolute + relative).

Designed for monthly/quarterly rebalancing with a regime filter.
"""

import pandas as pd

from backtester.strategies.base import CrossSectionalStrategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import roc, sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("momentum_rotation")
class MomentumRotation(CrossSectionalStrategy):
    """Cross-sectional momentum rotation across asset-class ETFs.

    Parameters:
        roc_period (int): ROC lookback in trading days (default 126 ~6mo).
        top_n (int): Number of top assets to hold (default 3).
        abs_momentum (bool): If True, only buy when ROC > 0 (default True).
        sma_period (int): SMA trend filter period; 0 = disabled (default 200).
    """

    def __init__(self):
        super().__init__()
        self.roc_period = 126   # ~6 months of trading days
        self.top_n_count = 3
        self.abs_momentum = True   # absolute momentum gate
        self.sma_period = 200      # per-asset trend filter; 0 = off

    def configure(self, params: dict) -> None:
        self.roc_period = params.get("roc_period", self.roc_period)
        self.top_n_count = params.get("top_n", self.top_n_count)
        self.abs_momentum = params.get("abs_momentum", self.abs_momentum)
        self.sma_period = params.get("sma_period", self.sma_period)

    def compute_indicators(self, df: pd.DataFrame, timeframe_data=None) -> pd.DataFrame:
        df = df.copy()
        df["roc"] = roc(df["Close"], period=self.roc_period)
        if self.sma_period > 0:
            df["trend_sma"] = sma(df["Close"], period=self.sma_period)
        return df

    def rank_universe(
        self,
        bar_data: dict[str, pd.Series],
        positions: dict[str, Position],
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> list[tuple[str, SignalAction]]:
        """Rank all symbols by ROC and generate BUY/SELL signals."""
        # Score all symbols by their ROC value
        scores: dict[str, float] = {}
        for sym, row in bar_data.items():
            val = row.get("roc")
            if pd.notna(val):
                # Absolute momentum gate: skip negative momentum
                if self.abs_momentum and val <= 0:
                    continue
                # SMA trend filter: skip if price below SMA
                if self.sma_period > 0:
                    trend = row.get("trend_sma")
                    if pd.notna(trend) and row.get("Close", 0) < trend:
                        continue
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
        row: pd.Series,
        portfolio_state: PortfolioState,
        max_alloc_pct: float,
    ) -> int:
        """Return -1 for SELL (sell all), defer to sizer for BUY."""
        if action == SignalAction.SELL:
            return -1
        return 0  # Let the position sizer handle BUY sizing
