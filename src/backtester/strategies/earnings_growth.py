"""Earnings Growth strategy: buy companies with accelerating earnings and revenue growth."""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("earnings_growth")
class EarningsGrowth(Strategy):
    """Long-only growth momentum strategy.

    Buys stocks with strong earnings and revenue growth, confirmed by
    price trend, and sells when growth turns negative or trend breaks.

    Parameters:
        min_earnings_growth: Minimum YoY earnings growth (default 0.10 = 10%)
        min_revenue_growth: Minimum YoY revenue growth (default 0.05 = 5%)
        sma_period: Trend filter SMA period (default 50)
        max_pe: Maximum P/E to avoid extreme overvaluation (default 50.0)
    """

    def __init__(self):
        super().__init__()
        self.min_earnings_growth = 0.10
        self.min_revenue_growth = 0.05
        self.sma_period = 50
        self.max_pe = 50.0

    def configure(self, params: dict) -> None:
        self.min_earnings_growth = params.get("min_earnings_growth", self.min_earnings_growth)
        self.min_revenue_growth = params.get("min_revenue_growth", self.min_revenue_growth)
        self.sma_period = params.get("sma_period", self.sma_period)
        self.max_pe = params.get("max_pe", self.max_pe)

    def compute_indicators(
        self,
        df: pd.DataFrame,
        timeframe_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        df["sma_trend"] = sma(df["Close"], self.sma_period)
        return df

    def generate_signals(
        self,
        symbol: str,
        row: pd.Series,
        position: Position | None,
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> SignalAction:
        sma_val = row.get("sma_trend")
        if pd.isna(sma_val):
            return SignalAction.HOLD

        has_position = position is not None and position.total_quantity > 0
        close = row["Close"]
        above_sma = close > sma_val

        # Graceful degradation: HOLD if EDGAR columns are absent or NaN
        earnings_growth = row.get("fund_earnings_growth_yoy")
        rev_growth = row.get("fund_revenue_growth_yoy")
        pe_ratio = row.get("fund_pe_ratio")

        if earnings_growth is None or pd.isna(earnings_growth):
            return SignalAction.HOLD
        if rev_growth is None or pd.isna(rev_growth):
            return SignalAction.HOLD
        if pe_ratio is None or pd.isna(pe_ratio):
            return SignalAction.HOLD

        if not has_position:
            # BUY: strong earnings + revenue growth, trend up, valuation not extreme
            if (
                earnings_growth >= self.min_earnings_growth
                and rev_growth >= self.min_revenue_growth
                and above_sma
                and pe_ratio < self.max_pe
            ):
                return SignalAction.BUY
        else:
            # SELL: earnings growth turns negative OR trend breaks
            if earnings_growth < 0 or not above_sma:
                return SignalAction.SELL

        return SignalAction.HOLD
