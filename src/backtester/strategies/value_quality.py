"""Value + Quality strategy: buy undervalued, profitable companies with trend confirmation."""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("value_quality")
class ValueQuality(Strategy):
    """Long-only value + quality strategy with SMA trend filter.

    Buys stocks that are profitable, reasonably valued, growing revenue,
    have manageable leverage, and are trading above their SMA trend line.

    Parameters:
        sma_period: Trend filter SMA period (default 200)
        max_pe: Maximum P/E ratio for entry (default 25.0)
        min_pe: Minimum P/E ratio — filters out suspiciously cheap (default 5.0)
        min_revenue_growth: Minimum YoY revenue growth (default 0.0)
        max_debt_to_equity: Maximum debt-to-equity ratio (default 2.0)
        min_net_margin: Minimum net margin (default 0.0)
    """

    def __init__(self):
        super().__init__()
        self.sma_period = 200
        self.max_pe = 25.0
        self.min_pe = 5.0
        self.min_revenue_growth = 0.0
        self.max_debt_to_equity = 2.0
        self.min_net_margin = 0.0

    def configure(self, params: dict) -> None:
        self.sma_period = params.get("sma_period", self.sma_period)
        self.max_pe = params.get("max_pe", self.max_pe)
        self.min_pe = params.get("min_pe", self.min_pe)
        self.min_revenue_growth = params.get("min_revenue_growth", self.min_revenue_growth)
        self.max_debt_to_equity = params.get("max_debt_to_equity", self.max_debt_to_equity)
        self.min_net_margin = params.get("min_net_margin", self.min_net_margin)

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

        # Check if fundamental columns are present
        fund_cols = [
            "fund_net_income", "fund_pe_ratio", "fund_revenue_growth_yoy",
            "fund_debt_to_equity", "fund_net_margin",
        ]
        for col in fund_cols:
            val = row.get(col)
            if val is None or pd.isna(val):
                return SignalAction.HOLD

        net_income = row["fund_net_income"]
        pe_ratio = row["fund_pe_ratio"]
        rev_growth = row["fund_revenue_growth_yoy"]
        debt_eq = row["fund_debt_to_equity"]
        net_margin = row["fund_net_margin"]

        if not has_position:
            # BUY: profitable, reasonable valuation, growing, low leverage,
            # good margins, price above SMA
            if (
                net_income > 0
                and self.min_pe < pe_ratio < self.max_pe
                and rev_growth >= self.min_revenue_growth
                and debt_eq <= self.max_debt_to_equity
                and net_margin >= self.min_net_margin
                and above_sma
            ):
                return SignalAction.BUY
        else:
            # SELL: unprofitable OR valuation extreme OR trend broken
            if (
                net_income <= 0
                or pe_ratio > self.max_pe * 1.5
                or not above_sma
            ):
                return SignalAction.SELL

        return SignalAction.HOLD
