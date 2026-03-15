"""Smart Money strategy: combine institutional, insider, and fundamental signals."""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("smart_money")
class SmartMoney(Strategy):
    """Long-only strategy combining institutional ownership changes,
    insider activity, and fundamental quality signals.

    Buys when institutions are accumulating shares, optionally confirmed
    by insider buying, with positive fundamentals and trend support.

    Parameters:
        inst_growth_threshold: Min QoQ institutional share increase (default 0.05 = 5%)
        insider_confirmation: Require insider buy confirmation (default True)
        min_revenue_growth: Minimum YoY revenue growth (default 0.0)
        sma_period: Trend filter SMA period (default 200)
    """

    def __init__(self):
        super().__init__()
        self.inst_growth_threshold = 0.05
        self.insider_confirmation = True
        self.min_revenue_growth = 0.0
        self.sma_period = 200

    def configure(self, params: dict) -> None:
        self.inst_growth_threshold = params.get(
            "inst_growth_threshold", self.inst_growth_threshold
        )
        self.insider_confirmation = params.get(
            "insider_confirmation", self.insider_confirmation
        )
        self.min_revenue_growth = params.get(
            "min_revenue_growth", self.min_revenue_growth
        )
        self.sma_period = params.get("sma_period", self.sma_period)

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

        # Required: institutional ownership change
        inst_change = row.get("inst_shares_change_pct")
        if inst_change is None or pd.isna(inst_change):
            return SignalAction.HOLD

        # Required: fundamental data
        net_income = row.get("fund_net_income")
        rev_growth = row.get("fund_revenue_growth_yoy")
        if net_income is None or pd.isna(net_income):
            return SignalAction.HOLD
        if rev_growth is None or pd.isna(rev_growth):
            return SignalAction.HOLD

        # Optional insider data (only required if insider_confirmation is True)
        buy_ratio = row.get("insider_buy_ratio_90d")
        net_shares = row.get("insider_net_shares_30d")
        has_buy_ratio = buy_ratio is not None and not pd.isna(buy_ratio)
        has_net_shares = net_shares is not None and not pd.isna(net_shares)

        if not has_position:
            # Institutional accumulation
            if inst_change <= self.inst_growth_threshold:
                return SignalAction.HOLD

            # Insider confirmation (if required)
            if self.insider_confirmation:
                if not has_buy_ratio or buy_ratio <= 0.5:
                    return SignalAction.HOLD

            # Fundamental quality
            if net_income <= 0 or rev_growth < self.min_revenue_growth:
                return SignalAction.HOLD

            # Trend confirmation
            if not above_sma:
                return SignalAction.HOLD

            return SignalAction.BUY
        else:
            # SELL: institutional + insider exit, OR fundamentals deteriorate,
            # OR trend breaks
            inst_and_insider_exit = (
                inst_change < -self.inst_growth_threshold
                and has_net_shares
                and net_shares < 0
            )
            fundamentals_bad = net_income <= 0
            trend_broken = not above_sma

            if inst_and_insider_exit or fundamentals_bad or trend_broken:
                return SignalAction.SELL

        return SignalAction.HOLD
