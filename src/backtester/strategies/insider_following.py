"""Insider Following strategy: follow insider buying/selling signals from EDGAR data."""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("insider_following")
class InsiderFollowing(Strategy):
    """Long-only strategy that follows insider trading activity.

    Buys when corporate officers are actively buying shares, confirmed
    by price trend. Sells on heavy insider selling or trend breakdown.

    Parameters:
        min_officer_buys: Minimum officer buy transactions in 90 days (default 2)
        insider_buy_ratio_threshold: Minimum buy ratio in 90 days (default 0.7)
        sma_period: Trend filter SMA period (default 100)
        sell_on_heavy_selling: Whether to sell on heavy insider selling (default True)
        heavy_selling_threshold: Net shares threshold for sell signal (default -1000)
    """

    def __init__(self):
        super().__init__()
        self.min_officer_buys = 2
        self.insider_buy_ratio_threshold = 0.7
        self.sma_period = 100
        self.sell_on_heavy_selling = True
        self.heavy_selling_threshold = -1000

    def configure(self, params: dict) -> None:
        self.min_officer_buys = params.get("min_officer_buys", self.min_officer_buys)
        self.insider_buy_ratio_threshold = params.get(
            "insider_buy_ratio_threshold", self.insider_buy_ratio_threshold
        )
        self.sma_period = params.get("sma_period", self.sma_period)
        self.sell_on_heavy_selling = params.get(
            "sell_on_heavy_selling", self.sell_on_heavy_selling
        )
        self.heavy_selling_threshold = params.get(
            "heavy_selling_threshold", self.heavy_selling_threshold
        )

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

        # Graceful degradation: HOLD if insider columns are absent or NaN
        officer_buys = row.get("insider_officer_buys_90d")
        buy_ratio = row.get("insider_buy_ratio_90d")
        net_shares = row.get("insider_net_shares_30d")

        # Need at least one of officer_buys or buy_ratio to make a decision
        has_officer_buys = officer_buys is not None and not pd.isna(officer_buys)
        has_buy_ratio = buy_ratio is not None and not pd.isna(buy_ratio)

        if not has_officer_buys and not has_buy_ratio:
            return SignalAction.HOLD

        if not has_position:
            # BUY: insider buying signal + trend confirmation
            officer_buy_signal = (
                has_officer_buys and officer_buys >= self.min_officer_buys
            )
            ratio_signal = (
                has_buy_ratio and buy_ratio >= self.insider_buy_ratio_threshold
            )

            if (officer_buy_signal or ratio_signal) and above_sma:
                return SignalAction.BUY
        else:
            # SELL: heavy insider selling OR trend break
            has_net_shares = net_shares is not None and not pd.isna(net_shares)

            if (
                self.sell_on_heavy_selling
                and has_net_shares
                and net_shares < self.heavy_selling_threshold
            ):
                return SignalAction.SELL

            if not above_sma:
                return SignalAction.SELL

        return SignalAction.HOLD
