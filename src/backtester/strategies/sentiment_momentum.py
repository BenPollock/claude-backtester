"""Sentiment Momentum strategy: combine analyst revisions, insider activity, and price trend."""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("sentiment_momentum")
class SentimentMomentum(Strategy):
    """Long-only strategy combining analyst revision breadth, insider buying,
    and price momentum via SMA.

    Counts bullish/bearish signals from multiple sentiment sources and
    trades when enough signals align. Gracefully degrades when data columns
    are missing — NaN inputs are treated as neutral (not counted).

    Parameters:
        sma_period: Trend filter SMA period (default 50)
        min_signals_buy: Minimum bullish signals to trigger BUY (default 2)
        min_signals_sell: Minimum bearish signals to trigger SELL (default 2)
        insider_buy_threshold: insider_buy_ratio_90d above this is bullish (default 0.5)
        insider_sell_threshold: insider_buy_ratio_90d below this is bearish (default 0.3)
    """

    def __init__(self):
        super().__init__()
        self.sma_period = 50
        self.min_signals_buy = 2
        self.min_signals_sell = 2
        self.insider_buy_threshold = 0.5
        self.insider_sell_threshold = 0.3

    def configure(self, params: dict) -> None:
        self.sma_period = params.get("sma_period", self.sma_period)
        self.min_signals_buy = params.get("min_signals_buy", self.min_signals_buy)
        self.min_signals_sell = params.get("min_signals_sell", self.min_signals_sell)
        self.insider_buy_threshold = params.get(
            "insider_buy_threshold", self.insider_buy_threshold
        )
        self.insider_sell_threshold = params.get(
            "insider_sell_threshold", self.insider_sell_threshold
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

        # --- Count bullish signals ---
        bullish = 0

        # 1. Analyst revision breadth > 0
        rev_breadth = row.get("analyst_rev_breadth")
        if rev_breadth is not None and not pd.isna(rev_breadth):
            if rev_breadth > 0:
                bullish += 1

        # 2. Insider buy ratio above threshold
        buy_ratio = row.get("insider_buy_ratio_90d")
        if buy_ratio is not None and not pd.isna(buy_ratio):
            if buy_ratio > self.insider_buy_threshold:
                bullish += 1

        # 3. Close above SMA
        if close > sma_val:
            bullish += 1

        # --- Count bearish signals ---
        bearish = 0

        # 1. Analyst revision breadth < 0
        if rev_breadth is not None and not pd.isna(rev_breadth):
            if rev_breadth < 0:
                bearish += 1

        # 2. Insider buy ratio below sell threshold
        if buy_ratio is not None and not pd.isna(buy_ratio):
            if buy_ratio < self.insider_sell_threshold:
                bearish += 1

        # 3. Close below SMA
        if close < sma_val:
            bearish += 1

        if not has_position:
            if bullish >= self.min_signals_buy:
                return SignalAction.BUY
        else:
            if bearish >= self.min_signals_sell:
                return SignalAction.SELL

        return SignalAction.HOLD
