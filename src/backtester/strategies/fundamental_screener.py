"""Fundamental Screener strategy: flexible JSON rule-based screening on EDGAR data."""

import operator
import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


# Supported comparison operators
_OPS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@register_strategy("fundamental_screener")
class FundamentalScreener(Strategy):
    """Flexible rule-based fundamental screening strategy.

    Uses configurable buy/sell rules expressed as [column, operator, value]
    triples. Values can be numeric literals or string references to other
    columns/context values (e.g., "sma_trend").

    Parameters:
        sma_period: SMA period added as 'sma_trend' column (default 200)
        buy_rules: List of [column, operator, value] triples for buy signals.
            Default empty list means the strategy never buys.
        sell_rules: List of [column, operator, value] triples for sell signals.
            Default empty list means sell only on trend break (price < SMA).
    """

    def __init__(self):
        super().__init__()
        self.sma_period = 200
        self.buy_rules: list[list] = []
        self.sell_rules: list[list] = []

    def configure(self, params: dict) -> None:
        self.sma_period = params.get("sma_period", self.sma_period)
        self.buy_rules = params.get("buy_rules", self.buy_rules)
        self.sell_rules = params.get("sell_rules", self.sell_rules)

    def compute_indicators(
        self,
        df: pd.DataFrame,
        timeframe_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        df["sma_trend"] = sma(df["Close"], self.sma_period)
        return df

    def _evaluate_rules(self, rules: list[list], row: pd.Series) -> bool | None:
        """Evaluate a list of [column, operator, value] rules against a row.

        Returns True if ALL rules pass, False if any rule fails,
        or None if any required column is NaN (treat as failed).
        """
        if not rules:
            return None  # No rules defined

        for rule in rules:
            if len(rule) != 3:
                continue

            col, op_str, val = rule

            # Get the left-hand value from the row
            lhs = row.get(col)
            if lhs is None or pd.isna(lhs):
                return False  # Missing data → rule fails for safety

            # Get the comparison operator
            op_func = _OPS.get(op_str)
            if op_func is None:
                continue  # Unknown operator, skip

            # Resolve the right-hand value: string → context lookup, else numeric
            if isinstance(val, str):
                rhs = row.get(val)
                if rhs is None or pd.isna(rhs):
                    return False  # Missing context value → rule fails
            else:
                rhs = val

            try:
                if not op_func(float(lhs), float(rhs)):
                    return False
            except (TypeError, ValueError):
                return False

        return True

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

        if not has_position:
            buy_result = self._evaluate_rules(self.buy_rules, row)
            if buy_result is True:
                return SignalAction.BUY
        else:
            # Check sell rules
            sell_result = self._evaluate_rules(self.sell_rules, row)
            if sell_result is True:
                return SignalAction.SELL
            # Also sell on trend break if no sell rules defined
            if not self.sell_rules and not above_sma:
                return SignalAction.SELL

        return SignalAction.HOLD
