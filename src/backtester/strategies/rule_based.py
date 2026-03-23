"""Rule-based DSL strategy: define indicators and buy/sell conditions via params JSON."""

import math
import operator
from datetime import date

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies import indicators as ind
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position

# Maps indicator name → (function, takes_df_not_series)
INDICATOR_REGISTRY = {
    "sma": (ind.sma, False),
    "ema": (ind.ema, False),
    "rsi": (ind.rsi, False),
    "atr": (ind.atr, True),
    "macd": (ind.macd, False),
    "bollinger": (ind.bollinger, False),
    "stochastic": (ind.stochastic, True),
    "adx": (ind.adx, True),
    "obv": (ind.obv, True),
    "keltner": (ind.keltner, True),
    "donchian": (ind.donchian, True),
    "williams_r": (ind.williams_r, True),
    "cci": (ind.cci, True),
    "mfi": (ind.mfi, True),
    "roc": (ind.roc, False),
    "psar": (ind.psar, True),
    "ichimoku": (ind.ichimoku, True),
    "vwap": (ind.vwap, True),
}

OPERATORS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


def _apply_indicator(df: pd.DataFrame, col_name: str, spec: dict) -> pd.DataFrame:
    """Apply a single indicator spec to the DataFrame, adding column(s)."""
    fn_name = spec.get("fn", col_name)
    if fn_name not in INDICATOR_REGISTRY:
        raise ValueError(f"Unknown indicator function '{fn_name}'. "
                         f"Available: {sorted(INDICATOR_REGISTRY)}")

    func, takes_df = INDICATOR_REGISTRY[fn_name]
    params = {k: v for k, v in spec.items() if k != "fn"}
    source = df if takes_df else df["Close"]

    result = func(source, **params)

    if fn_name == "macd":
        macd_line, signal_line, histogram = result
        df[f"{col_name}_line"] = macd_line
        df[f"{col_name}_signal"] = signal_line
        df[f"{col_name}_hist"] = histogram
    elif fn_name in ("bollinger", "keltner", "donchian"):
        upper, middle, lower = result
        df[f"{col_name}_upper"] = upper
        df[f"{col_name}_middle"] = middle
        df[f"{col_name}_lower"] = lower
    elif fn_name == "stochastic":
        k, d = result
        df[f"{col_name}_k"] = k
        df[f"{col_name}_d"] = d
    elif fn_name == "ichimoku":
        tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou = result
        df[f"{col_name}_tenkan"] = tenkan_sen
        df[f"{col_name}_kijun"] = kijun_sen
        df[f"{col_name}_senkou_a"] = senkou_a
        df[f"{col_name}_senkou_b"] = senkou_b
        df[f"{col_name}_chikou"] = chikou
    else:
        df[col_name] = result

    return df


def _validate_rules(rules: list, label: str) -> None:
    """Validate that rules are well-formed [left, op, right] triples."""
    for i, rule in enumerate(rules):
        if not isinstance(rule, (list, tuple)) or len(rule) != 3:
            raise ValueError(f"{label}[{i}]: must be a [left, operator, right] triple, "
                             f"got {rule!r}")
        _, op, _ = rule
        if op not in OPERATORS:
            raise ValueError(f"{label}[{i}]: unknown operator '{op}'. "
                             f"Allowed: {sorted(OPERATORS)}")


def _evaluate_rules(rules: list, context: dict) -> bool:
    """Evaluate all rules against a context dict. All must pass (AND logic)."""
    for left_key, op, right in rules:
        left_val = context.get(left_key)
        if left_val is None or (isinstance(left_val, float) and math.isnan(left_val)):
            return False

        if isinstance(right, str):
            right_val = context.get(right)
            if right_val is None or (isinstance(right_val, float) and math.isnan(right_val)):
                return False
        else:
            right_val = right

        if not OPERATORS[op](left_val, right_val):
            return False

    return True


@register_strategy("rule_based")
class RuleBasedStrategy(Strategy):
    """Configurable rule-based strategy.

    Params JSON keys:
        indicators: dict of {col_name: {fn, ...params}} for ticker data
        benchmark_indicators: dict of {col_name: {fn, ...params}} for benchmark
        buy_when: list of [left, op, right] rules (AND logic)
        sell_when: list of [left, op, right] rules (AND logic)
        short_when: list of [left, op, right] rules (AND logic) — open short
        cover_when: list of [left, op, right] rules (AND logic) — close short
        max_hold_days: int — force exit after N calendar days (optional)
    """

    def __init__(self):
        super().__init__()
        self._indicator_specs: dict = {}
        self._benchmark_specs: dict = {}
        self._buy_rules: list = []
        self._sell_rules: list = []
        self._short_rules: list = []
        self._cover_rules: list = []
        self._max_hold_days: int | None = None

    def configure(self, params: dict) -> None:
        self._indicator_specs = params.get("indicators", {})
        self._benchmark_specs = params.get("benchmark_indicators", {})
        self._buy_rules = params.get("buy_when", [])
        self._sell_rules = params.get("sell_when", [])
        self._short_rules = params.get("short_when", [])
        self._cover_rules = params.get("cover_when", [])
        self._max_hold_days = params.get("max_hold_days")

        # Validate indicator function names
        for col_name, spec in {**self._indicator_specs, **self._benchmark_specs}.items():
            fn_name = spec.get("fn", col_name)
            if fn_name not in INDICATOR_REGISTRY:
                raise ValueError(f"Unknown indicator function '{fn_name}'. "
                                 f"Available: {sorted(INDICATOR_REGISTRY)}")

        _validate_rules(self._buy_rules, "buy_when")
        _validate_rules(self._sell_rules, "sell_when")
        _validate_rules(self._short_rules, "short_when")
        _validate_rules(self._cover_rules, "cover_when")

    def compute_indicators(
        self,
        df: pd.DataFrame,
        timeframe_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        for col_name, spec in self._indicator_specs.items():
            df = _apply_indicator(df, col_name, spec)
        return df

    def compute_benchmark_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._benchmark_specs:
            return df
        df = df.copy()
        for col_name, spec in self._benchmark_specs.items():
            df = _apply_indicator(df, col_name, spec)
        return df

    def generate_signals(
        self,
        symbol: str,
        row: pd.Series,
        position: Position | None,
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> SignalAction:
        # Build unified context: ticker columns + bm_-prefixed benchmark columns
        context = row.to_dict()
        if benchmark_row is not None:
            for key, val in benchmark_row.items():
                context[f"bm_{key}"] = val

        has_long = position is not None and position.total_quantity > 0
        has_short = position is not None and position.is_short

        # Time-based exit: force sell/cover after max_hold_days
        if self._max_hold_days is not None and (has_long or has_short):
            current_date = pd.Timestamp(row.name).date() if hasattr(row.name, 'date') else row.name
            oldest_entry = min(lot.entry_date for lot in position.lots)
            if isinstance(oldest_entry, pd.Timestamp):
                oldest_entry = oldest_entry.date()
            days_held = (current_date - oldest_entry).days
            if days_held >= self._max_hold_days:
                return SignalAction.COVER if has_short else SignalAction.SELL

        # Close existing positions first
        if has_long and self._sell_rules:
            if _evaluate_rules(self._sell_rules, context):
                return SignalAction.SELL

        if has_short and self._cover_rules:
            if _evaluate_rules(self._cover_rules, context):
                return SignalAction.COVER

        # Open new positions only when flat
        if not has_long and not has_short:
            if self._buy_rules and _evaluate_rules(self._buy_rules, context):
                return SignalAction.BUY
            if self._short_rules and _evaluate_rules(self._short_rules, context):
                return SignalAction.SHORT

        return SignalAction.HOLD
