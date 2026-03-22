"""Macro-aware value strategy: tightens quality thresholds in contractions.

Uses FRED macro indicators (yield spread, credit spread) to determine the
macro regime, then applies stricter or looser F-Score / P/E filters
accordingly.  Always excludes distressed stocks (Altman Z < threshold)
and requires price above an SMA trend line for entry.
"""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("macro_aware_value")
class MacroAwareValue(Strategy):
    """Long-only value strategy that adapts to the macro regime.

    Parameters:
        sma_period: Trend filter SMA period (default 200)
        expansion_min_f: Min Piotroski F-Score in expansion (default 5)
        contraction_min_f: Min Piotroski F-Score in contraction (default 7)
        expansion_max_pe: Max P/E ratio in expansion (default 20.0)
        contraction_max_pe: Max P/E ratio in contraction (default 15.0)
        min_z_score: Minimum Altman Z-Score to avoid distress (default 1.8)
        credit_threshold: Max high-yield credit spread for expansion (default 5.0)
    """

    def __init__(self):
        super().__init__()
        self.sma_period = 200
        self.expansion_min_f = 5
        self.contraction_min_f = 7
        self.expansion_max_pe = 20.0
        self.contraction_max_pe = 15.0
        self.min_z_score = 1.8
        self.credit_threshold = 5.0

    def configure(self, params: dict) -> None:
        self.sma_period = params.get("sma_period", self.sma_period)
        self.expansion_min_f = params.get("expansion_min_f", self.expansion_min_f)
        self.contraction_min_f = params.get("contraction_min_f", self.contraction_min_f)
        self.expansion_max_pe = params.get("expansion_max_pe", self.expansion_max_pe)
        self.contraction_max_pe = params.get("contraction_max_pe", self.contraction_max_pe)
        self.min_z_score = params.get("min_z_score", self.min_z_score)
        self.credit_threshold = params.get("credit_threshold", self.credit_threshold)

    def compute_indicators(
        self,
        df: pd.DataFrame,
        timeframe_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        df["sma_trend"] = sma(df["Close"], self.sma_period)
        return df

    def _is_expansion(self, row: pd.Series) -> bool | None:
        """Determine macro regime from FRED indicators.

        Returns True for expansion, False for contraction, None if
        the required data is unavailable (graceful degradation).
        """
        yield_spread = row.get("fred_yield_spread_10y2y")
        credit_spread = row.get("fred_credit_spread_hy")

        if yield_spread is None or pd.isna(yield_spread):
            return None
        if credit_spread is None or pd.isna(credit_spread):
            return None

        return yield_spread > 0 and credit_spread < self.credit_threshold

    def generate_signals(
        self,
        symbol: str,
        row: pd.Series,
        position: Position | None,
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> SignalAction:
        # Trend filter must be available
        sma_val = row.get("sma_trend")
        if sma_val is None or pd.isna(sma_val):
            return SignalAction.HOLD

        close = row["Close"]
        above_sma = close > sma_val
        has_position = position is not None and position.total_quantity > 0

        # Required fundamental columns
        f_score = row.get("fund_piotroski_f")
        pe_ratio = row.get("fund_pe_ratio")
        z_score = row.get("fund_altman_z")

        # Graceful degradation: if any required column is missing, HOLD
        if any(
            v is None or pd.isna(v) for v in [f_score, pe_ratio, z_score]
        ):
            return SignalAction.HOLD

        # Determine regime-dependent thresholds
        expansion = self._is_expansion(row)
        if expansion is None:
            # No macro data — fall back to contraction (conservative)
            min_f = self.contraction_min_f
            max_pe = self.contraction_max_pe
        elif expansion:
            min_f = self.expansion_min_f
            max_pe = self.expansion_max_pe
        else:
            min_f = self.contraction_min_f
            max_pe = self.contraction_max_pe

        if not has_position:
            # BUY: quality + value + no distress + trend confirmation
            if (
                f_score >= min_f
                and pe_ratio < max_pe
                and z_score >= self.min_z_score
                and above_sma
            ):
                return SignalAction.BUY
        else:
            # SELL: quality deterioration OR distress OR trend broken
            if (
                f_score < 3
                or z_score < self.min_z_score
                or not above_sma
            ):
                return SignalAction.SELL

        return SignalAction.HOLD
