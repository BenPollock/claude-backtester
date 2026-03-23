"""Risk Regime strategy: allocate based on VIX, yield curve, credit spread, and intermarket signals."""

import pandas as pd

from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy
from backtester.strategies.indicators import sma
from backtester.types import SignalAction
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position


@register_strategy("risk_regime")
class RiskRegime(Strategy):
    """Long-only strategy that toggles risk-on/risk-off based on macro signals.

    Scores regime indicators: VIX term structure, yield curve spread,
    high-yield credit spread, copper/gold momentum, and SMA trend.
    Enters positions when all available signals agree on risk-on,
    exits when all signals agree on risk-off. Gracefully degrades
    when data columns are absent — NaN inputs are treated as neutral
    (not counted toward the score).

    Parameters:
        sma_period: Trend filter SMA period (default 200)
        vix_threshold: vix_ratio below this is risk-on (default 1.0, contango)
        yield_spread_threshold: 10y-2y spread above this is risk-on (default 0.0)
        credit_threshold: HY credit spread below this is risk-on (default 5.0)
        min_f_score: Minimum Piotroski F-Score for entry (default 5)
        use_trend: Include SMA trend in scoring (default True)
        use_intermarket: Include copper/gold momentum in scoring (default False)
    """

    def __init__(self):
        super().__init__()
        self.sma_period = 200
        self.vix_threshold = 1.0
        self.yield_spread_threshold = 0.0
        self.credit_threshold = 5.0
        self.min_f_score = 5
        self.use_trend = False
        self.use_intermarket = False
        self.use_pcr = False
        self.pcr_fear_threshold = 0.85  # PCR above this = fear = risk-off

    def configure(self, params: dict) -> None:
        self.sma_period = params.get("sma_period", self.sma_period)
        self.vix_threshold = params.get("vix_threshold", self.vix_threshold)
        self.yield_spread_threshold = params.get(
            "yield_spread_threshold", self.yield_spread_threshold
        )
        self.credit_threshold = params.get("credit_threshold", self.credit_threshold)
        self.min_f_score = params.get("min_f_score", self.min_f_score)
        self.use_trend = params.get("use_trend", self.use_trend)
        self.use_intermarket = params.get("use_intermarket", self.use_intermarket)
        self.use_pcr = params.get("use_pcr", self.use_pcr)
        self.pcr_fear_threshold = params.get("pcr_fear_threshold", self.pcr_fear_threshold)

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
        has_position = position is not None and position.total_quantity > 0

        # --- Score risk-on signals (each adds +1) ---
        score = 0
        counted = 0

        # 1. VIX term structure in contango (ratio < threshold)
        vix_ratio = row.get("vix_ratio")
        if vix_ratio is not None and not pd.isna(vix_ratio):
            counted += 1
            if vix_ratio < self.vix_threshold:
                score += 1

        # 2. Yield curve spread positive (no inversion)
        yield_spread = row.get("fred_yield_spread_10y2y")
        if yield_spread is not None and not pd.isna(yield_spread):
            counted += 1
            if yield_spread > self.yield_spread_threshold:
                score += 1

        # 3. Credit spreads below threshold (low stress)
        credit_spread = row.get("fred_credit_spread_hy")
        if credit_spread is not None and not pd.isna(credit_spread):
            counted += 1
            if credit_spread < self.credit_threshold:
                score += 1

        # 4. Put-call ratio below fear threshold (low fear = risk-on)
        if self.use_pcr:
            pcr_ma = row.get("sentiment_pcr_ma10")
            if pcr_ma is not None and not pd.isna(pcr_ma):
                counted += 1
                if pcr_ma < self.pcr_fear_threshold:
                    score += 1

        # 6. Copper/gold momentum positive (economic health improving)
        if self.use_intermarket:
            cu_au_mom = row.get("intermarket_cu_au_momentum")
            if cu_au_mom is not None and not pd.isna(cu_au_mom):
                counted += 1
                if cu_au_mom > 0:
                    score += 1

        # 7. Price above SMA trend
        if self.use_trend:
            sma_val = row.get("sma_trend")
            close = row.get("Close")
            if sma_val is not None and not pd.isna(sma_val):
                counted += 1
                if close > sma_val:
                    score += 1

        # If no macro data available, hold — can't determine regime
        if counted == 0:
            return SignalAction.HOLD

        # Risk-on: all available signals positive
        if score == counted:
            if not has_position:
                # Quality filter: prefer high F-Score stocks
                f_score = row.get("fund_piotroski_f")
                if f_score is not None and not pd.isna(f_score):
                    if f_score >= self.min_f_score:
                        return SignalAction.BUY
                else:
                    # No F-Score data — buy anyway in full risk-on
                    return SignalAction.BUY

        # Risk-off: no signals positive
        elif score == 0:
            if has_position:
                return SignalAction.SELL

        # Neutral: HOLD
        return SignalAction.HOLD
