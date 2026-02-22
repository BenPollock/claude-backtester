"""Pluggable position sizing models."""

from abc import ABC, abstractmethod

import pandas as pd


class PositionSizer(ABC):
    """Base class for position sizing strategies."""

    @abstractmethod
    def compute(
        self,
        symbol: str,
        price: float,
        row: pd.Series,
        equity: float,
        cash: float,
        max_alloc_pct: float,
    ) -> int:
        """Return number of shares to buy.

        Args:
            symbol: Ticker symbol
            price: Current close price
            row: Today's OHLCV + indicator data
            equity: Total portfolio equity
            cash: Available cash
            max_alloc_pct: Max allocation per position (from config)
        """
        ...


class FixedFractional(PositionSizer):
    """Default: allocate max_alloc_pct of equity per position (existing behavior)."""

    def compute(self, symbol, price, row, equity, cash, max_alloc_pct) -> int:
        if price <= 0:
            return 0
        target = min(equity * max_alloc_pct, cash)
        return int(target // price)


class ATRSizer(PositionSizer):
    """Risk a fixed fraction of equity per trade, sized by ATR.

    shares = (equity * risk_pct) / (atr_value * atr_multiple)

    E.g., risk 1% of equity, with a 2-ATR stop: risk_pct=0.01, atr_multiple=2.0
    """

    def __init__(self, risk_pct: float = 0.01, atr_multiple: float = 2.0,
                 atr_column: str = "ATR"):
        self._risk_pct = risk_pct
        self._atr_multiple = atr_multiple
        self._atr_col = atr_column

    def compute(self, symbol, price, row, equity, cash, max_alloc_pct) -> int:
        if price <= 0:
            return 0
        atr_val = row.get(self._atr_col)
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
            # Fallback to fixed fractional if ATR not available
            target = min(equity * max_alloc_pct, cash)
            return int(target // price)
        risk_per_share = atr_val * self._atr_multiple
        if risk_per_share <= 0:
            return 0
        shares = int((equity * self._risk_pct) / risk_per_share)
        # Cap by max allocation and available cash
        max_shares = int(min(equity * max_alloc_pct, cash) // price)
        return min(shares, max_shares)


class VolatilityParity(PositionSizer):
    """Weight positions inversely proportional to realized volatility.

    target_weight = (1 / vol) / sum(1 / vol_all)
    Simplified: size = (equity * target_vol) / (vol * price)

    For single-position sizing, uses target_vol as the annualized vol
    budget per position.
    """

    def __init__(self, target_vol: float = 0.10, lookback: int = 20):
        self._target_vol = target_vol
        self._lookback = lookback

    def compute(self, symbol, price, row, equity, cash, max_alloc_pct) -> int:
        if price <= 0:
            return 0
        # Use ATR as a proxy for daily vol, annualize
        atr_val = row.get("ATR")
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0 or price <= 0:
            target = min(equity * max_alloc_pct, cash)
            return int(target // price)
        daily_vol = atr_val / price
        annual_vol = daily_vol * (252 ** 0.5)
        if annual_vol <= 0:
            return 0
        target_value = equity * (self._target_vol / annual_vol)
        target_value = min(target_value, equity * max_alloc_pct, cash)
        return int(target_value // price)
