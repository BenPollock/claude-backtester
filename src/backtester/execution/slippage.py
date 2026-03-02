"""Slippage models for simulated order execution."""

from abc import ABC, abstractmethod

from backtester.portfolio.order import Order
from backtester.types import Side


class SlippageModel(ABC):
    @abstractmethod
    def compute(self, order: Order, fill_price: float, volume: float) -> float:
        """Return adjusted fill price after slippage."""
        ...


class FixedSlippage(SlippageModel):
    """Apply fixed basis points of slippage. BUY pays more, SELL receives less."""

    def __init__(self, bps: float = 10.0):
        self._multiplier = bps / 10_000.0

    def compute(self, order: Order, fill_price: float, volume: float) -> float:
        slip = fill_price * self._multiplier
        if order.side == Side.BUY:
            return fill_price + slip
        else:
            return fill_price - slip


class VolumeSlippage(SlippageModel):
    """Slippage proportional to order size vs. volume."""

    def __init__(self, impact_factor: float = 0.1):
        self._impact = impact_factor

    def compute(self, order: Order, fill_price: float, volume: float) -> float:
        if volume <= 0:
            return fill_price
        ratio = order.quantity / volume
        slip = fill_price * self._impact * ratio
        if order.side == Side.BUY:
            return fill_price + slip
        else:
            return fill_price - slip


class SqrtImpactSlippage(SlippageModel):
    """Square-root market impact model (Almgren-Chriss approximation).

    impact = impact_factor * sigma * sqrt(order_qty / volume) * fill_price
    BUY: fill_price + impact; SELL: fill_price - impact
    """

    def __init__(self, sigma: float = 0.02, impact_factor: float = 0.1):
        self._sigma = sigma
        self._impact_factor = impact_factor

    def compute(self, order: Order, fill_price: float, volume: float) -> float:
        if volume <= 0:
            return fill_price
        import math
        impact = (
            self._impact_factor
            * self._sigma
            * math.sqrt(order.quantity / volume)
            * fill_price
        )
        if order.side == Side.BUY:
            return fill_price + impact
        else:
            return fill_price - impact
