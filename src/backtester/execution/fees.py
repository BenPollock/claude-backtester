"""Fee models for simulated order execution."""

from abc import ABC, abstractmethod

from backtester.portfolio.order import Order


class FeeModel(ABC):
    @abstractmethod
    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        """Return commission/fee amount for this fill."""
        ...


class PerTradeFee(FeeModel):
    """Flat fee per trade regardless of size."""

    def __init__(self, fee: float = 0.0):
        self._fee = fee

    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        return self._fee
