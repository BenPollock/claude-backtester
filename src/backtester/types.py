"""Core enumerations used across the backtester."""

from enum import Enum, auto


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()


class OrderStatus(Enum):
    PENDING = auto()
    FILLED = auto()
    CANCELLED = auto()


class SignalAction(Enum):
    BUY = auto()
    SELL = auto()
    HOLD = auto()
